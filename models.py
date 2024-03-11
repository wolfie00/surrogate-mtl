import os

import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
import tensorflow_addons as tfa


def linear_baseline(regression):
    linear_clf = tf.keras.Sequential([
        tf.keras.layers.Dense(units=1, activation=None if regression else 'sigmoid')
    ])
    return linear_clf


def mlp_baseline(regression):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=512, activation='relu'))
    model.add(tf.keras.layers.Dense(units=256, activation='relu'))
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dense(units=1, activation=None if regression else 'sigmoid'))
    return model


def mlp_tunable(hp, regression=True):
    model = tf.keras.models.Sequential()
    for i in range(hp.Int('num_layers', 1, 5)):
        model.add(
            tf.keras.layers.Dense(
                # Tune number of units separately.
                units=hp.Choice(f'units_{i}', values=[4, 16, 32, 64, 128,
                                                      256, 512]),
                activation='relu',
            ))
        if hp.Boolean('batch_norm'):
            model.add(tf.keras.layers.BatchNormalization())
        if hp.Boolean('dropout'):
            model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Dense(units=1, name='clf_head',
                                    activation=None if regression else 'sigmoid'))
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.LogCosh() if regression else tf.keras.losses.BinaryCrossentropy(),
                  metrics=['mean_absolute_error', 'mean_squared_error',
                           tfa.metrics.RSquare()] if regression else ['binary_accuracy',
                                                                      tfa.metrics.F1Score(num_classes=1,
                                                                                          average='macro',
                                                                                          threshold=0.5)])
    return model


class SurrogateMTL(tf.keras.Model):
    def __init__(self, black_box, regression, alpha=0.5, *args, **kwargs):
        super().__init__(**kwargs)
        self.black_box = black_box
        self.explainer = tf.keras.Sequential([
            tf.keras.layers.Dense(units=1, activation=None if regression else 'sigmoid')
        ], name='explainer')
        self.alpha = alpha
        self.train_loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.val_loss_tracker = tf.keras.metrics.Mean(name='val_loss')
        self.train_clf_loss_tracker = tf.keras.metrics.Mean(name='clf_loss')
        self.val_clf_loss_tracker = tf.keras.metrics.Mean(name='val_clf_loss')
        self.train_xai_loss_tracker = tf.keras.metrics.Mean(name='xai_loss')
        self.val_xai_loss_tracker = tf.keras.metrics.Mean(name='val_xai_loss')

        self.train_score_tracker = tf.keras.metrics.MeanSquaredError(
            name='mse'
        ) if regression else tf.keras.metrics.BinaryAccuracy(name='binary_accuracy')
        self.val_score_tracker = tf.keras.metrics.MeanSquaredError(
            name='val_mse'
        ) if regression else tf.keras.metrics.BinaryAccuracy(name='val_binary_accuracy')

    @staticmethod
    def point_fidelity(f, g):
        x = (g - f) ** 2  # batch size
        return tf.math.reduce_mean(x)  # average batch fidelity --> lower is better

    def call(self, inputs, training=None, **kwargs):
        pass

    @property
    def metrics(self):
        return [self.train_loss_tracker, self.val_loss_tracker,
                self.train_score_tracker, self.val_score_tracker,
                self.train_clf_loss_tracker, self.val_clf_loss_tracker,
                self.train_xai_loss_tracker, self.val_xai_loss_tracker,
                # self.train_r2_tracker, self.val_r2_tracker
                ]
        # + [m for m in self.compiled_metrics]

    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            z = self.black_box(x)
            x_int = self.explainer(x)

            fidelity = self.point_fidelity(tf.squeeze(z), tf.squeeze(x_int))
            # tf.print('\nbatch fidelity:', fidelity)
            clf_loss = self.compiled_loss(y, tf.squeeze(z))
            # tf.print('\nbatch logcosh:', reg_loss)
            loss = self.alpha * clf_loss + (1. - self.alpha) * fidelity

        trainable_parameters = (self.black_box.trainable_variables +
                                self.explainer.trainable_variables
                                )

        gradients = tape.gradient(loss, trainable_parameters)

        self.optimizer.apply_gradients(zip(gradients, trainable_parameters))

        self.train_loss_tracker.update_state(loss)
        self.train_score_tracker.update_state(y, tf.squeeze(z))
        self.train_clf_loss_tracker.update_state(clf_loss)
        self.train_xai_loss_tracker.update_state(fidelity)
        # self.train_r2_tracker.update_state(y, tf.squeeze(z))
        return {
            'loss': self.train_loss_tracker.result(),
            'bb_metric': self.train_score_tracker.result(),
            'clf_loss': self.train_clf_loss_tracker.result(),
            'fidelity': self.train_xai_loss_tracker.result(),
            # 'bb_r2': self.train_r2_tracker.result()
        }

    @tf.function
    def test_step(self, data):
        x, y = data
        z = self.black_box(x, training=False)
        x_int = self.explainer(x, training=False)
        fidelity = self.point_fidelity(tf.squeeze(z), tf.squeeze(x_int))
        clf_loss = self.compiled_loss(y, tf.squeeze(z))
        loss = self.alpha * clf_loss + (1. - self.alpha) * fidelity
        # loss = reg_loss + self.alpha * fidelity
        self.val_loss_tracker.update_state(loss)
        self.val_score_tracker.update_state(y, tf.squeeze(z))
        self.val_clf_loss_tracker.update_state(clf_loss)
        self.val_xai_loss_tracker.update_state(fidelity)
        # self.val_r2_tracker.update_state(y, tf.squeeze(z))
        return {
            'loss': self.val_loss_tracker.result(),
            'bb_metric': self.val_score_tracker.result(),
            'clf_loss': self.val_clf_loss_tracker.result(),
            'fidelity': self.val_xai_loss_tracker.result(),
            # 'bb_r2': self.val_r2_tracker.result()
        }

    @tf.function
    def predict_step(self, data):
        return self.black_box(data, training=False)


class SurrogateMTLDecoupled(tf.keras.Model):
    def __init__(self, black_box, regression, **kwargs):
        super().__init__(**kwargs)
        self.black_box = black_box
        self.explainer = tf.keras.Sequential([
            tf.keras.layers.Dense(units=1, activation=None if regression else 'sigmoid')
        ], name='explainer')
        # self.train_loss_tracker = tf.keras.metrics.Mean(name='loss')
        # self.val_loss_tracker = tf.keras.metrics.Mean(name='val_loss')
        self.train_clf_loss_tracker = tf.keras.metrics.Mean(name='clf_loss')
        self.val_clf_loss_tracker = tf.keras.metrics.Mean(name='val_clf_loss')
        self.train_xai_loss_tracker = tf.keras.metrics.Mean(name='xai_loss')
        self.val_xai_loss_tracker = tf.keras.metrics.Mean(name='val_xai_loss')

        self.train_score_tracker = tf.keras.metrics.MeanSquaredError(
            name='mse'
        ) if regression else tf.keras.metrics.BinaryAccuracy(name='binary_accuracy')
        self.val_score_tracker = tf.keras.metrics.MeanSquaredError(
            name='val_mse'
        ) if regression else tf.keras.metrics.BinaryAccuracy(name='val_binary_accuracy')

        self.g_optimizer = tf.keras.optimizers.Adam()

    @staticmethod
    def point_fidelity(f, g):
        x = (g - f) ** 2  # batch size
        return tf.math.reduce_mean(x)  # average batch fidelity --> lower is better

    def call(self, inputs, training=None, **kwargs):
        pass

    @property
    def metrics(self):
        return [
            # self.train_loss_tracker, self.val_loss_tracker,
            self.train_score_tracker, self.val_score_tracker,
            self.train_clf_loss_tracker, self.val_clf_loss_tracker,
            self.train_xai_loss_tracker, self.val_xai_loss_tracker,
            # self.train_r2_tracker, self.val_r2_tracker
        ]
        # + [m for m in self.compiled_metrics]

    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as f_tape:
            z = self.black_box(x)
            # x_int = self.explainer(x)

            # fidelity = self.point_fidelity(tf.squeeze(z), tf.squeeze(x_int))
            clf_loss = self.compiled_loss(y, tf.squeeze(z))

        with tf.GradientTape() as g_tape:
            # z = self.black_box(x)
            x_int = self.explainer(x)

            fidelity = self.point_fidelity(tf.squeeze(z), tf.squeeze(x_int))
            # clf_loss = self.compiled_loss(y, tf.squeeze(z))

        f_parameters = self.black_box.trainable_variables
        g_parameters = self.explainer.trainable_variables

        f_gradients = f_tape.gradient(clf_loss, f_parameters)
        g_gradients = g_tape.gradient(fidelity, g_parameters)

        self.optimizer.apply_gradients(zip(f_gradients, f_parameters))
        self.g_optimizer.apply_gradients(zip(g_gradients, g_parameters))

        # self.train_loss_tracker.update_state(loss)
        self.train_score_tracker.update_state(y, tf.squeeze(z))
        self.train_clf_loss_tracker.update_state(clf_loss)
        self.train_xai_loss_tracker.update_state(fidelity)
        return {
            # 'loss': self.train_loss_tracker.result(),
            'bb_metric': self.train_score_tracker.result(),
            'clf_loss': self.train_clf_loss_tracker.result(),
            'fidelity': self.train_xai_loss_tracker.result(),
        }

    @tf.function
    def test_step(self, data):
        x, y = data
        z = self.black_box(x, training=False)
        x_int = self.explainer(x, training=False)
        fidelity = self.point_fidelity(tf.squeeze(z), tf.squeeze(x_int))
        clf_loss = self.compiled_loss(y, tf.squeeze(z))
        # loss = self.alpha * clf_loss + (1. - self.alpha) * fidelity
        # loss = reg_loss + self.alpha * fidelity
        # self.val_loss_tracker.update_state(loss)
        self.val_score_tracker.update_state(y, tf.squeeze(z))
        self.val_clf_loss_tracker.update_state(clf_loss)
        self.val_xai_loss_tracker.update_state(fidelity)
        # self.val_r2_tracker.update_state(y, tf.squeeze(z))
        return {
            # 'loss': self.val_loss_tracker.result(),
            'bb_metric': self.val_score_tracker.result(),
            'clf_loss': self.val_clf_loss_tracker.result(),
            'fidelity': self.val_xai_loss_tracker.result(),
            # 'bb_r2': self.val_r2_tracker.result()
        }

    @tf.function
    def predict_step(self, data):
        return self.black_box(data, training=False)


class SurrogateMTLDecoupledV2(tf.keras.Model):
    def __init__(self, black_box, regression, alpha, **kwargs):
        super().__init__(**kwargs)

        self.alpha = alpha
        self.black_box = black_box
        self.explainer = tf.keras.Sequential([
            tf.keras.layers.Dense(units=1, activation=None if regression else 'sigmoid')
        ], name='explainer')
        self.train_loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.val_loss_tracker = tf.keras.metrics.Mean(name='val_loss')
        self.train_clf_loss_tracker = tf.keras.metrics.Mean(name='clf_loss')
        self.val_clf_loss_tracker = tf.keras.metrics.Mean(name='val_clf_loss')
        self.train_xai_loss_tracker = tf.keras.metrics.Mean(name='xai_loss')
        self.val_xai_loss_tracker = tf.keras.metrics.Mean(name='val_xai_loss')

        self.train_score_tracker = tf.keras.metrics.MeanSquaredError(
            name='mse'
        ) if regression else tf.keras.metrics.BinaryAccuracy(name='binary_accuracy')
        self.val_score_tracker = tf.keras.metrics.MeanSquaredError(
            name='val_mse'
        ) if regression else tf.keras.metrics.BinaryAccuracy(name='val_binary_accuracy')

        self.g_optimizer = tf.keras.optimizers.Adam()

    @staticmethod
    def point_fidelity(f, g):
        x = (g - f) ** 2  # batch size
        return tf.math.reduce_mean(x)  # average batch fidelity --> lower is better

    def call(self, inputs, training=None, **kwargs):
        pass

    @property
    def metrics(self):
        return [
            self.train_loss_tracker, self.val_loss_tracker,
            self.train_score_tracker, self.val_score_tracker,
            self.train_clf_loss_tracker, self.val_clf_loss_tracker,
            self.train_xai_loss_tracker, self.val_xai_loss_tracker,
            # self.train_r2_tracker, self.val_r2_tracker
        ]
        # + [m for m in self.compiled_metrics]

    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape(persistent=True) as tape:
            z = self.black_box(x)
            x_int = self.explainer(x)

            fidelity = self.point_fidelity(tf.squeeze(z), tf.squeeze(x_int))
            clf_loss = self.compiled_loss(y, tf.squeeze(z))
            loss_mtl = self.alpha * clf_loss + (1. - self.alpha) * fidelity

        f_parameters = self.black_box.trainable_variables
        g_parameters = self.explainer.trainable_variables

        f_gradients = tape.gradient(clf_loss, f_parameters)
        g_gradients = tape.gradient(loss_mtl, g_parameters)

        self.optimizer.apply_gradients(zip(f_gradients, f_parameters))
        self.g_optimizer.apply_gradients(zip(g_gradients, g_parameters))

        self.train_loss_tracker.update_state(loss_mtl)
        self.train_score_tracker.update_state(y, tf.squeeze(z))
        self.train_clf_loss_tracker.update_state(clf_loss)
        self.train_xai_loss_tracker.update_state(fidelity)

        del tape

        return {
            'loss': self.train_loss_tracker.result(),
            'bb_metric': self.train_score_tracker.result(),
            'clf_loss': self.train_clf_loss_tracker.result(),
            'fidelity': self.train_xai_loss_tracker.result(),
        }

    @tf.function
    def test_step(self, data):
        x, y = data
        z = self.black_box(x, training=False)
        x_int = self.explainer(x, training=False)
        fidelity = self.point_fidelity(tf.squeeze(z), tf.squeeze(x_int))
        clf_loss = self.compiled_loss(y, tf.squeeze(z))
        loss_mtl = self.alpha * clf_loss + (1. - self.alpha) * fidelity
        # loss = reg_loss + self.alpha * fidelity
        self.val_loss_tracker.update_state(loss_mtl)
        self.val_score_tracker.update_state(y, tf.squeeze(z))
        self.val_clf_loss_tracker.update_state(clf_loss)
        self.val_xai_loss_tracker.update_state(fidelity)
        # self.val_r2_tracker.update_state(y, tf.squeeze(z))
        return {
            'loss': self.val_loss_tracker.result(),
            'bb_metric': self.val_score_tracker.result(),
            'clf_loss': self.val_clf_loss_tracker.result(),
            'fidelity': self.val_xai_loss_tracker.result(),
            # 'bb_r2': self.val_r2_tracker.result()
        }

    @tf.function
    def predict_step(self, data):
        return self.black_box(data, training=False)


class SurrogateMTLRegularized(tf.keras.Model):
    def __init__(self, black_box, regression, lambda_, alpha=0.5, **kwargs):
        super().__init__(**kwargs)

        self.alpha = alpha
        self.lambda_ = lambda_
        self.f: tf.keras.Model = black_box
        self.f_prime = tf.keras.models.clone_model(self.f)
        # for l1, l2 in zip(self.f.layers, self.f_prime.layers):
        #     l2.set_weights(l1.get_weights())
        self.f_prime.set_weights(self.f.get_weights())

        self.g = tf.keras.Sequential([
            tf.keras.layers.Dense(units=1, activation=None if regression else 'sigmoid')
        ], name='explainer')

        self.train_loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.val_loss_tracker = tf.keras.metrics.Mean(name='val_loss')
        self.train_clf_loss_tracker = tf.keras.metrics.Mean(name='clf_loss')
        self.val_clf_loss_tracker = tf.keras.metrics.Mean(name='val_clf_loss')
        self.train_xai_loss_tracker = tf.keras.metrics.Mean(name='xai_loss')
        self.val_xai_loss_tracker = tf.keras.metrics.Mean(name='val_xai_loss')

        self.train_score_tracker = tf.keras.metrics.MeanSquaredError(
            name='mse'
        ) if regression else tf.keras.metrics.BinaryAccuracy(name='binary_accuracy')
        self.val_score_tracker = tf.keras.metrics.MeanSquaredError(
            name='val_mse'
        ) if regression else tf.keras.metrics.BinaryAccuracy(name='val_binary_accuracy')

    def check_init_perf(self, data, labels, regression):
        print(f'{"MSE" if regression else "Accuracy"} of f*:',
              mean_squared_error(np.array(labels).flatten(),
                                 self.f.predict(data, verbose=0).flatten()) if regression else
              accuracy_score(labels, self.f.predict(data, verbose=0).flatten())
              )
        print(f"{'MSE' if regression else 'Accuracy'} of f':",
              mean_squared_error(np.array(labels).flatten(),
                                 self.f_prime.predict(data, verbose=0).flatten()) if regression else
              accuracy_score(labels, self.f_prime.predict(data, verbose=0).flatten())
              )
        for l1, l2 in zip(self.f.layers, self.f_prime.layers):
            for p1, p2 in zip(l1.get_weights(), l2.get_weights()):
                assert np.allclose(p1, p2)

    @staticmethod
    def point_fidelity(f, g):
        x = (g - f) ** 2  # batch size
        return tf.math.reduce_mean(x)  # average batch fidelity --> lower is better

    def call(self, inputs, training=None, **kwargs):
        pass

    @property
    def metrics(self):
        return [
            self.train_loss_tracker, self.val_loss_tracker,
            self.train_score_tracker, self.val_score_tracker,
            self.train_clf_loss_tracker, self.val_clf_loss_tracker,
            self.train_xai_loss_tracker, self.val_xai_loss_tracker,
            # self.train_r2_tracker, self.val_r2_tracker
        ]
        # + [m for m in self.compiled_metrics]

    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            z = self.f_prime(x)
            x_int = self.g(x)

            fidelity = self.point_fidelity(tf.squeeze(z), tf.squeeze(x_int))
            clf_loss = self.compiled_loss(y, tf.squeeze(z))
            # loss_mtl = \
            #     self.alpha * clf_loss + (1. - self.alpha) * fidelity + self.lambda_ * tf.keras.losses.KLDivergence()(
            #         self.f(x), z
            #     )
            # d = tf.keras.losses.KLDivergence()(tf.expand_dims(self.f(x), axis=-1), tf.expand_dims(z, axis=-1))
            # d = self.alpha * tf.keras.losses.MeanSquaredError()(self.f(x), z) + (1. - self.alpha) * clf_loss
            d = tf.keras.losses.MeanSquaredError()(self.f(x), z)
            # loss_dist = fidelity + self.lambda_ * d
            loss_dist = 0.5 * (clf_loss + d) + 0.5 * fidelity
            # loss_dist = 0.5 * d + 0.5 * fidelity
            # loss_mtl = fidelity + self.lambda_ * (tf.keras.losses.KLDivergence()(self.f(x), z) + clf_loss)

        parameters = self.f_prime.trainable_variables + self.g.trainable_variables

        gradients = tape.gradient(loss_dist, parameters)

        self.optimizer.apply_gradients(zip(gradients, parameters))

        self.train_loss_tracker.update_state(loss_dist)
        self.train_score_tracker.update_state(y, tf.squeeze(z))
        self.train_clf_loss_tracker.update_state(clf_loss)
        self.train_xai_loss_tracker.update_state(fidelity)

        return {
            'loss': self.train_loss_tracker.result(),
            'bb_metric': self.train_score_tracker.result(),
            'clf_loss': self.train_clf_loss_tracker.result(),
            'fidelity': self.train_xai_loss_tracker.result(),
        }

    @tf.function
    def test_step(self, data):
        x, y = data
        z = self.f_prime(x, training=False)
        x_int = self.g(x, training=False)
        fidelity = self.point_fidelity(tf.squeeze(z), tf.squeeze(x_int))
        clf_loss = self.compiled_loss(y, tf.squeeze(z))
        # d = tf.keras.losses.KLDivergence()(tf.expand_dims(self.f(x), axis=-1), tf.expand_dims(z, axis=-1))
        # d = self.alpha * tf.keras.losses.MeanSquaredError()(self.f(x), z) + (1. - self.alpha) * clf_loss
        d = tf.keras.losses.MeanSquaredError()(self.f(x), z)
        # loss_dist = fidelity + self.lambda_ * d
        loss_dist = 0.5 * (clf_loss + d) + 0.5 * fidelity
        # loss_dist = 0.5 * d + 0.5 * fidelity
        self.val_loss_tracker.update_state(loss_dist)
        self.val_score_tracker.update_state(y, tf.squeeze(z))
        self.val_clf_loss_tracker.update_state(clf_loss)
        self.val_xai_loss_tracker.update_state(fidelity)
        # self.val_r2_tracker.update_state(y, tf.squeeze(z))
        return {
            'loss': self.val_loss_tracker.result(),
            'bb_metric': self.val_score_tracker.result(),
            'clf_loss': self.val_clf_loss_tracker.result(),
            'fidelity': self.val_xai_loss_tracker.result(),
            # 'bb_r2': self.val_r2_tracker.result()
        }

    @tf.function
    def predict_step(self, data):
        return self.f(data, training=False)
