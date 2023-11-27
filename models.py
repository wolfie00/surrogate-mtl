import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf


def linear_baseline(regression):
    linear_clf = tf.keras.Sequential([
        tf.keras.layers.Dense(units=1, activation=None if regression else 'sigmoid')
    ], name='clf_keras')
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
