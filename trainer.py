import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
import tensorflow_addons as tfa
import keras_tuner as kt
from sklearn.linear_model import LinearRegression, LogisticRegression

from models import mlp_tunable
from functools import partial


class Trainer:
    def __init__(self, model, surrogate, x_train, y_train, regression):
        self.model = model if model is not None else None
        # if isinstance(model, tf.keras.Model):
        #     self.model = tf.keras.models.clone_model(model)
        self.surrogate = surrogate
        self.x_train = x_train
        self.y_train = y_train
        self.regression = regression

    def train_surrogate(self, black_box):
        y_train_surrogate = black_box.predict(self.x_train, verbose=1).flatten()
        if self.surrogate is None:
            self.surrogate = LinearRegression() if self.regression else LogisticRegression()
        self.surrogate.fit(self.x_train, y_train_surrogate)
        return self.surrogate

    def train(self, epochs=100, tune=False, es_patience=3, verbose=1,
              pl_patience=1):
        loss = tf.keras.losses.LogCosh() if self.regression else tf.keras.losses.BinaryCrossentropy()
        optimizer = tf.keras.optimizers.Adam()
        metrics = ['mean_absolute_error', 'mean_squared_error',
                   tfa.metrics.RSquare()] if self.regression else ['binary_accuracy',
                                                                   tfa.metrics.F1Score(num_classes=1,
                                                                                       average='macro',
                                                                                       threshold=0.5)]

        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', verbose=verbose,
                                                      patience=es_patience,
                                                      mode='min'),
                     tf.keras.callbacks.ReduceLROnPlateau(patience=pl_patience, verbose=verbose,
                                                          monitor='val_loss',
                                                          mode='min')]

        if not tune:
            self.model.compile(optimizer=optimizer, loss=loss,
                               metrics=metrics)

            self.model.fit(x=self.x_train, y=self.y_train,
                           epochs=epochs, verbose=verbose, validation_split=0.1,
                           callbacks=callbacks)
        else:
            metric = 'val_r_square' if self.regression else 'val_f1_score'
            tuner = kt.BayesianOptimization(
                hypermodel=partial(mlp_tunable, regression=self.regression),
                objective=kt.Objective(metric, direction='max'),
                max_trials=15,
                # max_epochs=100,
                executions_per_trial=1,
                # overwrite=True,
                # directory='results',
                # project_name='...',
            )
            print(tuner.search_space_summary())

            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                              mode='min',
                                                              patience=es_patience, verbose=verbose)
            tuner.search(x=self.x_train, y=self.y_train,
                         epochs=epochs, validation_split=0.1,
                         callbacks=[early_stopping], verbose=1)

            self.model = tuner.get_best_models()[0]

        return self.model  # return trained model instance
