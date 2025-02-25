        #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 09:27:45 2024

@author: kmesman
"""
from Utils import flatten_chain, sequential_pop

from tensorflow.dtypes import float32
import matplotlib.pyplot as plt
#from tensorflow import keras
import keras
from keras import callbacks
import tensorflow as tf
import numpy as np
import random
import copy
import os

@keras.saving.register_keras_serializable()
def unit_loss(y_true, y_predict):
    y_true = tf.multiply(y_true, np.pi*2)
    y_predict = tf.multiply(y_predict, np.pi*2)

    cos_t = tf.cos(y_true)
    cos_p = tf.cos(y_predict)

    sin_t = tf.sin(y_true)
    sin_p = tf.sin(y_predict)

    squared_dist = tf.add(
        tf.square(tf.subtract(cos_t, cos_p)), tf.square(tf.subtract(sin_t, sin_p))
    )
    return tf.reduce_mean(squared_dist)

class NeuralNetwork:
    def __init__(self):
        self.hparams = {
            "NUM_UNITS": 30,
            "NUM_LAYERS": 1,
            "DROPOUT": 0.3,
            "LOSS": unit_loss ,
            "OPTIMIZER": "adam",
            "VALIDATE_SPLIT": 0.2,
            "EPOCHS": 200,
        }
        self.metric_accuracy = "mae"
        tf.config.run_functions_eagerly(
            True
            )


    """
    @keras.saving.register_keras_serializable()
    def unit_loss(self, y_true, y_predict):

        y_true = tf.add(tf.multiply(y_true, self.range), self.offset)
        y_predict = tf.add(tf.multiply(y_predict, self.range), self.offset)

        cos_t = tf.cos(y_true)
        cos_p = tf.cos(y_predict)

        sin_t = tf.sin(y_true)
        sin_p = tf.sin(y_predict)

        squared_dist = tf.add(
            tf.square(tf.subtract(cos_t, cos_p)), tf.square(tf.subtract(sin_t, sin_p))
        )
        return tf.reduce_mean(squared_dist)
    """

    
    def cosine_loss(self, y_true, y_predict):
 

        y_true = tf.add(tf.multiply(y_true, self.range), self.offset)
        y_predict = tf.add(tf.multiply(y_predict, self.range), self.offset)



        cos_t =  tf.cos(y_true)
        cos_p = tf.cos(y_predict)



        sin_t =  tf.sin(y_true)
        sin_p = tf.sin(y_predict)

        a = tf.concat([cos_t, sin_t], 1)
        a = tf.reshape(a, (24, 2))
        b = tf.concat([cos_p, sin_p], 1)
        b = tf.reshape(b, (2, 24))
        
        
        dot = tf.tensordot(a, b, 1)
        
        #ai = tf.make_tensor_proto(a)
        #bi = tf.make_tensor_proto(b)
        
        #a = tf.make_ndarray(a)
        #b = tf.make_ndarray(b)
        #a_b_dot = []
        
        
        
        tf.print(dot)
        #for i, j in zip(a, b):
        #    dot = np.dot(a, b)
        #    a_b_dot.append(1-dot)

        #dot = tf.reduce_sum(tf.multiply(a, b))
        a_dist = tf.add(
            1., tf.multiply(dot, -1)
        )
        return tf.reduce_mean(a_dist)
        
        
        
        
    @keras.saving.register_keras_serializable()
    def normalize(self, params):
        fy = flatten_chain(params)
        fy = [f-2*np.pi if f>2*np.pi else f for f in fy ]
        fy = [f+2*np.pi if f<0 else f for f in fy ]

        #print(fy)
        #self.offset = min(fy)
        #self.range = max(fy) - min(fy)
        
        self.offset=0
        self.range=2*np.pi
        
        print("range : {}".format(self.range))
        print("offset : {}".format(self.offset))
        for a in range(len(params)):
            for k in range(len(params[0])):
                params[a][k] = (params[a][k] - self.offset) / self.range
        params = np.array(params)
        return params

    def invert_normalize(self, params):
        iparams = [a * self.range + self.offset for a in params]
        return iparams
    """
    @tf.function
    def make_model(self):
        dropout_layer = keras.layers.Dropout(self.hparams["DROPOUT"])
        layers = [
            tf.keras.layers.Normalization(),
            keras.layers.Dense(
                self.hparams["NUM_UNITS"],
                activation=tf.nn.relu,
                input_shape=self.input_shape,
            ),
        ]

        for i in range(self.hparams["NUM_LAYERS"]):
            layers.append(dropout_layer)
            layers.append(
                tf.keras.layers.Dense(
                    units=self.hparams["NUM_UNITS"], activation="relu"
                )
            )

        layers.append(dropout_layer)
        layers.append(
            tf.keras.layers.Dense(
                units=self.output_shape,
                activation="sigmoid",
                kernel_initializer="normal",
            )
        )
        self.model = tf.keras.Sequential(layers)
        """
 

    def DefineLayers(self):

        self.dropout_layer = keras.layers.Dropout(self.hparams["DROPOUT"])
        self.norm_layer =  tf.keras.layers.Normalization()
        self.dense_layer = tf.keras.layers.Dense(
                units=self.hparams["NUM_UNITS"], activation="relu"
            )
        self.input_layer = keras.layers.Dense(
            self.hparams["NUM_UNITS"],
            activation=tf.nn.relu,
            input_shape=self.input_shape,
        )
        self.output_layer = tf.keras.layers.Dense(
            units=self.output_shape,
            activation="sigmoid",
            kernel_initializer="normal",
        )
            
        
    def make_model(self):
        #self.input_shape = (None,1)
        model = [self.norm_layer]
        model.append(self.input_layer)
        #model = [self.input_layer]
        for i in range(self.hparams["NUM_LAYERS"]):
            #model.append(self.dropout_layer)
            model.append(self.dense_layer)
        model.append(self.dropout_layer)
        model.append(self.output_layer)
        self.model = tf.keras.Sequential(model)
        self.model.build(self.input_shape)
        print(self.model.summary())
        
    @tf.function
    def inner_compile(self, optimizer):
        self.model.compile(
            optimizer=optimizer,
            loss=self.hparams["LOSS"],
            metrics=["mean_absolute_error"],  # mean_squared_error
        )
        
    def fit_model(self, x_train, y_train, x_validate, y_validate, cp_callback):
        print("okay")
        print(len(x_train[0]), len(y_train[0]), len(x_validate[0]), len(y_validate[0]))
        history = self.model.fit(
            x_train,
            y_train,
            validation_data=(x_validate, y_validate),
            #batch_size=self.hparams["BATCH"],
            epochs=self.hparams["EPOCHS"],
            callbacks=[cp_callback],
        )
        print("fit")
        return history

    #@tf.function
    def compile_model(self, data):
        x_train, y_train, x_validate, y_validate = data
        self.input_shape = (None, x_validate.shape[1])
        self.output_shape = y_train.shape[1]

        self.DefineLayers()
        self.make_model()
        checkpoint_path = "training_logs/cp.weights.h5"
        checkpoint_dir = os.path.dirname(checkpoint_path)

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, save_weights_only=True, verbose=1
        )
        
        earlystopping = callbacks.EarlyStopping(monitor="val_loss",
                                                mode="min",
                                                patience=60,
                                                restore_best_weights=True)

        # Create an optimizer with the learning rate schedule
        learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.hparams["LEARNING RATE"], decay_steps=10000, decay_rate=0.96
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedule)
        self.inner_compile(optimizer)
        history = self.fit_model(x_train, y_train, x_validate, y_validate, cp_callback)

        print(self.model.summary())
        _, accuracy = self.model.evaluate(
            x_validate, y_validate
        )

        return accuracy, history
    
    def train_model(self, run_dir, data):
        with tf.summary.create_file_writer(run_dir).as_default():
            accuracy, his = self.compile_model(data)
            tf.summary.scalar(self.metric_accuracy, accuracy, step=1)
            return his


    def run(self, x_in, y_in, hparams={}):
        for key, val in hparams.items():
            self.hparams[key] = val

        x = copy.deepcopy(x_in)
        y = copy.deepcopy(y_in)
        y = self.normalize(y)

        xy = [[xi, yi] for xi, yi in zip(x, y)]

        # TODO: make hparam datasplit
        sample_size = round(len(x) * (1 - self.hparams["VALIDATE_SPLIT"]))
        indices = random.sample(range(len(xy)), sample_size)
        xy_train = [xy[k] for k in indices]

        x_train = np.array([xi[0] for xi in xy_train])
        self.x_train = x_train
        y_train = np.array([xi[1] for xi in xy_train])

        xy_val = sequential_pop(xy, indices)
        x_validate = np.array([xi[0] for xi in xy_val])
        y_validate = np.array([yi[1] for yi in xy_val])
        ##############################

        data = [x_train, y_train, x_validate, y_validate]
        print(data)
        self.history = self.train_model("logs/hparam_tuning/" + "test", data)
        self.plot_history()

    def plot_history(self):
        # summarize history for accuracy
        plt.plot(self.history.history["val_loss"])
        plt.plot(self.history.history["loss"])
        plt.title("model accuracy")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.yscale("log")
        plt.legend(["validation", "loss"], loc="upper left")
        plt.show()

    def load_model(self, file):
        self.model = tf.keras.models.load_model(self.filepath)
        return
