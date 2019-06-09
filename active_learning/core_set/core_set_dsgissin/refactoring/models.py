# https://github.com/dsgissin/DiscriminativeActiveLearning/blob/master/models.py

import numpy as np

from keras.callbacks import Callback
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, Input, UpSampling2D
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
from keras import regularizers
from keras import backend as K
from keras.models import load_model
from keras.utils import to_categorical, multi_gpu_model


def get_discriminative_model(input_shape):
    """
    The MLP model for discriminative active learning,
    without any regularization techniques
    """

    if np.sum(input_shape) < 30:
        nb_nodes = 20
    else:
        nb_nodes = 256

    x_input = Input(input_shape)
    x = Flatten()(x_input)
    x = Dense(units=nb_nodes, activation="relu")(x)
    x = Dense(units=nb_nodes, activation="relu")(x)
    x = Dense(units=nb_nodes, activation="relu")(x)
    z = Dense(units=2, activation="softmax", name="softmax")(x)

    model = Model(inputs=x_input, outputs=z)

    return model


def get_LeNet_model(input_shape, nb_labels=10):
    """
    A LeNet model
    """

    x_input = Input(shape=input_shape)

    x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x_input)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)

    x = Dense(units=128, activation="relu", name="embedding")(x)
    x = Dropout(0.5)(x)
    z = Dense(nb_labels, activation="softmax", name="softmax")(x)

    model = Model(inputs=x_input, outputs=z)
    return model


def get_VGG_model(input_shape, nb_labels=10):
    """
    A VGG model
    """
    # init
    weight_decay = 0.0005   # kernel_regularizer

    x_input = Input(shape=input_shape)

    # 1.
    x = Conv2D(filters=64, kernel_size=(3, 3), padding="same",
               kernel_regularizer=regularizers.l2(weight_decay))(x_input)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), padding="same",
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # 2.
    x = Conv2D(filters=128, kernel_size=(3, 3), padding="same",
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(filters=128, kernel_size=(3, 3), padding="same",
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # 3.
    x = Conv2D(filters=256, kernel_size=(3, 3), padding="same",
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(filters=256, kernel_size=(3, 3), padding="same",
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(filters=256, kernel_size=(3, 3), padding="same",
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # 4.
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same',
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same',
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same',
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # 5.
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same',
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same',
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same',
               kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.4)(x)

    x = Flatten()(x)

    x = Dense(units=512, kernel_regularizer=regularizers.l2(weight_decay),
              name='embedding')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    z = Dense(units=nb_labels, activation="softmax", name="softmax")(x)

    model = Model(inputs=x_input, outputs=z)
    return model


class DelayedModelCheckpoint(Callback):
    """
    A custom callback for saving the model
    each time the validation accuracy improves.

    The custom part is that
    we save the model
    when the accuracy stays the same as well,
    and also that we start saving
    only after a certain amount of iterations to save time
    """

    def __init__(self, filepath, monitor="val_acc",
                 delay=50, verbose=0, weights=False):
        super(DelayedModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.delay = delay
        if self.monitor == "val_acc":
            self.best = -np.inf
        else:
            self.best = np.inf
        self.weights = weights

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if self.monitor == "val_acc":
            current = logs.get(self.monitor)
            # delay to save until the predefined number of iterations, self.delay
            if current >= self.best and epoch > self.delay:
                if self.verbose > 0:
                    print("\nEpoch %05d: %s improved from %0.5f to %0.5f,"
                          " saving model to %s"
                          % (epoch, self.monitor, self.best,
                             current, self.filepath))
                self.best = current
                if self.weights:
                    self.model.save_weights(self.filepath, overwrite=True)
                else:
                    self.model.save(self.filepath, overwrite=True)


class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):    # Todo: init 에 밑줄 뜨는데 괜찮나 ... ?
        pmodel = multi_gpu_model(ser_model, gpus,
                                 cpu_relocation=False,
                                 cpu_merge=False)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)


def train_LeNet_model(X_train, Y_train, X_validation, Y_validation,
                      checkpoint_path,
                      input_shape=None,
                      nb_labels=10, gpu=1):
    """
    A function that trains and returns a LeNet model
    on the labeled MNIST data
    """

    if input_shape is None:
        if K.image_data_format() == "channels_last":
            input_shape = (28, 28, 1)   # mnist
        else:
            input_shape = (1, 28, 28)   # mnist

    # todo: Y_train.shape
    model = get_LeNet_model(input_shape=input_shape, nb_labels=nb_labels)
    optimizer = optimizers.Adam()
    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=['accuracy'])
    callbacks = [DelayedModelCheckpoint(filepath=checkpoint_path, verbose=1, weights=True)]

    if gpu > 1:
        gpu_model = ModelMGPU(model, gpus=gpu)
        gpu_model.compile(loss="categorical_crossentroy",
                          optimizer=optimizer,
                          metrics=['accuracy'])
        gpu_model.fit(X_train, Y_train,
                      epochs=150,
                      batch_size=32,
                      shuffle=True,
                      validation_data=(X_validation, Y_validation),
                      callbacks=callbacks,
                      verbose=2)

        del model
        del gpu_model

        model = get_LeNet_model(input_shape=input_shape, labels=10)
        model.compile(loss="categorical_crossentropy",
                      optimizer=optimizer,
                      metric=['accuracy'])
        model.load_weights(checkpoint_path)
        return model
    else:
        # Todo: error on
        #  check_array_length_consistency(x, y, sample_weights)
        """
        X_train.shape   # (80, 28, 28, 1)
        Y_train.shape   # (84, 10)
        """
        model.fit(X_train, Y_train,
                  epochs=150,
                  batch_size=32,
                  shuffle=True,
                  validation_data=(X_validation, Y_validation),
                  callbacks=callbacks,
                  verbose=2)
        model.load_weights(checkpoint_path)
        return model
































