from keras.layers.pooling import GlobalAveragePooling1D
from tensorflow.python.keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Wrapper
from keras.layers import GaussianNoise
from keras.regularizers import l2, l1
from keras.layers.merge import add
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Model
from keras.callbacks import *
from keras import optimizers
from keras.layers import *
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from keras.layers import Flatten, Dense, Reshape
from keras.layers import Concatenate

from keras.initializers import Ones, Zeros
import tensorflow as tf

from keras_contrib.applications.resnet import ResNet18

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from scipy import stats

import preprocessing as pre
import pickle
import dill
import os

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


class LayerNormalization(Layer):
    # Code from: https://github.com/keras-team/keras/issues/3878
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        # Creates the layer weights
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


def cnn_basic(input_shape, dropout=0.25,
              activation="softmax",
              loss="categorical_crossentropy",
              metrics="accuracy"
              ):
    # shape = (batch size, p1, p2, channel)

    #
    # Input
    #
    # input_shape = tuple([p1, p2, 1])
    x_input = Input(shape=input_shape)  # 256, 128, 1
    print("Input Shape:", x_input.shape)    # (?, 128, 256, 1)

    # Block 1
    x_cnn = Conv2D(filters=32, kernel_size=(11, 11), strides=(4, 4), padding="same", kernel_initializer="he_uniform")(x_input)
    print(x_cnn.shape, 10)
    # x_cnn = MaxPooling2D(pool_size=[2, 2], strides=1)(x_cnn)
    x_cnn = BatchNormalization()(x_cnn)
    x_cnn = Activation("relu")(x_cnn)
    print(x_cnn.shape, 11)

    x_cnn = Conv2D(filters=32, kernel_size=(11, 11), strides=(4, 4), padding="same", kernel_initializer="he_uniform")(x_cnn)
    print(x_cnn.shape, 20)
    x_cnn = MaxPooling2D(pool_size=[2, 2], strides=1)(x_cnn)
    x_cnn = BatchNormalization()(x_cnn)
    x_cnn = Activation("relu")(x_cnn)
    print(x_cnn.shape, 21)

    # Block 2
    x_cnn = Conv2D(filters=32, kernel_size=(7, 7), strides=(2, 2), padding="same", kernel_initializer="he_uniform")(x_input)
    print(x_cnn.shape, 10)
    # x_cnn = MaxPooling2D(pool_size=[2, 2], strides=1)(x_cnn)
    x_cnn = BatchNormalization()(x_cnn)
    x_cnn = Activation("relu")(x_cnn)
    print(x_cnn.shape, 11)

    x_cnn = Conv2D(filters=32, kernel_size=(7, 7), strides=(2, 2), padding="same", kernel_initializer="he_uniform")(x_cnn)
    print(x_cnn.shape, 20)
    x_cnn = MaxPooling2D(pool_size=[2, 2], strides=1)(x_cnn)
    x_cnn = BatchNormalization()(x_cnn)
    x_cnn = Activation("relu")(x_cnn)
    print(x_cnn.shape, 21)

    # Block 3
    x_cnn = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer="he_uniform")(x_cnn)
    print(x_cnn.shape, 30)
    x_cnn = MaxPooling2D(pool_size=[2, 2], strides=1)(x_cnn)
    x_cnn = BatchNormalization()(x_cnn)
    x_cnn = Activation("relu")(x_cnn)
    print(x_cnn.shape, 31)

    # Flatten
    x_cnn = Flatten()(x_cnn)
    print(x_cnn.shape, 4)

    # Dense + Dropout
    z = Dense(64, activation="relu")(x_cnn)
    z = Dropout(dropout)(z)
    print(z.shape, 5)
    z = Dense(23, activation="relu")(z)
    z = Dropout(dropout)(z)
    print(z.shape, 6)
    z = Dense(5, activation="relu")(z)
    print(z.shape, 7)

    model = Model(inputs=x_input, outputs=z)
    model.compile(loss=loss, optimizer=Adam(lr=0.0001), metrics=[metrics])
    return model


if __name__ == "__main__":
    #
    # Path
    #
    os.getcwd()
    path = ""
    os.chdir(path)

    #
    # Data
    #

    """ plot 
    os.getcwd()
    for idx in range(10, 20):
        plt.imshow(x[idx])
        plt.title("target: {}".format(y[idx]))
        plt.savefig("./figures/test {}".format(idx))
        plt.close()
    """

    #
    # Train
    #
    n, p1, p2, channel = x.shape
    input_shape = tuple([p1, p2, 1])

    cross_validation_k = 5
    cv_data = list(StratifiedKFold(n_splits=cross_validation_k, shuffle=True, random_state=2019).split(x, y))

    y_pred_arr = list()
    y_arr = list()

    y_one_hot = to_categorical(y)

    for idx, (idxs_train, idxs_validation) in tqdm(enumerate(cv_data)):
        # init
        K.clear_session()

        x_train, y_train = x[idxs_train], y_one_hot[idxs_train]
        x_validation, y_validation = x[idxs_validation], y_one_hot[idxs_validation]

        # model
        # input_shape = tuple([p1, p2, 1])
        model = cnn_basic(input_shape, dropout=0.1,
                          activation="softmax",
                          loss="categorical_crossentropy",
                          metrics="accuracy"
                          )

        # train
        batch_size = 50
        early_stop = EarlyStopping(patience=5, verbose=1, monitor="val_acc")
        check_point = ModelCheckpoint('./model_saved/cnn_basic_{}'.format(idx),
                                      save_best_only=True,
                                      save_weights_only=True,  # 'save_weights_only' is required to avoid error
                                      verbose=1,
                                      monitor="val_acc", mode='max')
        model.fit(x_train, y_train, batch_size=batch_size, epochs=50,
                  validation_data=[x_validation, y_validation], callbacks=[early_stop, check_point])

        scores_train_set = model.evaluate(x_train, y_train, batch_size=batch_size, verbose=1)
        print(model.metrics_names, "\n", scores_train_set)
        scores_validation_set = model.evaluate(x_validation, y_validation, batch_size=batch_size, verbose=1)
        print(model.metrics_names, "\n", scores_validation_set)

        # predict
        model.load_weights('./model_saved/cnn_basic_{}'.format(idx))
        y_pred_arr.append(model.predict(x_validation, batch_size=512))
        y_arr.append(y_validation)

    y_pred_arr = np.concatenate(y_pred_arr)
    y_pred_arr = pre.flat_list(y_pred_arr)
    y_arr = np.concatenate(y_arr)
