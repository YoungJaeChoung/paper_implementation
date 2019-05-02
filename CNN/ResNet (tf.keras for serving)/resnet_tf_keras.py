from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

import tensorflow as tf
import numpy as np
import copy
import os


def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if tf.keras.backend.image_data_format() == 'channels_last':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3


def _bn_relu(x, bn_name=None, relu_name=None):
    """Helper to build a BN -> relu block
    """
    norm = tf.keras.layers.BatchNormalization(axis=CHANNEL_AXIS, name=bn_name)(x)
    return tf.keras.layers.Activation("relu", name=relu_name)(norm)


def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu residual unit activation function.
       This is the original ResNet v1 scheme in https://arxiv.org/abs/1512.03385
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    dilation_rate = conv_params.setdefault("dilation_rate", (1, 1))
    conv_name = conv_params.setdefault("conv_name", None)
    bn_name = conv_params.setdefault("bn_name", None)
    relu_name = conv_params.setdefault("relu_name", None)
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", tf.keras.regularizers.l2(1.e-4))

    def f(x):
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                   strides=strides, padding=padding,
                                   dilation_rate=dilation_rate,
                                   kernel_initializer=kernel_initializer,
                                   kernel_regularizer=kernel_regularizer,
                                   name=conv_name)(x)
        return _bn_relu(x, bn_name=bn_name, relu_name=relu_name)

    return f


def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv residual unit with full pre-activation
    function. This is the ResNet v2 scheme proposed in
    http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    dilation_rate = conv_params.setdefault("dilation_rate", (1, 1))
    conv_name = conv_params.setdefault("conv_name", None)
    bn_name = conv_params.setdefault("bn_name", None)
    relu_name = conv_params.setdefault("relu_name", None)
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", tf.keras.regularizers.l2(1.e-4))

    def f(x):
        activation = _bn_relu(x, bn_name=bn_name, relu_name=relu_name)
        return tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                      strides=strides, padding=padding,
                                      dilation_rate=dilation_rate,
                                      kernel_initializer=kernel_initializer,
                                      kernel_regularizer=kernel_regularizer,
                                      name=conv_name)(activation)

    return f


def _residual_block(block_function, filters, blocks, stage,
                    transition_strides=None, transition_dilation_rates=None,
                    dilation_rates=None, is_first_layer=False, dropout=None,
                    residual_unit=_bn_relu_conv):
    """Builds a residual block with repeating bottleneck blocks.

       stage: integer, current stage label, used for generating layer names
       blocks: number of blocks 'a','b'..., current block label, used for generating
            layer names
       transition_strides: a list of tuples for the strides of each transition
       transition_dilation_rates: a list of tuples for the dilation rate of each
            transition
    """
    # if transition_dilation_rates is None:
    #     transition_dilation_rates = [(1, 1)] * blocks
    if transition_strides is None:
        transition_strides = [(1, 1)] * blocks
    if dilation_rates is None:
        dilation_rates = [1] * blocks

    def f(x):
        for i in range(blocks):
            is_first_block = is_first_layer and i == 0
            x = block_function(filters=filters, stage=stage, block=i,
                               transition_strides=transition_strides[i],
                               dilation_rate=dilation_rates[i],
                               is_first_block_of_first_layer=is_first_block,
                               dropout=dropout,
                               residual_unit=residual_unit)(x)
        return x

    return f


def _shortcut(input_feature, residual, conv_name_base=None, bn_name_base=None):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = tf.keras.backend.int_shape(input_feature)
    residual_shape = tf.keras.backend.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input_feature
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        print('reshaping via a convolution...')
        if conv_name_base is not None:
            conv_name_base = conv_name_base + '1'
        shortcut = tf.keras.layers.Conv2D(filters=residual_shape[CHANNEL_AXIS],
                                          kernel_size=(1, 1),
                                          strides=(stride_width, stride_height),
                                          padding="valid",
                                          kernel_initializer="he_normal",
                                          kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                                          name=conv_name_base)(input_feature)
        if bn_name_base is not None:
            bn_name_base = bn_name_base + '1'
        shortcut = tf.keras.layers.BatchNormalization(axis=CHANNEL_AXIS,
                                                      name=bn_name_base)(shortcut)

    return tf.keras.layers.add([shortcut, residual])


def _block_name_base(stage, block):
    """Get the convolution name base and batch normalization name base defined by
    stage and block.

    If there are less than 26 blocks they will be labeled 'a', 'b', 'c' to match the
    paper and keras and beyond 26 blocks they will simply be numbered.
    """
    if block < 27:
        block = '%c' % (block + 97)  # 97 is the ascii number for lowercase 'a'
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    return conv_name_base, bn_name_base


def basic_block(filters, stage, block, transition_strides=(1, 1),
                dilation_rate=(1, 1), is_first_block_of_first_layer=False, dropout=None,
                residual_unit=_bn_relu_conv):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input_features):
        conv_name_base, bn_name_base = _block_name_base(stage, block)
        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            x = tf.keras.layers.Conv2D(filters=filters, kernel_size=(3, 3),
                                       strides=transition_strides,
                                       dilation_rate=dilation_rate,
                                       padding="same",
                                       kernel_initializer="he_normal",
                                       kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                                       name=conv_name_base + '2a')(input_features)
        else:
            x = residual_unit(filters=filters, kernel_size=(3, 3),
                              strides=transition_strides,
                              dilation_rate=dilation_rate,
                              conv_name_base=conv_name_base + '2a',
                              bn_name_base=bn_name_base + '2a')(input_features)

        if dropout is not None:
            x = tf.keras.layers.Dropout(dropout)(x)

        x = residual_unit(filters=filters, kernel_size=(3, 3),
                          conv_name_base=conv_name_base + '2b',
                          bn_name_base=bn_name_base + '2b')(x)

        return _shortcut(input_features, x)

    return f


def resnet18_tf_keras(input_shape, n_classes):
    """ resnet18 with tf.keras for serving

    Parameters
    ----------
    input_shape: (nb_rows, nb_cols, nb_channels)
    n_classes: default = 5

    Returns
    -------
    model: resnet18 model

    References
    ----------
    # Code
    https://github.com/raghakot/keras-resnet/blob/master/resnet.py
    https://gist.github.com/mjdietzx/0cb95922aac14d446a6530f87b3a04ce
    https://www.tensorflow.org/api_docs/python/tf/contrib/saved_model/save_keras_model

    # Paper
    identity mapping in deep residual networks
    """
    #
    # init
    #
    initial_filters = 64
    initial_kernel_size = (7, 7)
    initial_strides = (2, 2)
    transition_dilation_rate = (1, 1)

    repetitions = [2, 2, 2, 2]  # resnet 18 architecture
    dropout = None
    block_fn = basic_block
    residual_unit = _bn_relu_conv
    activation = "softmax"

    _handle_dim_ordering()

    #
    # Build model
    #
    """
    x_input = Input(shape=input_shape)
    x = Conv2D(initial_filters, kernel_size=initial_kernel_size, strides=initial_strides, padding='same')(x_input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=initial_strides, padding="same")(x)
    """
    img_input = tf.keras.layers.Input(shape=input_shape)
    x = _conv_bn_relu(filters=initial_filters, kernel_size=initial_kernel_size, strides=initial_strides)(img_input)
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=initial_strides, padding="same")(x)

    block = x
    filters = initial_filters
    for i, r in enumerate(repetitions):
        transition_dilation_rates = [transition_dilation_rate] * r
        transition_strides = [(1, 1)] * r
        if transition_dilation_rate == (1, 1):
            transition_strides[0] = (2, 2)
        block = _residual_block(block_fn, filters=filters,
                                stage=i, blocks=r,
                                is_first_layer=(i == 0),
                                dropout=dropout,
                                transition_dilation_rates=transition_dilation_rates,
                                transition_strides=transition_strides,
                                residual_unit=residual_unit)(block)
        filters *= 2

    # Last activation
    x = _bn_relu(block)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(units=n_classes, activation=activation,
                              kernel_initializer="he_normal")(x)

    model = tf.keras.models.Model(inputs=img_input, outputs=x)
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=['accuracy'])
    return model


if __name__ == "__main__":
    #
    # Path
    #
    os.getcwd()
    path = "data"
    os.chdir(path)

    #
    # Data
    #
    data_path = path + "data.pkl"
    x, y = read_image_pickle(data_path)
    x_original = copy.deepcopy(x)
    x = reshape_image_x(x)
    print("Shape of X:", x.shape)
    print("The number of X:", len(x))
    print("Y:", np.unique(y, return_counts=True))

    #
    # Train
    #
    n, p1, p2, channel = x.shape
    input_shape = (p1, p2, 1)
    nb_classes = 5

    cross_validation_k = 5
    cv_data = list(StratifiedKFold(n_splits=cross_validation_k, shuffle=True, random_state=2019).split(x, y))

    y_pred_arr = list()
    y_arr = list()

    y_one_hot = tf.keras.utils.to_categorical(y)

    # Divide train, validation set
    idxs_train, idxs_validation = None, None
    for idx, (idxs_train, idxs_validation) in tqdm(enumerate(cv_data)):
        pass

    # init
    tf.keras.backend.clear_session()

    x_train, y_train = x[idxs_train], y_one_hot[idxs_train]
    x_validation, y_validation = x[idxs_validation], y_one_hot[idxs_validation]

    # train model
    epochs = 50      # 50
    batch_size = 50
    n_classes = 5

    model = resnet18_tf_keras(input_shape, n_classes)
    early_stop = tf.keras.callbacks.EarlyStopping(patience=7, verbose=1, monitor="val_acc")
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=3, min_lr=0.5e-6)
    # todo: original) './model_saved/resnet_tf_keras'
    check_point = tf.keras.callbacks.ModelCheckpoint('./model_saved/test',
                                                     save_best_only=True,
                                                     save_weights_only=True,
                                                     verbose=1,
                                                     monitor="val_acc", mode='max')
    model.fit(x_train, y_train, validation_data=(x_validation, y_validation), batch_size=batch_size,
              shuffle=True, epochs=epochs, verbose=1, callbacks=[early_stop, lr_reducer, check_point])

    # load model
    load_model = True
    if load_model:
        try:
            model.load_weights('./model_saved/resnet_tf_keras')  # train: cnn_resnet.py
            model.compile(loss="categorical_crossentropy",
                          optimizer="adam",
                          metrics=['accuracy'])
            print("ResNet Model is loaded")
        except IOError:
            print("Model does not exist...!")

    scores_train_set = model.evaluate(x_train, y_train, batch_size=batch_size, verbose=1)
    print(model.metrics_names, "\n", scores_train_set)
    scores_validation_set = model.evaluate(x_validation, y_validation, batch_size=batch_size, verbose=1)
    print(model.metrics_names, "\n", scores_validation_set)

    #
    # Save model
    #
    """ path_model_saved: example) 'test/1554880455' """
    path_model_saved = tf.contrib.saved_model.save_keras_model(model=model, saved_model_path='test')

    #
    # Load model
    #
    model_prime = tf.contrib.saved_model.load_keras_model(path_model_saved)
    model_prime.summary()
    model_prime.compile(loss="categorical_crossentropy",
                        optimizer="adam",
                        metrics=['accuracy'])

    scores_train_set = model_prime.evaluate(x_train, y_train, batch_size=batch_size, verbose=1)
    print(model.metrics_names, "\n", scores_train_set)
    scores_validation_set = model_prime.evaluate(x_validation, y_validation, batch_size=batch_size, verbose=1)
    print(model.metrics_names, "\n", scores_validation_set)
