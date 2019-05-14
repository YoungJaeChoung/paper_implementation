# https://github.com/dsgissin/DiscriminativeActiveLearning/blob/master/models.py

"""
The file containing implementations to all of the neural network models used in our experiments. These include a LeNet
model for MNIST, a VGG model for CIFAR and a multilayer perceptron model for dicriminative active learning, among others.
"""

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


class DiscriminativeEarlyStopping(Callback):
    """
    A custom callback for discriminative active learning, to stop the training a little bit before the classifier is
    able to get 100% accuracy on the training set. This makes sure examples which are similar to ones already in the
    labeled set won't have a very high confidence.
    """

    def __init__(self, monitor='acc', threshold=0.98, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.threshold = threshold
        self.verbose = verbose
        self.improved = 0

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)

        if current > self.threshold:
            if self.verbose > 0:
                print("Epoch {e}: early stopping at accuracy {a}".format(e=epoch, a=current))
            self.model.stop_training = True


class Callback(object):
    # original keras Callback class
    """Abstract base class used to build new callbacks.

    # Properties
        params: dict. Training parameters
            (eg. verbosity, batch size, number of epochs...).
        model: instance of `keras.models.Model`.
            Reference of the model being trained.

    The `logs` dictionary that callback methods
    take as argument will contain keys for quantities relevant to
    the current batch or epoch.

    Currently, the `.fit()` method of the `Sequential` model class
    will include the following quantities in the `logs` that
    it passes to its callbacks:

        on_epoch_end: logs include `acc` and `loss`, and
            optionally include `val_loss`
            (if validation is enabled in `fit`), and `val_acc`
            (if validation and accuracy monitoring are enabled).
        on_batch_begin: logs include `size`,
            the number of samples in the current batch.
        on_batch_end: logs include `loss`, and optionally `acc`
            (if accuracy monitoring is enabled).
    """

    def __init__(self):
        self.validation_data = None
        self.model = None

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self.model = model

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass


class ModelCheckpoint(Callback):
    # original keras ModelCheckpoint class
    """Save the model after every epoch.

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)


class EarlyStopping(Callback):
    # original keras EarlyStopping class
    """Stop training when a monitored quantity has stopped improving.

    # Arguments
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        baseline: Baseline value for the monitored quantity to reach.
            Training will stop if the model doesn't show improvement
            over the baseline.
        restore_best_weights: whether to restore model weights from
            the epoch with the best value of the monitored quantity.
            If False, the model weights obtained at the last step of
            training are used.
    """

    def __init__(self,
                 monitor='val_loss',
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 restore_best_weights=False):
        super(EarlyStopping, self).__init__()

        self.monitor = monitor
        self.baseline = baseline
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % mode,
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of '
                              'the best epoch')
                    self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    def get_monitor_value(self, logs):
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
        return monitor_value


class DelayedModelCheckpoint(Callback):
    # 필사 2 회
    # todo: 왜 이게 delayed 일까 ... ?
    """
    A custom callback for saving the model
    each time the validation accuracy improves.

    The custom part is that
    we save the model
    when the accuracy stays the same as well,
    and also that we start saving
    only after a certain amount of iterations to save time.
    """

    def __init__(self, filepath, monitor='val_acc',
                 delay=50, verbose=0, weights=False):

        super(DelayedModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.delay = delay
        if self.monitor == 'val_acc':
            self.best = -np.Inf
        else:
            self.best = np.Inf      # entropy ... ?
        self.weights = weights      # save only weights

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}   # Todo: ... ?

        if self.monitor == 'val_acc':
            # todo: self.monitor 에 예측한게 어디서 저장되는 건가 확인 필요
            current = logs.get(self.monitor)    # dictionary
            # delay until epoch > self.delay
            if current >= self.best and epoch > self.delay:
                if self.verbose > 0:
                    # todo: self.monitor 에는 어떤 값이 들어가있을까 ... ?
                    print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                          ' saving model to %s'
                          % (epoch, self.monitor, self.best,
                             current, self.filepath))
                self.best = current
                if self.weights:
                    self.model.save_weights(self.filepath, overwrite=True)
                else:
                    self.model.save(self.filepath, overwrite=True)
        else:
            current = logs.get(self.monitor)
            if current <= self.best and epoch > self.delay:
                if self.verbose > 0:
                    print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                          ' saving model to %s'
                          % (epoch, self.monitor, self.best,
                             current, self.filepath))
                self.best = current
                if self.weights:
                    self.model.save_weights(self.filepath, overwrite=True)
                else:
                    self.model.save(self.filepath, overwrite=True)



class ModelMGPU(Model):
    # Todo: 필사 2 회
    # Todo: Model class ... ?
    # Todo: 사용된 예제 보면 좋을 듯
    def __init__(self, ser_model, gpus):
        # todo: ser_model ... ?
        # todo: multi_gpu_model ... ?
        #  함수 ... ?
        #  cpu_relocation ... ?
        #  cpu_merge ... ?
        # todo: pmodel ... ? 이건 왜 안사용하지 ... ?
        pmodel = multi_gpu_model(ser_model, gpus,
                                 cpu_relocation=False, cpu_merge=False)
        # todo: self.__dict__ ... ?
        # todo: self.__dict__.update ... ?
        # todo: pmodel.__dict__ ... ?
        # todo: self._smodel ... ?
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        # Todo: serial-model ... ?
        """Override load and save methods to be used from the serial-model.
        The serial-model holds references to the weights in the multi-gpu model.
        """
        if 'load' in attrname or 'save' in attrname:
            # todo: self._smodel ... ?
            return getattr(self._smodel, attrname)

        # todo: __getattribute__ ... ? 블로그 정리 필요
        return super(ModelMGPU, self).__getattribute__(attrname)


def get_discriminative_model(input_shape):
    """
    The MLP model for discriminative active learning, without any regularization techniques.
    """

    if np.sum(input_shape) < 30:
        width = 20
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(width, activation='relu'))
        model.add(Dense(width, activation='relu'))
        model.add(Dense(width, activation='relu'))
        model.add(Dense(2, activation='softmax', name='softmax'))
    else:
        width=256
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(width, activation='relu'))
        model.add(Dense(width, activation='relu'))
        model.add(Dense(width, activation='relu'))
        model.add(Dense(2, activation='softmax', name='softmax'))

    return model


def get_LeNet_model(input_shape, labels=10):
    """
    A LeNet model for MNIST.
    """

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', name='embedding'))
    model.add(Dropout(0.5))
    model.add(Dense(labels, activation='softmax', name='softmax'))

    return model


def get_VGG_model(input_shape, labels=10):
    """
    A VGG model for CIFAR.
    """

    weight_decay = 0.0005
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same',
                     input_shape=input_shape, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer=regularizers.l2(weight_decay), name='embedding'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(labels, activation='softmax', name='softmax'))

    return model


def get_autoencoder_model(input_shape, labels=10):
    """
    An autoencoder for MNIST to be used in the DAL implementation.
    """

    image = Input(shape=input_shape)
    encoder = Conv2D(32, (3, 3), activation='relu', padding='same')(image)
    encoder = MaxPooling2D((2, 2), padding='same')(encoder)
    encoder = Conv2D(8, (3, 3), activation='relu', padding='same')(encoder)
    encoder = Conv2D(4, (3, 3), activation='relu', padding='same')(encoder)
    encoder = MaxPooling2D((2, 2), padding='same')(encoder)

    decoder = UpSampling2D((2, 2), name='embedding')(encoder)
    decoder = Conv2D(4, (3, 3), activation='relu', padding='same')(decoder)
    decoder = Conv2D(8, (3, 3), activation='relu', padding='same')(decoder)
    decoder = UpSampling2D((2, 2))(decoder)
    decoder = Conv2D(32, (3, 3), activation='relu', padding='same')(decoder)
    decoder = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(decoder)

    autoencoder = Model(image, decoder)
    return autoencoder


def train_discriminative_model(labeled, unlabeled, input_shape, gpu=1):
    """
    A function that trains and returns a discriminative model on the labeled and unlabaled data.
    """

    # create the binary dataset:
    y_L = np.zeros((labeled.shape[0],1),dtype='int')
    y_U = np.ones((unlabeled.shape[0],1),dtype='int')
    X_train = np.vstack((labeled, unlabeled))
    Y_train = np.vstack((y_L, y_U))
    Y_train = to_categorical(Y_train)

    # build the model:
    model = get_discriminative_model(input_shape)

    # train the model:
    batch_size = 1024
    if np.max(input_shape) == 28:
        optimizer = optimizers.Adam(lr=0.0003)
        epochs = 200
    elif np.max(input_shape) == 128:
        # optimizer = optimizers.Adam(lr=0.0003)
        # epochs = 200
        batch_size = 128
        optimizer = optimizers.Adam(lr=0.0001)
        epochs = 1000 #TODO: was 200
    elif np.max(input_shape) == 512:
        optimizer = optimizers.Adam(lr=0.0002)
        # optimizer = optimizers.RMSprop()
        epochs = 500
    elif np.max(input_shape) == 32:
        optimizer = optimizers.Adam(lr=0.0003)
        epochs = 500
    else:
        optimizer = optimizers.Adam()
        # optimizer = optimizers.RMSprop()
        epochs = 1000
        batch_size = 32

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    callbacks = [DiscriminativeEarlyStopping()]
    model.fit(X_train, Y_train,
              epochs=epochs,
              batch_size=batch_size,
              shuffle=True,
              callbacks=callbacks,
              class_weight={0 : float(X_train.shape[0]) / Y_train[Y_train==0].shape[0],
                            1 : float(X_train.shape[0]) / Y_train[Y_train==1].shape[0]},
              verbose=2)

    return model


def train_mnist_model(X_train, Y_train, X_validation, Y_validation,
                      checkpoint_path, gpu=1):
    # todo: 필사 1 회
    """
    A function that trains and returns a LeNet model on the labeled MNIST data.
    """

    if K.image_data_format() == 'channels_last':
        input_shape = (28, 28, 1)
    else:
        input_shape = (1, 28, 28)

    model = get_LeNet_model(input_shape=input_shape, labels=10)
    optimizer = optimizers.Adam()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])
    callbacks = [DelayedModelCheckpoint(filepath=checkpoint_path,
                                        verbose=1,
                                        weights=True)]

    if gpu > 1:
        gpu_model = ModelMGPU(model, gpus = gpu)
        gpu_model.compile(loss='categorical_crossentropy',
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
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        model.load_weights(checkpoint_path)
        return model

    else:
        model.fit(X_train, Y_train,
                  epochs=150,
                  batch_size=32,
                  shuffle=True,
                  validation_data=(X_validation, Y_validation),
                  callbacks=callbacks,
                  verbose=2)
        model.load_weights(checkpoint_path)
        return model


def train_cifar10_model(X_train, Y_train, X_validation, Y_validation, checkpoint_path, gpu=1):
    """
    A function that trains and returns a VGG model on the labeled CIFAR-10 data.
    """

    if K.image_data_format() == 'channels_last':
        input_shape = (32, 32, 3)
    else:
        input_shape = (3, 32, 32)

    model = get_VGG_model(input_shape=input_shape, labels=10)
    optimizer = optimizers.Adam()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    callbacks = [DelayedModelCheckpoint(filepath=checkpoint_path, verbose=1, weights=True)]

    if gpu > 1:
        gpu_model = ModelMGPU(model, gpus = gpu)
        gpu_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        gpu_model.fit(X_train, Y_train,
                      epochs=400,
                      batch_size=32,
                      shuffle=True,
                      validation_data=(X_validation, Y_validation),
                      callbacks=callbacks,
                      verbose=2)

        del gpu_model
        del model

        model = get_VGG_model(input_shape=input_shape, labels=10)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        model.load_weights(checkpoint_path)

        return model

    else:
        model.fit(X_train, Y_train,
                      epochs=400,
                      batch_size=32,
                      shuffle=True,
                      validation_data=(X_validation, Y_validation),
                      callbacks=callbacks,
                      verbose=2)

        model.load_weights(checkpoint_path)
        return model


def train_cifar100_model(X_train, Y_train, X_validation, Y_validation, checkpoint_path, gpu=1):
    """
    A function that trains and returns a VGG model on the labeled CIFAR-100 data.
    """

    if K.image_data_format() == 'channels_last':
        input_shape = (32, 32, 3)
    else:
        input_shape = (3, 32, 32)

    model = get_VGG_model(input_shape=input_shape, labels=100)
    optimizer = optimizers.Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    callbacks = [DelayedModelCheckpoint(filepath=checkpoint_path, verbose=1, weights=True)]

    if gpu > 1:
        gpu_model = ModelMGPU(model, gpus = gpu)
        gpu_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        gpu_model.fit(X_train, Y_train,
                      epochs=1000,
                      batch_size=128,
                      shuffle=True,
                      validation_data=(X_validation, Y_validation),
                      callbacks=callbacks,
                      verbose=2)

        del gpu_model
        del model

        model = get_VGG_model(input_shape=input_shape, labels=100)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        model.load_weights(checkpoint_path)

        return model

    else:
        model.fit(X_train, Y_train,
                      epochs=1000,
                      batch_size=128,
                      shuffle=True,
                      validation_data=(X_validation, Y_validation),
                      callbacks=callbacks,
                      verbose=2)

        model.load_weights(checkpoint_path)
        return model