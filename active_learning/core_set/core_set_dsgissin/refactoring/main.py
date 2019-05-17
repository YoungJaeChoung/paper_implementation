import pickle
import os
import sys
from keras.utils import to_categorical
from sklearn.datasets import load_boston, load_diabetes

from refactoring.models import *
from refactoring.query_methods import *

import numpy as np
import dill


class Args:
    def __init__(self,
                 experiment_folder,
                 data_type,
                 method,
                 batch_size=32,
                 initial_size=100,  # Todo: default ... ?
                 iterations=100,    # Todo: default ... ?
                                    # Todo: 의미 ... ? number of active learning batches to sample ... ?
                 gpu=1):
        """
        Args
        ----
        experiment_folder: (str) path
        data_type: (str) name of data
        method: (str) the name of active learning method
        batch_size: (int) batch size
        initial_size: todo ???
        iterations: todo ???
        gpu: (int) default = 1, if gpu > 1: multiple GPUs are used
        """
        self.experiment_folder = experiment_folder
        self.data_type = data_type
        self.batch_size = batch_size
        self.initial_size = initial_size
        self.iterations = iterations
        self.method = method
        self.gpu = gpu
        # path_initial_label_idx: path where pkl for initial label indexes exists.
        self.path_initial_label_idx = self.experiment_folder + '/' + 'initial_label_idx' + '/' + \
                                      "{data}_{initial_size}.pkl".format(data=self.data_type,
                                                                         initial_size=self.initial_size)

def load_batch(fpath, label_key="labels"):
    with open(fpath, 'rb') as f:
        if sys.version_info < (3,):
            d = pickle.load(f)
        else:
            d = pickle.load(f, encoding="bytes")
            # decode utf8
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode('utf8')] = v
            d = d_decoded
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels


def load_mnist():
    """
    load nad pre-process the MNIST data
    """

    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == "channels_last":
        x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
        x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    else:
        x_train = x_train.reshape((x_train.shape[0], 1, 28, 28))
        x_test = x_test.reshape((x_test.shape[0], 1, 28, 28))

    # standardize the dataset:
    x_train = np.array(x_train).astype('float32') / 255
    x_test = np.array(x_test).astype('float32') / 255

    # shuffle the data:
    perm = np.random.permutation(x_train.shape[0])
    x_train = x_train[perm]
    y_train = y_train[perm]

    return (x_train, y_train), (x_test, y_test)


def evaluate_sample(training_function,
                    X_train, Y_train,
                    X_test, Y_test,
                    checkpoint_path):
    """
    A function that accepts a labeled-unlabeled data split
    and trains the relevant model on the labeled data,
    returning the model and it`s accuracy on the test set
    """

    # shuffle the training set:
    nb_obs = X_train.shape[0]
    perm = np.random.permutation(nb_obs)
    X_train = X_train[perm]
    Y_train = Y_train[perm]

    # create the validation set:
    X_validation = X_train[:int(0.2*nb_obs)]
    Y_validation = Y_train[:int(0.2*nb_obs)]
    X_train = X_train[int(0.2*nb_obs):]
    Y_train = Y_train[int(0.2*nb_obs):]

    # train and evaluate the model
    # train on labeled data
    # use validation set to check generalization error
    model = training_function(X_train, Y_train,
                              X_validation, Y_validation,
                              checkpoint_path, gpu=args.gpu)
    if args.data_type in ['imdb', "wiki"]:
        acc = model.evaluate(X_test, Y_test, verbose=0)
    else:
        loss, acc = model.evaluate(X_test, Y_test, verbose=0)

    return acc, model


def get_data(data_name="mnist"):
    """
    get data

    Args
    ----
    data_name: (str) i.e. 'mnist'

    Returns
    -------
    X_train,    Y_train
    X_test,    Y_test
    nb_labels
    input_shape
    evaluation_function
    """
    if data_name == "mnist":
        (X_train, Y_train), (X_test, Y_test) = load_mnist()
        nb_labels = 10
        if K.image_data_format() == "channels_last":
            input_shape = (28, 28, 1)
        else:
            input_shape = (1, 28, 28)

    # make categorical:
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)

    return X_train, Y_train, X_test, Y_test, nb_labels, input_shape


def make_model_folder():
    model_folder = os.path.join(args.experiment_folder, 'models')
    if not os.path.isdir(model_folder):
        os.mkdir(model_folder)
        print("folder is made...")
    else:
        print("folder exists...")
    return model_folder


def save_obj(data, obj_name):
    # Code from: https://lovit.github.io/analytics/2019/01/15/python_dill/
    with open(obj_name, 'wb') as f:
        dill.dump(data, f)
    print("Save is completed:", obj_name)


def load_initial_label_idx():
    try:
        with open(args.path_initial_label_idx, 'rb') as f:
            labeled_idx = pickle.load(f)
        print("labeled_idx is loaded")
    except:
        print("No initial indices found - drawing random indices...")
        nb_obs = X_train.shape[0]
        labeled_idx = np.random.choice(nb_obs, args.initial_size,
                                       replace=False)
        save_obj(data=labeled_idx, obj_name=args.path_initial_label_idx)
    return labeled_idx


if __name__ == "__main__":
    #
    # arguments
    #
    path = r"D:\소프트팩토리\소프트팩토리_대전\Git\paper_implementation" \
           r"\active_learning\core_set\core_set_dsgissin\refactoring"
    data_type = "mnist"
    method = 'CoreSet'
    args = Args(experiment_folder=path,
                data_type=data_type,
                method=method)
    print(args.__dict__)

    # set path
    print(args.experiment_folder)
    model_folder = make_model_folder()

    # check point path
    checkpoint_path = \
        os.path.join(model_folder, '{alg}_{datatype}_{init}_{batch_size}.hdf5'.format(
                     alg=args.method, datatype=args.data_type, init=args.initial_size, batch_size=args.batch_size))

    #
    # load the data set
    #
    # Todo: get_data 부분 바꿈 ... ? 개인 데이터 쓸 수 있도록
    X_train, Y_train, X_test, Y_test, nb_labels, input_shape = \
        get_data(data_name="mnist")

    #
    # load the indices
    #
    # Todo: cost-effective 방법이랑 - 방법 비교할 수 있도록 정리 필요
    labeled_idx = load_initial_label_idx()

    #
    # Set Query Method
    #
    # Todo: CoreSetSampling 코드가 어떻게 다른 것과 연결되서 돌아가는지 확인 필요
    method = CoreSetSampling

    # Todo: 왜 여기 model 이 None 으로 들어가있을까 ... ?
    query_method = method(None, input_shape, nb_labels, args.gpu)

    #
    # run the experiment
    #
    # Todo: 여기서부터 하면 됨 ... !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    accuracies = []
    entropies = []
    label_distributions = []
    queries = []
    # Todo: checkpoint_path ... ?
    # evaluate_sample: train model and calculate acc on test set
    # evaluation function:
    # evaluate with certain model by training the data with active learning
    training_function = train_LeNet_model
    acc, model = evaluate_sample(training_function,
                                 X_train[labeled_idx,:], Y_train[labeled_idx],
                                 X_test, Y_test,
                                 checkpoint_path)
    accuracies.append(acc)
    print("Test Accuracy Is " + str(acc))
    query_method.update_model(model)    # change model

    for _ in range(args.iterations):
        # get the new indices from the algorithm
        old_labeled = np.copy(labeled_idx)
        # old_labeled + new labeled indexes
        labeled_idx = query_method.query(X_train, Y_train, labeled_idx, args.batch_size)

        # calculate and store the label entropy:
        # Todo: 이 부분 돌려보면서 값 확인해본 후 ㅡ entropy 부분 함수로 빼냄
        new_idx = labeled_idx[np.logical_not(np.isin(labeled_idx, old_labeled))]    # Todo: 이 부분 겹칠수도 있나 ... ?
        new_labels = Y_train[new_idx]       # Todo: Y_train 의 데이터 형태 ... ?
        new_labels /= np.sum(new_labels)    # Todo: 이거 왜 나눠줄까 ... ? entropy 계산 때문에 그런 듯 ... ?
        new_labels = np.sum(new_labels, axis=0)
        entropy = -np.sum(new_labels * np.log(new_labels + 1e-10))  # todo: 1e-10 은 0 안되게 하려고 넣은건가 ... ?
        entropies.append(entropy)
        label_distributions.append(new_labels)
        queries.append(new_idx)

        # evaluate the new sample:
        # Todo: checkpoint_path
        acc, model = evaluate_sample(evaluation_function,
                                     X_train[labeled_idx], Y_train[labeled_idx],
                                     X_test, Y_test,
                                     checkpoint_path)
        query_method.update_model(model)
        accuracies.append(acc)
        print("Test Accuracy Is " + str(acc))

    # Todo: results path ... ?
    # Todo: entropy path ... ?
    # save the results:
    with open(results_path, 'wb') as f:
        pickle.dump([accuracies, args.initial_size, args.batch_size], f)
        print("Saved results to " + results_path)
    with open(entropy_path, 'wb') as f:
        pickle.dump([entropies, label_distributions, queries], f)
        print("Saved entropy statistics to " + entropy_path)

