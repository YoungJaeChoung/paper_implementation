from tf_keras_for_serving import resnet_tf_keras as serving
import tensorflow as tf

from scipy.spatial import distance_matrix
from tqdm import tqdm

import pickle
import dill
import copy
import os

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
matplotlib.use("Agg")

# to plot partial discharge image
color_map = ListedColormap(
    ['#faeb78', '#fadc3c', '#fac81e', '#fab400', '#faa500', '#fa9b00', '#fa9100', '#fa8200', '#fa3232',
     '#ff0000'])
color_map.set_bad('#000080')
plt.figure(figsize=(7, 4))

# index
idx_obs = 0


class Args:
    def __init__(self, epochs=5, batch_size=32,
                 fine_tunning_interval=1, maximum_iterations=10,
                 initial_annotated_percent=0.1,
                 interval_update_labels=1,
                 high_confidence_samples_selection_threshold=0.05, threshold_decay=0.0033,
                 uncertain_samples_size=1000, uncertain_criteria="en",
                 cost_effective=True,
                 chkt_filename="ResNet18v2-CIFAR-10_init_ceal.hdf5",
                 verbose=1):
        self.epochs = epochs
        self.batch_size = batch_size
        self.fine_tunning_interval = fine_tunning_interval
        self.maximum_iterations = maximum_iterations
        self.initial_annotated_percent = initial_annotated_percent
        self.interval_update_labels = interval_update_labels
        self.cnt_label_updated = 0

        # delta
        self.high_confidence_sample_selection_threshold = high_confidence_samples_selection_threshold
        self.threshold_decay = threshold_decay

        self.uncertain_samples_size = uncertain_samples_size
        self.uncertain_criteria = uncertain_criteria

        self.cost_effective = cost_effective

        self.chkt_filename = chkt_filename
        self.verbose = verbose


def set_hyper_params(x):
    global n, p1, p2, channel
    global input_shape
    n, p1, p2, channel = x.shape
    input_shape = tuple([p1, p2, 1])

    # model
    global epochs, batch_size, n_classes
    epochs = 35
    batch_size = 50
    n_classes = 5


def divide_train_test_set(x, y_one_hot, ratio_train=0.8, stratify=True):
    #
    # Init
    #
    idx_ele = 0
    idx_n = 0
    n = x.shape[idx_n]
    nb_train = int(n * ratio_train)

    #
    # Get Index
    #
    if stratify:
        y_arr = np.apply_along_axis(np.argmax, axis=1, arr=y_one_hot)
        unique_classes, cnt_classes = np.unique(y_arr, return_counts=True)
        ratio_classes = cnt_classes / sum(cnt_classes)
        nb_train_per_class = [int(x) for x in nb_train * ratio_classes]

        indexes_train = np.array([])
        for label, num_train in zip(unique_classes, nb_train_per_class):
            indexes_label = np.where(y_arr == label)[idx_ele]
            index_train_tmp = np.random.choice(indexes_label, num_train, replace=False)
            indexes_train = np.concatenate([indexes_train, index_train_tmp]).astype(np.int64)

        indexes_test = np.delete(range(n), indexes_train)
    else:
        indexes_train = np.random.choice(range(n), nb_train, replace=False)
        indexes_test = np.delete(range(n), indexes_train)

    #
    # Split Data
    #
    x_train, y_train = x[indexes_train], y_one_hot[indexes_train]
    x_validation, y_validation = x[indexes_test], y_one_hot[indexes_test]
    return x_train, y_train, x_validation, y_validation


def train_model(x, y_one_hot, name_load=None, name_save=None,
                load_trained_model=True,
                monitor="acc",
                path='../model_saved/active_learning'):
    """ train resnet18 model

    Parameters
    ----------
    name: (str) name of file where model is saved
    load_trained_model: (bool) whether pre trained model is used or not
    monitor: (str) monitoring measure i.e. "acc", "val_acc"
    path: (str) folder path of model saved

    """
    # error check
    assert not (name_load is None and load_trained_model is True), "loading model is not adequate...!"

    # init
    if name_save is None:
        name_save = name_load
    if name_load is None:
        name_load = name_save

    # divide data
    x_train, y_train, x_validation, y_validation = divide_train_test_set(x, y_one_hot, ratio_train=0.8, stratify=True)

    # Init
    tf.keras.backend.clear_session()

    # Load Model if exists
    model = serving.resnet18_tf_keras(input_shape, n_classes)  # original: model = ResNet18(input_shape, n_classes)
    if load_trained_model:
        try:
            model.load_weights(path + "/active_learning_{}".format(name_load))  # train: cnn_resnet.py
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
            print("ResNet Model is loaded")
        except IOError:
            print("Model does not exist...!")
    else:
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    # Train
    early_stop = tf.keras.callbacks.EarlyStopping(patience=5, verbose=1, monitor=monitor)
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=3, min_lr=0.5e-6)
    check_point = tf.keras.callbacks.ModelCheckpoint(path + "/active_learning_{}".format(name_save),
                                                     save_best_only=True,
                                                     save_weights_only=True,
                                                     # 'save_weights_only' is required to avoid error
                                                     verbose=1,
                                                     monitor=monitor, mode='max')
    # Todo: 데이터 불균형을 고려한 평가지표 사용 못하나 ... ? ㅡ macro f1 micro f1 ... ?
    # Todo: https://arxiv.org/pdf/1402.1892.pdf 여기에서는 f1 이 안좋다고 나오는 것 같은데 ...
    history = model.fit(x_train, y_train, validation_data=(x_validation, y_validation), batch_size=batch_size,
                        shuffle=True, epochs=epochs, verbose=1, callbacks=[early_stop, lr_reducer, check_point])

    """ test
    # ---------------------------------------------------------------------------------------
    del model

    model = ResNet18(input_shape, n_classes)
    model.load_weights(path + "/active_learning_{}".format(name_save))  # train: cnn_resnet.py
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    # ---------------------------------------------------------------------------------------

    scores_train_set = model.evaluate(x_train, y_train, batch_size=batch_size, verbose=1)
    print(model.metrics_names, "\n", scores_train_set)
    scores_validation_set = model.evaluate(x_validation, y_validation, batch_size=batch_size, verbose=1)
    print(model.metrics_names, "\n", scores_validation_set)
    """

    return history


def check_acc(x, y_one_hot, name_load, path='../model_saved/active_learning'):
    idx_acc = 1

    tf.keras.backend.clear_session()
    model = serving.resnet18_tf_keras(input_shape, n_classes)
    try:
        model.load_weights(path + "/active_learning_{}".format(name_load))  # train: cnn_resnet.py
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        print("ResNet Model is loaded")
    except IOError:
        print("Model does not exist...!")

    scores_train_set = model.evaluate(x, y_one_hot, batch_size=batch_size, verbose=1)
    acc = round(scores_train_set[idx_acc] * 100, 2)
    print("Accuracy: {}%".format(acc))  # caution: normalization
    return acc


def greedy_k_center(x_labeled, x_unlabeled, amount):
    greedy_indices = []

    min_dist = np.min(distance_matrix(x_labeled[0, :].reshape((1, x_labeled.shape[1])),
                                      x_unlabeled), axis=0)
    min_dist = min_dist.reshape((1, min_dist.shape[0]))
    nb_obs = x_labeled.shape[0]

    for idx in range(1, nb_obs, 100):
        if idx + 100 < nb_obs:
            dist = distance_matrix(x_labeled[idx:idx + 100, :], x_unlabeled)
        else:
            dist = distance_matrix(x_labeled[idx:, :], x_unlabeled)
        min_dist = np.vstack((min_dist,
                              np.min(dist, axis=0).reshape((1, min_dist.shape[1]))))
        min_dist = np.min(min_dist, axis=0)
        min_dist = min_dist.reshape((1, min_dist.shape[0]))

    farthest = np.argmax(min_dist)
    greedy_indices.append(farthest)
    for _ in tqdm(range(amount - 1)):
        dist = distance_matrix(
            x_unlabeled[greedy_indices[-1], :].reshape((1, x_unlabeled.shape[1])),
            x_unlabeled
        )
        min_dist = np.vstack((min_dist, dist.reshape((1, min_dist.shape[1]))))
        min_dist = np.min(min_dist, axis=0)
        min_dist = min_dist.reshape((1, min_dist.shape[0]))
        farthest = np.argmax(min_dist)
        greedy_indices.append(farthest)

    return np.array(greedy_indices)


def get_core_set_query_index(x_train, x_pool, n_samples=100):
    """ update labeled data by labeling uncertain data of unlabeled data """
    #
    # query: new label
    #

    # load model
    model = serving.resnet18_tf_keras(input_shape, n_classes)
    model.load_weights("../model_saved/active_learning/active_learning_{}".format(name_save))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    # get embedding
    representation_model = tf.keras.models.Model(inputs=model.input,
                                                 outputs=model.get_layer('output_layer').input)
    representation_labeled = representation_model.predict(x_train, verbose=0)
    representation_unlabeled = representation_model.predict(x_pool, verbose=0)

    # get query idx & new labeled data
    idx_uncertain = greedy_k_center(representation_labeled,
                                    representation_unlabeled,
                                    n_samples)
    return idx_uncertain


def update_label_set_core_set(x_train, y_train, x_pool, y_pool, n_samples=100):
    """ update labeled data by labeling uncertain data of unlabeled data """
    #
    # Init
    #
    idx_obs = 0

    #
    # query: new label
    #

    # load model
    model = serving.resnet18_tf_keras(input_shape, n_classes)
    model.load_weights("../model_saved/active_learning/active_learning_{}".format(name_save))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    # get embedding
    representation_model = tf.keras.models.Model(inputs=model.input,
                                                 outputs=model.get_layer('output_layer').input)
    representation_labeled = representation_model.predict(x_train, verbose=0)
    representation_unlabeled = representation_model.predict(x_pool, verbose=0)

    # get query idx & new labeled data
    idx_uncertain = greedy_k_center(representation_labeled,
                                    representation_unlabeled,
                                    n_samples)
    x_new_label, y_new_label = x_pool[idx_uncertain, :, :, :], y_pool[idx_uncertain, :]

    # print the number of obs (before update)
    nb_train = x_train.shape[idx_obs]
    nb_pool = x_pool.shape[idx_obs]
    print("The number of train set: (before update)", nb_train)
    print("The number of pool set: (before update)", nb_pool, "\n")

    # update train set
    x_train = np.concatenate([x_train, x_new_label])
    y_train = np.concatenate([y_train, y_new_label])

    # delete label obs from x_pool, y_pool
    x_pool = np.delete(x_pool, idx_uncertain, axis=idx_obs)
    y_pool = np.delete(y_pool, idx_uncertain, axis=idx_obs)

    # print the number of obs (after update)
    nb_train = x_train.shape[idx_obs]
    nb_pool = x_pool.shape[idx_obs]
    print("The number of train set (after update):", nb_train)
    print("The number of pool set (after update):", nb_pool)

    return x_train, y_train, x_pool, y_pool


def save_obj(data, obj_name):
    # Code from: https://lovit.github.io/analytics/2019/01/15/python_dill/
    with open(obj_name, 'wb') as f:
        dill.dump(data, f)
    print("Save is completed:", obj_name)


def load_obj(pkl_name):
    # Code from: https://lovit.github.io/analytics/2019/01/15/python_dill/
    with open(pkl_name, 'rb') as f:
        obj = dill.load(f)
    print("Load is completed:", pkl_name)
    return obj


def save_initial_data(idx_exp):
    path = r''
    save_obj(x_train, path + '/' + "x_train_{}.pkl".format(idx_exp))
    save_obj(y_train, path + '/' + "y_train_{}.pkl".format(idx_exp))
    save_obj(x_test, path + '/' + "x_test_{}.pkl".format(idx_exp))
    save_obj(y_test, path + '/' + "y_test_{}.pkl".format(idx_exp))
    save_obj(x_pool, path + '/' + "x_pool_{}.pkl".format(idx_exp))
    save_obj(y_pool, path + '/' + "y_pool_{}.pkl".format(idx_exp))
    print("exp {} is saved...".format(idx_exp))


def load_initial_data(idx_exp):
    """
    x_train, y_train, x_test, y_test, x_pool, y_pool = load_initial_data(idx_exp)
    """
    path = r''
    x_train = load_obj(path + '/' + "x_train_{}.pkl".format(idx_exp))
    y_train = load_obj(path + '/' + "y_train_{}.pkl".format(idx_exp))
    x_test = load_obj(path + '/' + "x_test_{}.pkl".format(idx_exp))
    y_test = load_obj(path + '/' + "y_test_{}.pkl".format(idx_exp))
    x_pool = load_obj(path + '/' + "x_pool_{}.pkl".format(idx_exp))
    y_pool = load_obj(path + '/' + "y_pool_{}.pkl".format(idx_exp))

    return x_train, y_train, x_test, y_test, x_pool, y_pool


# high confidence
# Todo: 이 부분 수정해야하나 ... ?
def predict_prob(x, name_load, path='../model_saved/active_learning'):
    tf.keras.backend.clear_session()
    model = serving.resnet18_tf_keras(input_shape, n_classes)
    try:
        model.load_weights(path + "/active_learning_{}".format(name_load))  # train: cnn_resnet.py
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        print("ResNet Model is loaded")
    except IOError:
        print("Model does not exist...!")

    pred = model.predict(x, batch_size=batch_size, verbose=1)
    return pred


def least_confidence(prob_pred_arr, n_samples):
    """ Rank all the unlabeled samples in an ascending order according to the least confidence """
    # init
    row_wise = 1
    idx_stat = 1
    idx_index = 0

    # get stat
    origin_index = np.arange(0, len(prob_pred_arr))
    max_prob = np.max(prob_pred_arr, axis=row_wise)
    pred_label = np.argmax(prob_pred_arr, axis=row_wise)

    stat_df = np.column_stack((origin_index,
                               max_prob,
                               pred_label))
    stat_df = stat_df[stat_df[:, idx_stat].argsort()]

    return stat_df[:n_samples], stat_df[:, idx_index].astype(int)[:n_samples]


def get_high_confidence_samples(x_pool, name_load, threshold=0.95):
    # init
    idx_index = 0
    idx_stat = 1
    idx_label = 2

    # get stat
    prob_pred_arr = predict_prob(x_pool, name_load)
    high_confident_df, high_confident_index = least_confidence(prob_pred_arr, len(prob_pred_arr))
    """ print
    np.min(high_confident_df[:, idx_stat])
    np.max(high_confident_df[:, idx_stat])
    """

    # get index of high confident elements
    high_confident_df = high_confident_df[high_confident_df[:, idx_stat] > threshold]     # original
    idx_hc, label_hc = high_confident_df[:, idx_index].astype(int), high_confident_df[:, idx_label].astype(int)
    return idx_hc, label_hc


def make_trainset_with_hc(x_train, y_train, x_pool, name_load, threshold=0.95):
    """ add labeled data and high coinfient data """
    #   2) High Confident Data 중, 선택된 것을 Labeled Data 와 '임시로' 합친다.
    idx_hc, label_hc = get_high_confidence_samples(x_pool, name_load, threshold=threshold)  # Todo: hc 너무 많은 것 아닌가 ... ?
    x_train_with_hc, y_train_with_hc = \
        np.concatenate([x_train, x_pool[idx_hc, :, :, :]]), np.concatenate([y_train, tf.keras.utils.to_categorical(label_hc, num_classes=n_classes)])

    return x_train_with_hc, y_train_with_hc


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

    # init
    n = None
    input_shape = None
    set_hyper_params(x)     # epochs = 35, batch_size = 50
    y_one_hot = tf.keras.utils.to_categorical(y)

    #
    # Active Learning
    #
    nb_exp = 3
    nb_query_epochs = 10
    exp_result = np.zeros([nb_exp, nb_query_epochs])
    for idx_exp in range(0, nb_exp):
        print("exp {} starts...".format(idx_exp))
        """
        idx_exp = 1
        """
        # 1) initialize data set
        # train: 563 / pool: 2266 / test: 317
        """
        x_train, y_train, x_test, y_test = divide_train_test_set(x, y_one_hot, ratio_train=0.9, stratify=True)
        x_train, y_train, x_pool, y_pool = divide_train_test_set(x_train, y_train, ratio_train=0.2, stratify=True)
        """

        x_train, y_train, x_test, y_test, x_pool, y_pool = load_initial_data(idx_exp)
        print(x_train.shape[0] + x_test.shape[0] + x_pool.shape[0])
        # save_initial_data(idx_exp)  # save initial data

        # 2) train
        acc_arr = []
        idx_acc = 1
        x_train_with_hc, y_train_with_hc = None, None

        for epoch in range(0, nb_query_epochs):   # add label 5 times x n_samples=100
            """
            epoch = 0
            # epochs = 1 # Todo: 제거
            """
            name_save = "core_set"
            if epoch == 0:
                name_load, load_trained_model = None, False
            else:
                name_load, load_trained_model = name_save, True

            # 2-1) train
            if epoch == 0:
                history = train_model(x_train, y_train, name_load=name_load, name_save=name_save,
                            load_trained_model=load_trained_model, monitor="val_acc")
            else:
                history = train_model(x_train_with_hc, y_train_with_hc, name_load=name_save, name_save=name_save,
                            load_trained_model=load_trained_model, monitor="val_acc")

            # 2-2) check accuracy
            acc = check_acc(x_test, y_test, name_load=name_save, path='../model_saved/active_learning')    # 84.2 %
            print("The Accuracy of Model (epoch: {}): {}%".format(epoch, acc))
            acc_arr.append(acc)

            # 2-3) uncertain data
            x_train, y_train, x_pool, y_pool = update_label_set_core_set(x_train, y_train, x_pool, y_pool,
                                                                         n_samples=100)  # uncertain
            x_train_with_hc, y_train_with_hc = x_train, y_train
            """ core-set -> hc
            x_train, y_train, x_pool, y_pool = update_label_set_core_set(x_train, y_train, x_pool, y_pool, n_samples=100)    # uncertain
            x_train_with_hc, y_train_with_hc = make_trainset_with_hc(x_train, y_train, x_pool, y_pool, name_load=name_save, threshold=0.95)  # certain: cost-effective
            """
        print(acc_arr)  # [90.22, 91.17, 94.95, 94.95, 95.9, 95.27, 99.05, 99.05, 97.79, 97.48]
        exp_result[idx_exp, :] = acc_arr
        save_obj(None, "./log/{}.pkl".format(idx_exp))  # to log
    save_obj(exp_result, "exp_result_core_set_greedy.pkl")

    os.getcwd()
    plt.rcParams["figure.figsize"] = (7, 4)  # 계속 유지됨
    plt.plot(acc_arr, "o-", c="blue")
    plt.ylim([-2, 102])
    plt.savefig("../acc")
    plt.close()

    plt.plot(acc_arr, "o-", c="blue")
    plt.ylim([90, 100])
    plt.savefig("../acc 2")
    plt.close()

    """
    # 4) train again 
    name_load = "model_updating"
    name_save = "model_updating"
    train_model(x_train, y_train, name_load=name_load, name_save=name_save, load_trained_model=False, monitor="val_acc")
    """
