from tf_keras_for_serving import resnet_tf_keras as serving
import tensorflow as tf

import pickle
import copy
import os

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
matplotlib.use("Agg")



def margin_sampling(y_pred_prob, n_samples):
    """ Rank all the unlabeled samples in an ascending order according to the margin sampling """
    origin_index = np.arange(0, len(y_pred_prob))
    margin_sampling = np.diff(-np.sort(y_pred_prob)[:, ::-1][:, :2])
    pred_label = np.argmax(y_pred_prob, axis=1)
    msi = np.column_stack((origin_index,
                           margin_sampling,
                           pred_label))
    msi = msi[msi[:, 1].argsort()]
    return msi[:n_samples], msi[:, 0].astype(int)[:n_samples]


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


def get_idx_of_uncertain_samples(y_pred_prob, n_samples, criteria):
    if criteria == 'lc':
        return least_confidence(y_pred_prob, n_samples)
    elif criteria == 'ms':
        return margin_sampling(y_pred_prob, n_samples)
    elif criteria == 'en':
        return entropy(y_pred_prob, n_samples)
    elif criteria == 'rs':
        return None, random_sampling(y_pred_prob, n_samples)
    else:
        raise ValueError(
            'Unknown criteria value \'%s\', use one of [\'rs\',\'lc\',\'ms\',\'en\']' % criteria)


def random_sampling(y_pred_prob, n_samples):
    """ Random sampling """
    return np.random.choice(range(len(y_pred_prob)), n_samples)


def entropy(y_pred_prob, n_samples):
    """ calculate entropy
        Rank all the unlabeled samples in an descending order according to their entropy

    Parameters
    ----------
    y_pred_prob: y probabilities in the format of one-hot encoding
    n_samples: the number of samples to take

    Returns
    -------
    entropy: entropy whose length is predefined
    idxs_entropy: indexes of selected entropy
    """
    # entropy = stats.entropy(y_pred_prob.T)
    # entropy = np.nan_to_num(entropy)
    origin_index = np.arange(0, len(y_pred_prob))
    entropy = -np.nansum(np.multiply(y_pred_prob, np.log(y_pred_prob)), axis=1)
    pred_label = np.argmax(y_pred_prob, axis=1)
    eni = np.column_stack((origin_index,
                           entropy,
                           pred_label))

    eni = eni[(-eni[:, 1]).argsort()]
    entropy, idxs_entropy = eni[:n_samples], eni[:, 0].astype(int)[:n_samples]
    return entropy, idxs_entropy


def return_y_str(label):
    if label == 0:
        return "particle"
    elif label == 1:
        return "floating"
    elif label == 2:
        return "corona"
    elif label == 3:
        return "void"
    elif label == 4:
        return "noise"


def cnt_uncertains(y, stat_uncertain, idxs_uncertain, threshold=0.5):
    """ count the num of uncertain samples """
    idx_prob = 1
    idxs_selected_uncertain = idxs_uncertain[stat_uncertain[:, idx_prob] < threshold]
    count_dict = dict({"particle": 0, "floating": 0, "corona": 0, "void": 0, "noise": 0})
    for idx in idxs_selected_uncertain:
        label = return_y_str(y[idx])
        count_dict[label] += 1
    return count_dict


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

    # divide data
    # Todo: use only train set 
    x_train, y_train, x_validation, y_validation = divide_train_test_set(x, y_one_hot, ratio_train=0.8, stratify=True)

    # Init
    tf.keras.backend.clear_session()

    # Load Model if exists
    model = serving.resnet18_tf_keras(input_shape, n_classes)   # original: model = ResNet18(input_shape, n_classes)
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
                                                     verbose=1,
                                                     monitor=monitor, mode='max')

    model.fit(x_train, y_train, validation_data=(x_validation, y_validation), batch_size=batch_size,
              shuffle=True, epochs=epochs, verbose=1, callbacks=[early_stop, lr_reducer, check_point])

    """ test
    scores_train_set = model.evaluate(x_train, y_train, batch_size=batch_size, verbose=1)
    print(model.metrics_names, "\n", scores_train_set)
    scores_validation_set = model.evaluate(x_validation, y_validation, batch_size=batch_size, verbose=1)
    print(model.metrics_names, "\n", scores_validation_set)
    """


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
    high_confident_df = high_confident_df[high_confident_df[:, idx_stat] > threshold]
    idx_hc, label_hc = high_confident_df[:, idx_index].astype(int), high_confident_df[:, idx_label].astype(int)
    return idx_hc, label_hc


def update_label_set(x_train, y_train, x_pool, y_pool, prob_pred_arr, n_samples=100):
    """ update labeled data by labeling uncertain data of unlabeled data """
    # Init
    idx_obs = 0

    # uncertain_df: origin_index, margin_sampling, pred_label
    _, idx_uncertain = get_idx_of_uncertain_samples(prob_pred_arr, n_samples=n_samples, criteria="lc")
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


def make_trainset_with_hc(x_train, y_train, x_pool, y_pool, name_load, threshold=0.95):
    """ add labeled data and high coinfient data """
    #   2) Combine 'High Confident Data' and 'Labeled Data' temporarily
    idx_hc, _ = get_high_confidence_samples(x_pool, name_load, threshold=threshold)
    x_train_with_hc, y_train_with_hc = \
        np.concatenate([x_train, x_pool[idx_hc, :, :, :]]), np.concatenate([y_train, y_pool[idx_hc, :]])

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
    data_path = path + "data.pkl"
    x, y = read_image_pickle(data_path)
    x_original = copy.deepcopy(x)
    x = reshape_image_x(x)
    print("Shape of X:", x.shape)
    print("The number of X:", len(x))
    print("Y:", np.unique(y, return_counts=True))

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
    set_hyper_params(x)
    y_one_hot = tf.keras.utils.to_categorical(y)

    #
    # Active Learning
    #

    # 1) initialize data set
    # train: 563 / pool: 2266 / test: 317
    x_train, y_train, x_test, y_test = divide_train_test_set(x, y_one_hot, ratio_train=0.9, stratify=True)
    x_train, y_train, x_pool, y_pool = divide_train_test_set(x_train, y_train, ratio_train=0.2, stratify=True)

    # 2) train
    acc_arr = []
    idx_acc = 1
    x_train_with_hc, y_train_with_hc = None, None
    for idx in range(0, 15):
        """
        idx = 0
        """
        if idx == 0:
            name_load, load_trained_model = None, False
            name_save = "initial_model"
        elif idx == 1:
            name_load, load_trained_model = "initial_model", True
            name_save = "model_updating {}".format(idx)
        else:
            name_load, load_trained_model = "model_updating {}".format(idx-1), True
            name_save = "model_updating {}".format(idx)

        # 2-1) train
        if idx == 0:
            train_model(x_train, y_train, name_load=name_load, name_save=name_save,
                        load_trained_model=load_trained_model, monitor="val_acc")
        else:
            train_model(x_train_with_hc, y_train_with_hc, name_load=name_load, name_save=name_save,
                        load_trained_model=load_trained_model, monitor="val_acc")

        # 2-2) check accuracy
        acc = check_acc(x_test, y_test, name_load=name_save, path='../model_saved/active_learning')    # 84.2 %
        print("The Accuracy of Model {}: {}%".format(idx, acc))
        acc_arr.append(acc)

        # 2-3) uncertain data
        prob_pred_arr = predict_prob(x_pool, name_save)
        x_train, y_train, x_pool, y_pool = update_label_set(x_train, y_train,
                                                            x_pool, y_pool,
                                                            prob_pred_arr,
                                                            n_samples=100)    # uncertain
        x_train_with_hc, y_train_with_hc = make_trainset_with_hc(x_train, y_train,
                                                                 x_pool, y_pool,
                                                                 name_save,
                                                                 threshold=0.95)  # certain

    print(acc_arr)  # [90.22, 91.17, 94.95, 94.95, 95.9, 95.27, 99.05, 99.05, 97.79, 97.48]

    os.getcwd()
    plt.rcParams["figure.figsize"] = (7, 4)
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
