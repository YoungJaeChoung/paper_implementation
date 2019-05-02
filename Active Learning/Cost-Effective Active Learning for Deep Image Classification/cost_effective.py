""" Code from: https://github.com/dhaalves/CEAL_keras/blob/master/CEAL_keras.py"""

import argparse
import os

import numpy as np
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.datasets import cifar10
from keras.models import load_model
from keras.utils import np_utils
from keras_contrib.applications.resnet import ResNet18
from sklearn.model_selection import train_test_split

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

# os.makedirs("./figures")


def initialize_dataset():
    idx_row = 0

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    """ shape 
    x_train.shape   # (50000, 32, 32, 3)
    y_train.shape   # (50000, 1)
    x_test.shape    # (10000, 32, 32, 3)
    y_test.shape    # (10000, 1)
    
    idx_file = 0
    print(y_train[idx_file])
    im_r = x_train[idx_file, :, :, 0]/255.0
    im_g = x_train[idx_file, :, :, 1]/255.0
    im_b = x_train[idx_file, :, :, 2]/255.0
    img = np.dstack((im_r, im_g, im_b))
    plt.imshow(img)
    plt.savefig("./figures/test {}".format(idx_file))
    plt.close()
    """
    n_classes = np.max(y_test) + 1

    # Convert class vectors to binary class matrices.
    y_train = np_utils.to_categorical(y_train, n_classes)   # one hot
    y_test = np_utils.to_categorical(y_test, n_classes)     # one hot
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # subtract mean and normalize
    mean_image = np.mean(x_train, axis=idx_row)   # (32, 32, 3)
    x_train -= mean_image
    x_test -= mean_image
    x_train /= 128.
    x_test /= 128.

    initial_train_size = int(x_train.shape[idx_row] * args.initial_annotated_percent)
    x_pool, x_initial, y_pool, y_initial = train_test_split(x_train, y_train, test_size=initial_train_size,
                                                            random_state=1, stratify=y_train)

    return x_pool, y_pool, x_initial, y_initial, x_test, y_test, n_classes


def initialize_model(x_initial, y_initial, x_test, y_test, n_classes):
    if os.path.exists(args.chkt_filename):
        model = load_model(args.chkt_filename)
    else:
        input_shape = (x_initial[-1,].shape)
        model = ResNet18(input_shape, n_classes)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=3, min_lr=0.5e-6)
        checkpoint = ModelCheckpoint(args.chkt_filename, monitor='val_acc', save_best_only=True)
        model.fit(x_initial, y_initial, validation_data=(x_test, y_test), batch_size=args.batch_size,
                  shuffle=True, epochs=args.epochs, verbose=args.verbose, callbacks=[lr_reducer, checkpoint])

    scores = model.evaluate(x_test, y_test, batch_size=args.batch_size, verbose=args.verbose)
    print('Initial Test Loss: ', scores[0], ' Initial Test Accuracy: ', scores[1])
    return model


# Random sampling
def random_sampling(y_pred_prob, n_samples):
    return np.random.choice(range(len(y_pred_prob)), n_samples)


# Rank all the unlabeled samples in an ascending order according to the least confidence
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


# Rank all the unlabeled samples in an ascending order according to the margin sampling
def margin_sampling(y_pred_prob, n_samples):
    origin_index = np.arange(0, len(y_pred_prob))
    margim_sampling = np.diff(-np.sort(y_pred_prob)[:, ::-1][:, :2])
    pred_label = np.argmax(y_pred_prob, axis=1)
    msi = np.column_stack((origin_index,
                           margim_sampling,
                           pred_label))
    msi = msi[msi[:, 1].argsort()]
    return msi[:n_samples], msi[:, 0].astype(int)[:n_samples]


# Rank all the unlabeled samples in an descending order according to their entropy
def entropy(y_pred_prob, n_samples):
    """ calculate entropy

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


def get_high_confidence_samples(y_pred_prob, delta):
    eni, eni_idx = entropy(y_pred_prob, len(y_pred_prob))
    hcs = eni[eni[:, 1] < delta]
    return hcs[:, 0].astype(int), hcs[:, 2].astype(int)


def get_uncertain_samples(y_pred_prob, n_samples, criteria):
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


def run_ceal(args):
    idx_x = 0
    idx_y = 1
    row_wise = 0

    x_pool, y_pool, x_initial, y_initial, x_test, y_test, n_classes = initialize_dataset()

    model = initialize_model(x_initial, y_initial, x_test, y_test, n_classes)

    w, h, c = x_pool[-1, ].shape    # width, height, channel

    # unlabeled samples
    DU = x_pool, y_pool

    # initially labeled samples
    DL = x_initial, y_initial  # np.empty((0, w, h, c)), np.empty((0, n_classes))

    # high confidence samples
    DH = np.empty((0, w, h, c)), np.empty((0, n_classes))

    for iter in range(args.maximum_iterations):     # this is somewhat different concept of epoch
        # update labels
        if iter % args.Interval_update_labels:
            y_pred_prob = model.predict(DU[idx_x], verbose=args.verbose)  # (45000, 10)

            # update Labeled Data-set
            # by labeling ambiguous unlabeled data
            _, idxs_uncertain = get_uncertain_samples(y_pred_prob, args.uncertain_samples_size,
                                                      criteria=args.uncertain_criteria)

            DL_x = DL[idx_x]
            DL_y = DL[idx_y]
            selected_Du_x = np.take(DU[idx_x], idxs_uncertain, axis=row_wise)
            selected_Du_y = np.take(DU[idx_y], idxs_uncertain, axis=row_wise)
            DL = np.append(DL_x, selected_Du_x, axis=row_wise), np.append(DL_y, selected_Du_y, axis=row_wise)

            # Todo: Balancing Data set ... ?
            # hc: highly confident (unlabeled data)
            idxs_hc, labels_hc = get_high_confidence_samples(y_pred_prob,
                                                             args.high_confidence_sample_selection_threshold)

            # update high confident threshold
            if args.cnt_label_updated % args.fine_tunning_interval == 0:
                args.high_confidence_sample_selection_threshold -= args.threshold_decay
                args.cnt_label_updated = 0
            else:
                args.cnt_label_updated += 1

            # remove samples also selected through uncertain
            # ... hc: highly confident
            hc = np.array([[idx_hc, label_hc] for idx_hc, label_hc in zip(idxs_hc, labels_hc) if idx_hc not in idxs_uncertain])
            idx_index = 0
            idx_label = 1
            if hc.size != 0:
                Du_x = DU[idx_x]
                highly_confident_Du_x = np.take(Du_x, hc[:, idx_index], axis=0)
                label_hc_onehot = np_utils.to_categorical(hc[:, idx_label], n_classes)

                DH = highly_confident_Du_x, label_hc_onehot

            dtrain_x = np.concatenate((DL[idx_x], DH[idx_x])) if DH[idx_x].size != 0 else DL[idx_x]
            dtrain_y = np.concatenate((DL[idx_y], DH[idx_y])) if DH[idx_y].size != 0 else DL[idx_y]
        else:
            dtrain_x = DL[idx_x]
            dtrain_y = DL[idx_y]

        # train
        early_stop = EarlyStopping(monitor='val_loss', patience=1)
        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=3, min_lr=0.5e-6)
        checkpoint = ModelCheckpoint(args.chkt_filename, monitor='val_acc', save_best_only=True)
        model.fit(dtrain_x, dtrain_y, validation_data=(x_test, y_test), batch_size=args.batch_size,
                  shuffle=True, epochs=args.epochs, verbose=args.verbose,
                  callbacks=[lr_reducer, checkpoint, early_stop])

        # reset unlabeled, high confident data
        if iter % args.Interval_update_labels:
            DU = np.delete(DU[idx_x], idxs_uncertain, axis=row_wise), \
                 np.delete(DU[idx_y], idxs_uncertain, axis=row_wise)
            DH = np.empty((0, w, h, c)), np.empty((0, n_classes))

        # evaluate
        _, acc = model.evaluate(x_test, y_test, batch_size=args.batch_size, verbose=args.verbose)
        print(
            'Iteration: %d; High Confidence Samples: %d; Uncertain Samples: '
            '%d; high_confidence_sample_selection_threshold: %.5f; Labeled Dataset Size: %d; Accuracy: %.2f'
            % (iter, len(DH[0]), len(DL[0]), args.high_confidence_sample_selection_threshold, len(DL[0]), acc))


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


if __name__ == '__main__':
    args = Args()
    run_ceal(args)

    """ argparse original parameters 
    parser = argparse.ArgumentParser()
    parser.add_argument('-verbose', default=0, type=int,
                        help="Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. default: 0")
    parser.add_argument('-epochs', default=5, type=int, help="Number of epoch to train. default: 5")
    parser.add_argument('-batch_size', default=32, type=int, help="Number of samples per gradient update. default: 32")
    parser.add_argument('-chkt_filename', default="ResNet18v2-CIFAR-10_init_ceal.hdf5",
                        help="Model Checkpoint filename to save")
    parser.add_argument('-t', '--fine_tunning_interval', default=1, type=int, help="Fine-tuning interval. default: 1")
    parser.add_argument('-T', '--maximum_iterations', default=45, type=int,
                        help="Maximum iteration number. default: 10")
    parser.add_argument('-i', '--initial_annotated_perc', default=0.1, type=float,
                        help="Initial Annotated Samples Percentage. default: 0.1")
    parser.add_argument('-dr', '--threshold_decay', default=0.0033, type=float,
                        help="Threshold decay rate. default: 0.0033")
    parser.add_argument('-delta', default=0.05, type=float,
                        help="High confidence samples selection threshold. default: 0.05")
    parser.add_argument('-K', '--uncertain_samples_size', default=1000, type=int,
                        help="Uncertain samples selection size. default: 1000")
    parser.add_argument('-uc', '--uncertain_criteria', default='lc',
                        help="Uncertain selection Criteria: \'rs\'(Random Sampling), \'lc\'(Least Confidence), \'ms\'(Margin Sampling), \'en\'(Entropy). default: lc")
    parser.add_argument('-ce', '--cost_effective', default=True,
                        help="whether to use Cost Effective high confidence sample pseudo-labeling. default: True")
    args = parser.parse_args()
    """

    """ parameters
    epochs = 5
    batch_size = 32
    fine_tunning_interval = 1
    max_iter = 10   # 45
    initial_annotated_percent = 0.1
    threshold_decay = 0.0033
    high_confidence_samples_selection_threshold = 0.05
    uncertain_sample_size = 1000
    uncertain_critreia = "en"
    cost_effective = True
    chkt_filename = "ResNet18v2-CIFAR-10_init_ceal.hdf5" 
    """
