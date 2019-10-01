import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, classification_report
import os
from scipy import misc


class SPARCS_Callback(keras.callbacks.Callback):
    def __init__(self, valid_datasets, valid_datagens, steps_per_epoch=float('inf'), frequency=1):
        keras.callbacks.Callback.__init__(self)
        self.datasets = valid_datasets
        self.datagens = valid_datagens
        self.steps_per_epoch = steps_per_epoch
        self.frequency = frequency

        return

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.frequency == self.frequency-1:
            print('|{0: ^12}|{1: ^12}|{2: ^12}|{3: ^12}|{4: ^12}|{5: ^12}\n -----------------------------------------------------------------'.
                  format('Biome', '% Correct', '% Omission', '% Comission', '% Cloud', '%Shadow'))

            for dataset, gen in zip(self.datasets, self.datagens):
                y_true, y_pred = np.array([]), np.array([])
                biome = os.path.split(dataset.dirs)[-1]
                stepper = 0
                n_samples = 1
                n_pixels = 1
                n_cloud = 1
                n_clear = 1
                n_shadow = 1
                percent_cloud = 0
                percent_shadow = 0
                percent_correct = 0
                percent_omission = 0
                percent_comission = 0
                while n_samples < len(dataset) and stepper < self.steps_per_epoch:
                    x, y_t = next(gen)
                    y_p = self.model.predict(x)
                    y_true = y_t.reshape([-1, y_t.shape[-1]]).argmax(axis=-1)
                    y_pred = y_p.reshape([-1, y_p.shape[-1]]).argmax(axis=-1)

                    total = len(y_true)
                    total_shadow = y_true == 0
                    total_clear = y_true == 1
                    total_cloud = y_true == 2

                    cloud_as_shadow = (y_true == 2) * (y_pred == 0)
                    cloud_as_clear = (y_true == 2) * (y_pred == 1)
                    cloud_as_cloud = (y_true == 2) * (y_pred == 2)

                    clear_as_shadow = (y_true == 1) * (y_pred == 0)
                    clear_as_clear = (y_true == 1) * (y_pred == 1)
                    clear_as_cloud = (y_true == 1) * (y_pred == 2)

                    shadow_as_shadow = (y_true == 0) * (y_pred == 0)
                    shadow_as_clear = (y_true == 0) * (y_pred == 1)
                    shadow_as_cloud = (y_true == 0) * (y_pred == 2)

                    i_percent_cloud = 100*np.sum(total_cloud)/np.sum(total)
                    i_percent_shadow = 100*np.sum(total_shadow)/np.sum(total)
                    i_percent_correct = 100 * \
                        (np.sum(shadow_as_shadow)+np.sum(cloud_as_cloud) +
                         np.sum(clear_as_clear))/np.sum(total)
                    if np.sum(total_cloud):
                        i_percent_omission = 100 * \
                            (np.sum(total_shadow) - np.sum(shadow_as_shadow)) / \
                            np.sum(total_shadow)
                    else:
                        i_percent_omission = 0

                    if np.sum(total_clear):
                        i_percent_comission = 100 * \
                            np.sum(clear_as_shadow+cloud_as_shadow) / \
                            (np.sum(total_clear)+np.sum(total_cloud))
                    else:
                        i_percent_comission = 0

                    percent_cloud = (
                        percent_cloud*n_pixels + i_percent_cloud*np.sum(total))/(n_pixels+np.sum(total))
                    percent_shadow = (
                        percent_shadow*n_pixels + i_percent_shadow*np.sum(total))/(n_pixels+np.sum(total))
                    percent_correct = (
                        percent_correct*n_pixels + i_percent_correct*np.sum(total))/(n_pixels+np.sum(total))
                    percent_omission = (percent_omission*n_shadow + i_percent_omission*np.sum(
                        total_shadow))/(n_shadow+np.sum(total_shadow))
                    percent_comission = (percent_comission*(n_clear+n_cloud) + i_percent_comission*(np.sum(
                        total_clear)+np.sum(total_cloud)))/(n_clear+n_cloud+np.sum(total_clear)+np.sum(total_cloud))

                    n_pixels += np.sum(total)
                    n_cloud += np.sum(total_cloud)
                    n_clear += np.sum(total_clear)
                    n_shadow += np.sum(total_shadow)

                    stepper += 1
                    n_samples += x.shape[0]
                    print(' {0: ^12},{1: ^12},{2: ^12},{3: ^12},{4: ^12},{5: ^12}, '.format(biome, np.round(percent_correct, 3), np.round(
                        percent_omission, 3), np.round(percent_comission, 3), np.round(percent_cloud, 3), np.round(percent_shadow, 3)), n_samples, end='\r')

                print(' {0: ^12},{1: ^12},{2: ^12},{3: ^12},{4: ^12},{5: ^12}, '.format(biome, np.round(percent_correct, 3), np.round(
                    percent_omission, 3), np.round(percent_comission, 3), np.round(percent_cloud, 3), np.round(percent_shadow, 3)))
        self.model.save(
            "./models/shadow_multi/epoch{}_model_split1.H5".format(epoch))
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


class foga_table5_Callback_no_thin(keras.callbacks.Callback):
    def __init__(self, valid_datasets, valid_datagens, steps_per_epoch=float('inf'), frequency=1):
        keras.callbacks.Callback.__init__(self)
        self.datasets = valid_datasets
        self.datagens = valid_datagens
        self.steps_per_epoch = steps_per_epoch
        self.frequency = frequency

        return

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.frequency == self.frequency-1:
            print('|{0: ^12}|{1: ^12}|{2: ^12}|{3: ^12}|{4: ^12}|{5: ^12}|\n ------------------------------------------------------------------------------'.
                  format('Biome', '% Correct', '% Omission', '% Comission', '% Cloud', 'N. Samples'))

            for dataset, gen in zip(self.datasets, self.datagens):
                y_true, y_pred = np.array([]), np.array([])
                biome = os.path.split(dataset.dirs)[-1]
                stepper = 0
                n_samples = 1
                n_pixels = 1
                n_cloud = 1
                n_clear = 1
                percent_cloud = 0
                percent_correct = 0
                percent_omission = 0
                percent_comission = 0
                while n_samples < len(dataset) and stepper < self.steps_per_epoch:
                    x, y_t = next(gen)
                    y_p = self.model.predict(x)
                    y_true = y_t.reshape([-1, y_t.shape[-1]]).argmax(axis=-1)
                    y_pred = y_p.reshape([-1, y_p.shape[-1]]).argmax(axis=-1)

                    total = y_true != 0  # exclude fill pixels
                    cloud_as_cloud = (y_true == 2) * (y_pred == 2)
                    clear_as_clear = (y_true == 1) * (y_pred == 1)
                    clear_as_cloud = (y_true == 1) * (y_pred == 2)
                    cloud_as_clear = (y_true == 2) * (y_pred == 1)

                    total_clear = y_true == 1
                    total_cloud = y_true == 2

                    i_percent_cloud = 100*np.sum(total_cloud)/np.sum(total)
                    i_percent_correct = 100 * \
                        (np.sum(cloud_as_cloud)+np.sum(clear_as_clear))/np.sum(total)
                    if np.sum(total_cloud):
                        i_percent_omission = 100 * \
                            (np.sum(total_cloud) - np.sum(cloud_as_cloud)) / \
                            np.sum(total_cloud)
                    else:
                        i_percent_omission = 0

                    if np.sum(total_clear):
                        i_percent_comission = 100 * \
                            np.sum(clear_as_cloud)/np.sum(total_clear)
                    else:
                        i_percent_comission = 0

                    percent_cloud = (
                        percent_cloud*n_pixels + i_percent_cloud*np.sum(total))/(n_pixels+np.sum(total))
                    percent_correct = (
                        percent_correct*n_pixels + i_percent_correct*np.sum(total))/(n_pixels+np.sum(total))
                    percent_omission = (percent_omission*n_cloud + i_percent_omission *
                                        np.sum(total_cloud))/(n_cloud+np.sum(total_cloud))
                    percent_comission = (percent_comission*n_clear + i_percent_comission*np.sum(
                        total_clear))/(n_clear+np.sum(total_clear))

                    n_pixels += np.sum(total)
                    n_cloud += np.sum(total_cloud)
                    n_clear += np.sum(total_clear)

                    stepper += 1
                    n_samples += x.shape[0]
                    print(' {0: ^12},{1: ^12},{2: ^12},{3: ^12},{4: ^12}, '.format(biome, np.round(percent_correct, 3), np.round(
                        percent_omission, 3), np.round(percent_comission, 3), np.round(percent_cloud, 3)), n_samples-1, end='\r')

                print(' {0: ^12},{1: ^12},{2: ^12},{3: ^12},{4: ^12}, '.format(biome, np.round(percent_correct, 3), np.round(
                    percent_omission, 3), np.round(percent_comission, 3), np.round(percent_cloud, 3)))
        self.model.save(
            "../results/foga_multi_SNOWONLY/epoch{}_model_split1.H5".format(epoch))

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
