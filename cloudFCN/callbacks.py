import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import tensorflow as tf
import keras
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, classification_report
import os
from scipy import misc



class metric_Callback(keras.callbacks.Callback):
    """
    Callback for statistical metrics on validation data. Includes F1 and confusion matrix.


    Attributes
    ----------
    datagen : iterator
        Yields image/mask pairs from validation Dataset.
    steps_per_epoch : int, optional
        Number of batches to be used.
    frequency : int, optional
        Number of epochs between each validation.
    classes : list, optional
        Output classes.
        
    """

    def __init__(self, datagen, steps_per_epoch=10, frequency=1, class_labels=['Fill', 'Clear', 'Cloud'],unlabelled_mask=False):
        keras.callbacks.Callback.__init__(self)
        self.datagen = datagen
        self.steps_per_epoch = steps_per_epoch
        self.frequency = frequency
        self.class_labels = class_labels
        self.unlabelled_mask = unlabelled_mask
        return

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.frequency == self.frequency-1:
            y_true, y_pred = np.array([]), np.array([])
            for i in range(self.steps_per_epoch):
                x, y_t = next(self.datagen)
                y_p = self.model.predict(x)
                n_classes = y_t.shape[-1]
                y_t = y_t.reshape([-1, n_classes]).argmax(axis=-1)
                y_p = y_p.reshape([-1, n_classes]).argmax(axis=-1)
                if self.unlabelled_mask: #Last mask pane is unlabelled class
                    labelled_idxs = y_t!=n_classes-1
                    y_t = y_t[labelled_idxs]
                    y_p = y_p[labelled_idxs]
                y_true = np.concatenate((y_true, y_t))
                y_pred = np.concatenate((y_pred, y_p))

            conf_mat = confusion_matrix(y_true, y_pred, [i for i in range(len(self.class_labels)-int(self.unlabelled_mask))])
            class_rep = classification_report(y_true, y_pred, labels=[
                                              i for i in range(len(self.class_labels))], target_names=self.class_labels)
            print(conf_mat)
            print(class_rep)
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return


class visual_Callback(keras.callbacks.Callback):
    """
    Callback for displaying visual examples of cloud masks from validation data.


    Attributes
    ----------
    datagen : iterator
        Yields image/mask pairs from validation Dataset.
    RGB_bands : list, optional
        Spectral bands of image to use for RGB.

    """

    def __init__(self, datagen, RGB_bands=[0, 1, 2],class_labels=None):
        keras.callbacks.Callback.__init__(self)
        self.datagen = datagen
        self.RGB_bands = RGB_bands
        self.class_labels = class_labels
        return

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        ims, masks = next(self.datagen)
        num_plots = min(6, ims.shape[0])
        fig, ax = plt.subplots(3, num_plots)
        jet_colors = cm.get_cmap('jet',masks.shape[-1])
        for i in range(num_plots):
            ims, masks = next(self.datagen)
            im, mask = ims[0, ...], masks[0, ...]
            pred = self.model.predict(im[np.newaxis, ...])
            im_display = im[..., self.RGB_bands].astype('float')
            im_display = (im_display-im_display.min()) / \
                (im_display.max()-im_display.min())
            mask_display = np.argmax(mask, axis=-1)
            pred_display = np.argmax(pred, axis=-1)
            ax[0, i].imshow(im_display)
            ax[1, i].imshow(np.squeeze(pred_display).astype(
                'float'), vmin=0, vmax=masks.shape[-1], cmap=jet_colors)
            last_im = ax[2, i].imshow(np.squeeze(mask_display).astype(
                'float'), vmin=0, vmax=masks.shape[-1], cmap=jet_colors)
            ax[0, i].set_axis_off()
            ax[1, i].set_axis_off()
            ax[2, i].set_axis_off()
            ax[0, i].set_title('Input')
            ax[1, i].set_title('Pred')
            ax[2, i].set_title('Truth')

            cax = fig.add_axes([0.2, 0.05, 0.6, 0.02])

            cbar = fig.colorbar(last_im, cax=cax, orientation='horizontal')
            if self.class_labels is not None:
                cbar.set_ticks([0.5+i for i in range(len(self.class_labels))])
                cbar.set_ticklabels(self.class_labels)
        plt.show()

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
