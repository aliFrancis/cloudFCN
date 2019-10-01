from keras import backend as K
import numpy as np


def w_categorical_crossentropy(y_true, y_pred, weights):
    """
    Keras-style categorical crossentropy loss function, with weighting for each class.

    Parameters
    ----------
    y_true : Tensor
        Truth labels.
    y_pred : Tensor
        Predicted values.
    weights: Tensor
        Multiplicative factor for loss per class.

    Returns
    -------
    loss : Tensor
        Weighted crossentropy loss between labels and predictions.

    """
    y_true_max = K.argmax(y_true, axis=-1)
    weighted_true = K.gather(weights, y_true_max)
    loss = K.categorical_crossentropy(y_pred, y_true) * weighted_true
    return loss
