from keras.layers import Input, Conv2D, Concatenate, BatchNormalization, \
    LeakyReLU, Conv2DTranspose, Activation, Reshape
from keras.models import Model
from keras import regularizers


def build_model1(batch_norm=True, num_channels=12, num_classes=1):
    """
    Builds a 760 parameter model, min. input size = 1-by-1

    Parameters
    ----------
    batch_norm : bool, optional
        True for batch_norm layers to be used. False otherwise.
    num_channels : int, optional
        Number of spectral bands in inputs.
    num_classes : int, optional
        Number of output classes.

    Returns
    -------
    model : keras.models.Model
        Uncompiled keras model.
    """
    inputs = Input(shape=(None, None, num_channels))

    x = Conv2D(20, (1, 1), activation='linear',
               kernel_initializer='glorot_uniform')(inputs)
    x = LeakyReLU(alpha=0.1)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    x = Conv2D(20, (1, 1), activation='linear',
               kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.1)(x)

    outputs = Conv2D(num_classes, (1, 1), strides=1, padding='SAME', activation='linear',
                     kernel_initializer='glorot_uniform')(x)
    if num_classes > 1:
        outputs = Activation('softmax')(outputs)

    return Model(inputs=inputs, outputs=outputs)


def build_model2(batch_norm=True, num_channels=12, num_classes=1):
    """
    Builds a 64 thousand parameter model, min. input size = 2-by-2 (recommended atleast 16-by-16)

    Parameters
    ----------
    batch_norm : bool, optional
        True for batch_norm layers to be used. False otherwise.
    num_channels : int, optional
        Number of spectral bands in inputs.
    num_classes : int, optional
        Number of output classes.

    Returns
    -------
    model : keras.models.Model
        Uncompiled keras model.
    """
    inputs = Input(shape=(None, None, num_channels))
    # ------LOCAL INFORMATION GATHERING

    x0 = Conv2D(12, (3, 3), strides=1, padding='SAME', activation='tanh',
                kernel_initializer='glorot_uniform')(inputs)

    x1 = Conv2D(16, (5, 5), strides=1, padding='SAME', activation='tanh',
                kernel_initializer='glorot_uniform')(inputs)

    x_in = Concatenate()([inputs, x0, x1])

    x_in = Conv2D(24, (1, 1), strides=1, padding='VALID', activation='tanh',
                  kernel_initializer='glorot_uniform')(x_in)
    if batch_norm:
        x_in = BatchNormalization(axis=-1, momentum=0.99)(x_in)

    x_in = Conv2D(18, (1, 1), strides=1, padding='VALID', activation='tanh',
                  kernel_initializer='glorot_uniform')(x_in)
    if batch_norm:
        x_in = BatchNormalization(
            axis=-1, momentum=0.99)(x_in)

    x_RES_1 = Conv2D(9, (1, 1), strides=1, padding='VALID', activation='tanh',
                     kernel_initializer='glorot_uniform')(x_in)
    if batch_norm:
        x_RES_1 = BatchNormalization(axis=-1, momentum=0.99)(x_RES_1)

    # =================================

    x = Conv2D(64, (5, 5), strides=2, padding='SAME', activation='linear',
               kernel_initializer='glorot_uniform')(x_RES_1)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    x = Conv2DTranspose(24, (5, 5), strides=2, padding='SAME', activation='linear',
                        kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    x = Concatenate()([x, x_RES_1, inputs])

    # =============FCL-type convolutions at each pixel...

    x = Conv2DTranspose(32, (1, 1), strides=1, padding='SAME', activation='linear',
                        kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    x = Conv2DTranspose(32, (1, 1), strides=1, padding='SAME', activation='linear',
                        kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    x = Conv2DTranspose(12, (1, 1), strides=1, padding='SAME', activation='linear',
                        kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    x = Conv2DTranspose(4, (1, 1), strides=1, padding='SAME', activation='linear',
                        kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    X0 = Conv2DTranspose(3, (3, 3), strides=1, padding='SAME', activation='linear',
                         kernel_initializer='glorot_uniform')(x)
    X0 = LeakyReLU(alpha=0.01)(X0)

    x = Concatenate()([x, X0, inputs])  # even more local info
    x = Conv2DTranspose(3, (1, 1), strides=1, padding='SAME', activation='linear',
                        kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)

    outputs = Conv2DTranspose(num_classes, (1, 1), strides=1, padding='SAME', activation='linear',
                              kernel_initializer='glorot_uniform')(x)
    if num_classes > 1:
        outputs = Activation('softmax')(outputs)
    return Model(inputs=inputs, outputs=outputs)


def build_model3(batch_norm=True, num_channels=12, num_classes=1):
    """
    Builds a 2-million parameter model, min. input size = 38-by-38

    Parameters
    ----------
    batch_norm : bool, optional
        True for batch_norm layers to be used. False otherwise.
    num_channels : int, optional
        Number of spectral bands in inputs.
    num_classes : int, optional
        Number of output classes.

    Returns
    -------
    model : keras.models.Model
        Uncompiled keras model.
    """
    inputs = Input(shape=(None, None, num_channels))
    # ------LOCAL INFORMATION GATHERING

    x0 = Conv2D(12, (3, 3), strides=1, padding='SAME', activation='tanh',
                kernel_initializer='glorot_uniform')(inputs)

    x1 = Conv2D(16, (5, 5), strides=1, padding='SAME', activation='tanh',
                kernel_initializer='glorot_uniform')(inputs)

    x_in = Concatenate()([inputs, x0, x1])

    x_in = Conv2D(24, (1, 1), strides=1, padding='VALID', activation='tanh',
                  kernel_initializer='glorot_uniform')(x_in)
    if batch_norm:
        x_in = BatchNormalization(axis=-1, momentum=0.99)(x_in)

    x_in = Conv2D(18, (1, 1), strides=1, padding='VALID', activation='tanh',
                  kernel_initializer='glorot_uniform')(x_in)
    if batch_norm:
        x_in = BatchNormalization(
            axis=-1, momentum=0.99)(x_in)

    x_RES_1 = Conv2D(9, (1, 1), strides=1, padding='VALID', activation='tanh',
                     kernel_initializer='glorot_uniform')(x_in)
    if batch_norm:
        x_RES_1 = BatchNormalization(axis=-1, momentum=0.99)(x_RES_1)

    # =================================

    x = Conv2D(64, (5, 5), strides=2, padding='SAME', activation='linear',
               kernel_initializer='glorot_uniform')(x_RES_1)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    x = Conv2D(128, (7, 7), strides=3, padding='VALID', activation='linear',
               kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    x_RES_2 = Conv2D(64, (1, 1), strides=1, padding='VALID', activation='linear',
                     kernel_initializer='glorot_uniform')(x)
    x_RES_2 = LeakyReLU(alpha=0.01)(x_RES_2)
    if batch_norm:
        x_RES_2 = BatchNormalization(
            axis=-1, momentum=0.99)(x_RES_2)

    x = Conv2D(256, (5, 5), strides=2, padding='VALID', activation='linear',
               kernel_initializer='glorot_uniform')(x_RES_2)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    x = Conv2DTranspose(96, (5, 5), strides=2, padding='VALID', activation='linear',
                        kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    x = Concatenate()([x, x_RES_2])
    x = Conv2D(128, (1, 1), strides=1, padding='VALID', activation='linear',
               kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    x = Conv2DTranspose(72, (7, 7), strides=3, padding='VALID', activation='linear',
                        kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    x = Conv2D(48, (1, 1), strides=1, padding='VALID', activation='linear',
               kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    x = Conv2DTranspose(24, (5, 5), strides=2, padding='SAME', activation='linear',
                        kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    x = Concatenate()([x, x_RES_1, inputs])

    # =============FCL-type convolutions at each pixel...

    x = Conv2DTranspose(32, (1, 1), strides=1, padding='SAME', activation='linear',
                        kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    x = Conv2DTranspose(32, (1, 1), strides=1, padding='SAME', activation='linear',
                        kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    x = Conv2DTranspose(12, (1, 1), strides=1, padding='SAME', activation='linear',
                        kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    x = Conv2DTranspose(4, (1, 1), strides=1, padding='SAME', activation='linear',
                        kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    X0 = Conv2DTranspose(3, (3, 3), strides=1, padding='SAME', activation='linear',
                         kernel_initializer='glorot_uniform')(x)
    X0 = LeakyReLU(alpha=0.01)(X0)

    x = Concatenate()([x, X0, inputs])  # even more local info
    x = Conv2DTranspose(3, (1, 1), strides=1, padding='SAME', activation='linear',
                        kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)

    outputs = Conv2DTranspose(num_classes, (1, 1), strides=1, padding='SAME', activation='linear',
                              kernel_initializer='glorot_uniform')(x)
    if num_classes > 1:
        outputs = Activation('softmax')(outputs)
    return Model(inputs=inputs, outputs=outputs)


def build_model4(batch_norm=True, num_channels=12, num_classes=1):
    """
    Builds a 4.4 million parameter model, min. input size = 86-by-86

    Parameters
    ----------
    batch_norm : bool, optional
        True for batch_norm layers to be used. False otherwise.
    num_channels : int, optional
        Number of spectral bands in inputs.
    num_classes : int, optional
        Number of output classes.

    Returns
    -------
    model : keras.models.Model
        Uncompiled keras model.
    """
    inputs = Input(shape=(None, None, num_channels))
    # ------LOCAL INFORMATION GATHERING

    x0 = Conv2D(12, (3, 3), strides=1, padding='SAME', activation='tanh',
                kernel_initializer='glorot_uniform')(inputs)

    x1 = Conv2D(16, (5, 5), strides=1, padding='SAME', activation='tanh',
                kernel_initializer='glorot_uniform')(inputs)

    x_in = Concatenate()([inputs, x0, x1])

    x_in = Conv2D(24, (1, 1), strides=1, padding='VALID', activation='tanh',
                  kernel_initializer='glorot_uniform')(x_in)
    if batch_norm:
        x_in = BatchNormalization(axis=-1, momentum=0.99)(x_in)

    x_in = Conv2D(18, (1, 1), strides=1, padding='VALID', activation='tanh',
                  kernel_initializer='glorot_uniform')(x_in)
    if batch_norm:
        x_in = BatchNormalization(
            axis=-1, momentum=0.99)(x_in)

    x_RES_1 = Conv2D(9, (1, 1), strides=1, padding='VALID', activation='tanh',
                     kernel_initializer='glorot_uniform')(x_in)
    if batch_norm:
        x_RES_1 = BatchNormalization(axis=-1, momentum=0.99)(x_RES_1)

    # =================================

    x = Conv2D(64, (5, 5), strides=2, padding='SAME', activation='linear',
               kernel_initializer='glorot_uniform')(x_RES_1)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    x = Conv2D(128, (7, 7), strides=3, padding='VALID', activation='linear',
               kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    x_RES_2 = Conv2D(64, (1, 1), strides=1, padding='VALID', activation='linear',
                     kernel_initializer='glorot_uniform')(x)
    x_RES_2 = LeakyReLU(alpha=0.01)(x_RES_2)
    if batch_norm:
        x_RES_2 = BatchNormalization(
            axis=-1, momentum=0.99)(x_RES_2)

    x = Conv2D(256, (5, 5), strides=2, padding='VALID', activation='linear',
               kernel_initializer='glorot_uniform')(x_RES_2)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    x = Conv2D(128, (1, 1), strides=1, padding='VALID', activation='linear',
               kernel_initializer='glorot_uniform')(x)
    x_RES_3 = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x_RES_3 = BatchNormalization(
            axis=-1, momentum=0.99)(x_RES_3)

    x = Conv2D(392, (5, 5), strides=2, padding='VALID', activation='linear',
               kernel_initializer='glorot_uniform')(x_RES_3)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    # -----------------------CODE LAYER

    x = Conv2DTranspose(192, (5, 5), strides=2, padding='VALID', activation='linear',
                        kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    x = Concatenate()([x, x_RES_3])
    x = Conv2D(128, (1, 1), strides=1, padding='VALID', activation='linear',
               kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    x = Conv2DTranspose(96, (5, 5), strides=2, padding='VALID', activation='linear',
                        kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    x = Concatenate()([x, x_RES_2])
    x = Conv2D(128, (1, 1), strides=1, padding='VALID', activation='linear',
               kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    x = Conv2DTranspose(72, (7, 7), strides=3, padding='VALID', activation='linear',
                        kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    x = Conv2D(48, (1, 1), strides=1, padding='VALID', activation='linear',
               kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    x = Conv2DTranspose(24, (5, 5), strides=2, padding='SAME', activation='linear',
                        kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    x = Concatenate()([x, x_RES_1, inputs])

    # =============FCL-type convolutions at each pixel...

    x = Conv2DTranspose(32, (1, 1), strides=1, padding='SAME', activation='linear',
                        kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    x = Conv2DTranspose(32, (1, 1), strides=1, padding='SAME', activation='linear',
                        kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    x = Conv2DTranspose(12, (1, 1), strides=1, padding='SAME', activation='linear',
                        kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    x = Conv2DTranspose(4, (1, 1), strides=1, padding='SAME', activation='linear',
                        kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    X0 = Conv2DTranspose(3, (3, 3), strides=1, padding='SAME', activation='linear',
                         kernel_initializer='glorot_uniform')(x)
    X0 = LeakyReLU(alpha=0.01)(X0)

    x = Concatenate()([x, X0, inputs])  # even more local info
    x = Conv2DTranspose(3, (1, 1), strides=1, padding='SAME', activation='linear',
                        kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)

    outputs = Conv2DTranspose(num_classes, (1, 1), strides=1, padding='SAME', activation='linear',
                              kernel_initializer='glorot_uniform')(x)
    if num_classes > 1:
        outputs = Activation('softmax')(outputs)
    return Model(inputs=inputs, outputs=outputs)


def build_model5(batch_norm=True, num_channels=12, num_classes=1):
    """
    Builds a 7.7 million parameter model, min. input size = 86-by-86

    Parameters
    ----------
    batch_norm : bool, optional
        True for batch_norm layers to be used. False otherwise.
    num_channels : int, optional
        Number of spectral bands in inputs.
    num_classes : int, optional
        Number of output classes.

    Returns
    -------
    model : keras.models.Model
        Uncompiled keras model.
    """
    inputs = Input(shape=(None, None, num_channels))
    # ------LOCAL INFORMATION GATHERING

    x0 = Conv2D(12, (3, 3), strides=1, padding='SAME', activation='tanh',
                kernel_initializer='glorot_uniform')(inputs)

    x1 = Conv2D(16, (7, 7), strides=1, padding='SAME', activation='tanh',
                kernel_initializer='glorot_uniform')(inputs)

    x_in = Concatenate()([inputs, x0, x1])

    x_in = Conv2D(31, (1, 1), strides=1, padding='VALID', activation='tanh',
                  kernel_initializer='glorot_uniform')(x_in)
    if batch_norm:
        x_in = BatchNormalization(axis=-1, momentum=0.99)(x_in)

    x_in = Conv2D(18, (1, 1), strides=1, padding='VALID', activation='tanh',
                  kernel_initializer='glorot_uniform')(x_in)
    if batch_norm:
        x_in = BatchNormalization(
            axis=-1, momentum=0.99)(x_in)

    x_RES_1 = Conv2D(9, (1, 1), strides=1, padding='VALID', activation='tanh',
                     kernel_initializer='glorot_uniform')(x_in)
    if batch_norm:
        x_RES_1 = BatchNormalization(axis=-1, momentum=0.99)(x_RES_1)

    # =================================

    x = Conv2D(64, (5, 5), strides=2, padding='SAME', activation='linear',
               kernel_initializer='glorot_uniform')(x_RES_1)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    x = Conv2D(128, (7, 7), strides=3, padding='VALID', activation='linear',
               kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    x_RES_2 = Conv2D(64, (1, 1), strides=1, padding='VALID', activation='linear',
                     kernel_initializer='glorot_uniform')(x)
    x_RES_2 = LeakyReLU(alpha=0.01)(x_RES_2)
    if batch_norm:
        x_RES_2 = BatchNormalization(
            axis=-1, momentum=0.99)(x_RES_2)

    x = Conv2D(256, (5, 5), strides=2, padding='VALID', activation='linear',
               kernel_initializer='glorot_uniform')(x_RES_2)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    x = Conv2D(128, (1, 1), strides=1, padding='VALID', activation='linear',
               kernel_initializer='glorot_uniform')(x)
    x_RES_3 = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x_RES_3 = BatchNormalization(
            axis=-1, momentum=0.99)(x_RES_3)

    x = Conv2D(512, (5, 5), strides=2, padding='VALID', activation='linear',
               kernel_initializer='glorot_uniform')(x_RES_3)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    # -----------------------CODE LAYER

    x = Conv2DTranspose(256, (5, 5), strides=2, padding='VALID', activation='linear',
                        kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    x = Concatenate()([x, x_RES_3])
    x = Conv2D(192, (1, 1), strides=1, padding='VALID', activation='linear',
               kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    x = Conv2DTranspose(128, (5, 5), strides=2, padding='VALID', activation='linear',
                        kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    x = Concatenate()([x, x_RES_2])
    x = Conv2DTranspose(128, (7, 7), strides=3, padding='VALID', activation='linear',
                        kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    x = Conv2D(64, (1, 1), strides=1, padding='VALID', activation='linear',
               kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    x = Conv2D(48, (1, 1), strides=1, padding='VALID', activation='linear',
               kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    x = Conv2DTranspose(48, (5, 5), strides=2, padding='SAME', activation='linear',
                        kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    x = Concatenate()([x, x_RES_1, inputs])

    # =============FCL-type convolutions at each pixel...

    x = Conv2DTranspose(48, (1, 1), strides=1, padding='SAME', activation='linear',
                        kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    x = Conv2DTranspose(32, (1, 1), strides=1, padding='SAME', activation='linear',
                        kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    x = Conv2DTranspose(12, (1, 1), strides=1, padding='SAME', activation='linear',
                        kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    x = Conv2DTranspose(4, (1, 1), strides=1, padding='SAME', activation='linear',
                        kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)
    if batch_norm:
        x = BatchNormalization(axis=-1, momentum=0.99)(x)

    X0 = Conv2DTranspose(3, (3, 3), strides=1, padding='SAME', activation='linear',
                         kernel_initializer='glorot_uniform')(x)
    X0 = LeakyReLU(alpha=0.01)(X0)

    x = Concatenate()([x, X0, inputs])  # even more local info
    x = Conv2DTranspose(5, (1, 1), strides=1, padding='SAME', activation='linear',
                        kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(alpha=0.01)(x)

    x = Conv2DTranspose(num_classes, (1, 1), strides=1, padding='SAME', activation='linear',
                        kernel_initializer='glorot_uniform')(x)
    if num_classes > 1:
        outputs = Activation('softmax')(x)
    return Model(inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    model2 = build_model2(num_classes=2)
    model2.summary()
