import numpy as np
import os
from scipy import misc
import time
import random

def train_base(patch_size,fixed = False):
    """
    Makes transformation for image/mask pair that is a randomly cropped, rotated
    and flipped portion of the original.

    Parameters
    ----------
    patch_size : int
        Spatial dimension of output image/mask pair (assumes Width==Height).
    fixed : bool, optional
        If True, always take patch from top-left of scene, with no rotation or
        flipping. This is useful for validation and reproducability.

    Returns
    -------
    apply_transform : func
        Transformation for image/mask pairs.
    """
    def apply_transform(img, mask):
        if fixed:
            left = 0
            top = 0
            crop_size = int(
                min(patch_size, img.shape[0] - 1, img.shape[1] - 1))
            img = img[top:top + crop_size, left:left + crop_size, ...]
            mask = mask[top:top + crop_size, left:left + crop_size, ...]

        else:
            if not patch_size == img.shape[0]:
                crop_size = int(
                    min(patch_size, img.shape[0] - 1, img.shape[1] - 1))

                left = int(
                    random.randint(
                        0,
                        img.shape[1] -
                        crop_size))
                top = int(
                    random.randint(
                        0,
                        img.shape[0] -
                        crop_size))

            img = img[top:top + crop_size, left:left + crop_size, ...]
            mask = mask[top:top + crop_size, left:left + crop_size, ...]

        rota = random.choice([0, 1, 2, 3])
        flip = random.choice([True, False])
        if rota and not fixed:
            img = np.rot90(img, k=rota)
            mask = np.rot90(mask, k=rota)
        if flip and not fixed:
            img = np.fliplr(img)
            mask = np.fliplr(mask)
        return img, mask
    return apply_transform


def band_select(bands):
    """
    Return image/mask pair where spectral bands in image have been selected as a list.

    Parameters
    ----------
    bands : list
        Spectral bands, defined with respect to input's final dimension.

    Returns
    -------
    apply_transform : func
        Transformation for image/mask pairs.
    """
    def apply_transform(img, mask):
        if bands is not None:
            img = img[..., bands]
        return img, mask
    return apply_transform


def class_merge(class1, class2):
    """
    Create image/mask pairs where classes in mask have been merged (reduces final mask
    dimension by 1).

    Parameters
    ----------
    class1 : int
        Index of class to be fused.
    class2 : int
        Index of class to be fused.

    Returns
    -------
    apply_transform : func
        Transformation for image/mask pairs.
    """

    def apply_transform(img, mask):
        mask[..., class1] += mask[..., class2]
        mask = mask[..., np.arange(mask.shape[-1]) != class2]
        return img, mask
    return apply_transform

def sometimes(p, transform):
    """
    Wrapper function which randomly applies the transform with probability p.

    Parameters
    ----------
    p : float
        Probability of transform being applied
    transform : func
        Function which transforms image/mask pairs.

    Returns
    -------
    apply_transform : func
        Transformation for image/mask pairs.
    """

    def apply_transform(img, mask):
        random_apply = random.random() < p
        if random_apply:
            return transform(img, mask)
        else:
            return img, mask
    return apply_transform


def chromatic_shift(shift_min=-0.10, shift_max=0.10):
    """
    Adds a different random amount to each spectral band in image.

    Parameters
    ----------
    shift_min : float, optional
        Lower bound for random shift.
    shift_max : float, optional
        Upper bound for random shift.

    Returns
    -------
    apply_transform : func
        Transformation for image/mask pairs.
    """

    def apply_transform(img, mask):
        img = img + np.random.uniform(low=shift_min,
                                      high=shift_max, size=[1, 1, img.shape[-1]]).astype(np.float32)
        return img, mask
    return apply_transform


def chromatic_scale(factor_min=0.90, factor_max=1.10):
    """
    Multiplies each spectral band in batch by a different random factor.

    Parameters
    ----------
    factor_min : float, optional
        Lower bound for random factor.
    factor_max : float, optional
        Upper bound for random factor.

    Returns
    -------
    apply_transform : func
        Transformation for image/mask pairs.
    """
    def apply_transform(img, mask):
        img = img * np.random.uniform(low=factor_min,
                                      high=factor_max, size=[1, 1, img.shape[-1]]).astype(np.float32)
        return img, mask
    return apply_transform


def intensity_shift(shift_min=-0.10, shift_max=0.10):
    """
    Adds single random amount to all spectral bands.

    Parameters
    ----------
    shift_min : float, optional
        Lower bound for random shift.
    shift_max : float, optional
        Upper bound for random shift.

    Returns
    -------
    apply_transform : func
        Transformation for image/mask pairs.
    """
    def apply_transform(img, mask):
        img = img + (shift_max-shift_min)*random.random()+shift_min
        return img, mask
    return apply_transform


def intensity_scale(factor_min=0.95, factor_max=1.05):
    """
    Multiplies all spectral bands by a single random factor.

    Parameters
    ----------
    factor_min : float, optional
        Lower bound for random factor.
    factor_max : float, optional
        Upper bound for random factor.

    Returns
    -------
    apply_transform : func
        Transformation for image/mask pairs.
    """
    def apply_transform(img, mask):
        img = img * random.uniform(factor_min, factor_max)
        return img, mask
    return apply_transform


def white_noise(sigma=0.1):
    """
    Adds white noise to image.

    Parameters
    ----------
    sigma : float, optional
        Standard deviation of white noise

    Returns
    -------
    apply_transform : func
        Transformation for image/mask pairs.
    """

    def apply_transform(img, mask):
        noise = (np.random.randn(*img.shape) * sigma).astype(np.float32)
        return img + noise, mask
    return apply_transform


def bandwise_salt_and_pepper(salt_rate, pepp_rate, pepp_value=0, salt_value=255):
    """
    Adds salt and pepper (light and dark) noise to image,  treating each band independently.

    Parameters
    ----------
    salt_rate : float
        Percentage of pixels that are set to salt_value.
    pepp_rate : float
        Percentage of pixels that are set to pepp_value.
    pepp_value : float, optional
        Value that pepper pixels are set to.
    salt_value : float, optional
        Value that salt pixels are set to.

    Returns
    -------
    apply_transform : func
        Transformation for image/mask pairs.
    """
    def apply_transform(img, mask):
        salt_mask = np.random.choice([False, True], size=img.shape, p=[
                                     1 - salt_rate, salt_rate])
        pepp_mask = np.random.choice([False, True], size=img.shape, p=[
                                     1 - pepp_rate, pepp_rate])

        img[salt_mask] = salt_value
        img[pepp_mask] = pepp_value

        return img, mask
    return apply_transform


def salt_and_pepper(salt_rate, pepp_rate, pepp_value=0, salt_value=255):
    """
    Adds salt and pepper (light and dark) noise to image, to all bands in a pixel.

    Parameters
    ----------
    salt_rate : float
        Percentage of pixels that are set to salt_value.
    pepp_rate : float
        Percentage of pixels that are set to pepp_value.
    pepp_value : float, optional
        Value that pepper pixels are set to.
    salt_value : float, optional
        Value that salt pixels are set to.

    Returns
    -------
    apply_transform : func
        Transformation for image/mask pairs.
    """
    def apply_transform(img, mask):
        salt_mask = np.random.choice(
            [False, True], size=img.shape[:-1], p=[1 - salt_rate, salt_rate])
        pepp_mask = np.random.choice(
            [False, True], size=img.shape[:-1], p=[1 - pepp_rate, pepp_rate])

        img[salt_mask] = [salt_value for i in range(img.shape[-1])]
        img[pepp_mask] = [pepp_value for i in range(img.shape[-1])]

        return img, mask
    return apply_transform


def quantize(number_steps, min_value=0, max_value=255, clip=False):
    """
    Quantizes an image based on a given number of steps by rounding values to closest
    value.

    Parameters
    ----------
    number_steps : int
        Number of values to round to
    min_value : float
        Lower bound of quantization
    max_value : float
        Upper bound of quantization
    clip : bool
        True if values outside of [min_value:max_value] are clipped. False otherwise.

    Returns
    -------
    apply_transform : func
        Transformation for image/mask pairs.
    """
    stepsize = (max_value-min_value)/number_steps

    def apply_transform(img, mask):
        img = (img//stepsize)*stepsize
        if clip:
            img = np.clip(img, min_value, max_value)
        return img, mask
    return apply_transform
