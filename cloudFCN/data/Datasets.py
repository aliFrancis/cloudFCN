import numpy as np
import os
from random import shuffle


class Dataset():
    """
    A class used to load image/mask pairs from a set of directories

    Attributes
    ----------
    dirs : str, list
        Paths for each parent directory
    paths : list
        Paths to every subdirectory within self.dirs that contain valid image/maks pairs.
    """

    def __init__(self, dirs, orderShuffle=True):
        self.dirs = dirs
        self.paths = self.parse_dirs()  # returns full paths for annotation folders
        if orderShuffle:
            shuffle(self.paths)
        print('LandsatDataset with {} tiles'.format(len(self.paths)))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        # print(self.paths[index].split(os.sep)[-2:])

        im = np.load(os.path.join(
            self.paths[index], 'image.npy')).astype('float')
        mask = np.load(os.path.join(self.paths[index], 'mask.npy'))

        try:
            return im, mask
        except BaseException:
            print('Could not read:', self.paths[index])

    def parse_dirs(self):
        """
        Look for all valid subdirectories in self.dirs

        Returns
        -------
        valid_subdirs : list
            Paths to all subdirectories containing valid image/mask pair
        """
        valid_subdirs = []
        if isinstance(self.dirs, list):
            for dir in self.dirs:
                for root, dirs, paths in os.walk(dir):
                    valid_subdirs += [os.path.join(root, dir) for dir in dirs
                                      if os.path.isfile(os.path.join(root, dir, 'image.npy'))
                                      and os.path.isfile(os.path.join(root, dir, 'mask.npy'))]

        else:
            for root, dirs, paths in os.walk(self.dirs):
                valid_subdirs += [os.path.join(root, dir) for dir in dirs
                                  if os.path.isfile(os.path.join(root, dir, 'image.npy'))
                                  and os.path.isfile(os.path.join(root, dir, 'mask.npy'))]
        return valid_subdirs

    def randomly_reduce(self, factor):
        """
        Updates self.paths to random sample of original, with count len(self)*factor
        """
        new_length = int(len(self)*factor)
        self.paths = np.random.choice(self.paths, new_length, replace=False)
        return


class LandsatDataset(Dataset):
    """
    Extension of Dataset class specifically for Landsat data. Includes channel-wise
    averaging and display functions.
    """

    def channel_means(self, samplesize=1.):
        """
        Calculate mean value of each band in dataset's images

        Parameters
        ----------
        samplesize : float, optional
            Factor of sampled pixels, set between 0->1.

        Returns
        -------
        mean : list
            Mean average values in each spectral band
        """
        total_pix = 0
        mean = 0
        frequency = int(1/samplesize)
        for i in range(0, len(self), frequency):
            im, _ = self[i]
            mean_i = im.mean(axis=(0, 1))
            pix_i = im.shape[0] * im.shape[1]
            mean = np.divide(mean * total_pix + mean_i *
                             pix_i, total_pix + pix_i)
            total_pix += pix_i

        return mean

    def display(self, im, mask, rgb_selection=[3, 2, 1]):
        """
        Create viewable RGB image/mask pair from original

        Parameters
        ----------
        im : array
            Original multispectral image
        mask : array
            Original one-hot encoded mask
        rgb_selection : list, optional
            Specify which bands correspond to Red, Green, Blue  for output


        Returns
        -------
        im : array
            N-by-M-by-3 array with renormalised bands
        mask : array
            N-by-M array with different values for each class
        """
        assert len(rgb_selection) == 3, 'rgb_selection must have length 3'
        im = (im[..., rgb_selection]+1.2)/4  # 4 standard deviations from 0->1
        im = np.clip(im, 0, 1)
        mask = np.argmax(mask, axis=-1)
        return im, mask


if __name__ == '__main__':
    import sys
    dataset_path = sys.argv[1]
    dataset = LandsatDataset(dataset_path)
    dataset.randomly_reduce(0.1)
    mn = dataset.channel_means()
    print(mn)
