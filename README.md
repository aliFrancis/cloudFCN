# CloudFCN

CloudFCN is a python 3 package for developing and evaluating Fully Convolutional Networks, specifically for cloud-masking. Currently, the [Biome dataset](https://landsat.usgs.gov/landsat-8-cloud-cover-assessment-validation-data) for Landsat 8 is supported.

If you find this project helpful in your work, please cite our paper: https://www.mdpi.com/2072-4292/11/19/2312

## Installation

Install from source:
```
git clone https://github.com/aliFrancis/cloudFCN
cd cloudFCN
python setup.py install
```

## Dataset

First, we download the dataset from https://landsat.usgs.gov/landsat-8-cloud-cover-assessment-validation-data
. Then we use a 'cleaning' script to convert it into an appropriate format, normalising the channels and splitting scenes up into smaller tiles. E.g. for biome:

```
python cloudFCN/data/clean_biome_data.py  path/to/download  path/for/output  [-s splitsize] [-d downsample] ...
```

The exact settings we used for our experiments were:

```
python cloudFCN/data/clean_biome_data.py  path/to/download  path/for/output  -s 398 -n True -t 0.8
```

Using these settings will create tiles of 398-by-398 with all available bands. It will also add an extra 'band' to the data, which denotes whether a pixel has nodata values in it. This is useful so that later we can give zero weight to these pixels during training and in the statistics at validation.

Currently, the data cleaning script does not sort data into training/validation sets. So, after cleaning the data, it's up to you to sort the data in the appropriate way (see the .csv files in the cloudFCN/experiments/biome folder to see how we organised it). It is recommended to use symlinks in order to not have to store multiple copies of the dataset (it can take up over 200GB).

With a cleaned dataset, we then use a fit_model.py script, an example of which can be found in cloudFCN/demo/fit_model.py, which takes a fit_config.json as input. This handles the training and validation of a model for a given experiment.


## Experiments

Once the dataset is ready, to recreate the experiments conducted in the paper, use the scripts provided in the 'experiments' folder. This will not replicate results exactly, as there is random model initialisation, however results should be relatively close. Please consult the csv files contained in the 'experiments' folder for the scenes that were placed in which dataset split.

The 'Biome' experiment can be run with, e.g.:

```
python /path/to/experiments/biome/fit_model.py /path/to/experiments/biome/RGB/fit_config_1.json
```

Please ensure that the config file is set up with the right paths to your datasets, and also that if you use a model_checkpoint_dir, it already exists.

After each epoch, a table is printed containing the percentage accuracy, commission and omission on each of the validation folders given (e.g. Barren, Forest etc.). This takes a fairly long time, so if you'd like to skip this you can change the frequency of the callback in fit_model.py, or increase the number of steps_per_epoch in the config, to do it less often.
