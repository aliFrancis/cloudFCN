"""
Experiment on noise tolerance
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

from cloudFCN.data import loader, transformations as trf
from cloudFCN.data.Datasets import LandsatDataset
from cloudFCN import callbacks
from cloudFCN.experiments import custom_callbacks


SIGNAL = 12.8

patch_size = 206
batch_size = 12
bands = [3, 2, 1, 0, 4, 5, 6, 7, 8, 9, 10, 11]
num_channels = len(bands)
modelpath = sys.argv[1]
model = load_model(modelpath)
parent_datadir = sys.argv[2]

datasets = [LandsatDataset(os.path.join(parent_datadir, datadir))
            for datadir in os.listdir(parent_datadir) if not datadir.startswith('.')]


dataloaders0 = [loader.dataloader(dataset, batch_size, patch_size, shuffle=False, num_classes=3, num_channels=num_channels,
                                  transformations=[trf.train_base(patch_size,fixed=True),
                                                   trf.band_select(bands),
                                                   trf.class_merge(3, 4),
                                                   trf.class_merge(1, 2),
                                                   trf.white_noise(12.8/50)
                                                   ]) for dataset in datasets]
datagens0 = [dataloader() for dataloader in dataloaders0]

dataloaders1 = [loader.dataloader(dataset, batch_size, patch_size, shuffle=False, num_classes=3, num_channels=num_channels,
                                  transformations=[trf.train_base(patch_size,fixed=True),
                                                   trf.band_select(bands),
                                                   trf.class_merge(3, 4),
                                                   trf.class_merge(1, 2),
                                                   trf.white_noise(12.8/20)
                                                   ]) for dataset in datasets]
datagens1 = [dataloader() for dataloader in dataloaders1]

dataloaders2 = [loader.dataloader(dataset, batch_size, patch_size, shuffle=False, num_classes=3, num_channels=num_channels,
                                  transformations=[trf.train_base(patch_size,fixed=True),
                                                   trf.band_select(bands),
                                                   trf.class_merge(3, 4),
                                                   trf.class_merge(1, 2),
                                                   trf.white_noise(12.8/10)
                                                   ]) for dataset in datasets]
datagens2 = [dataloader() for dataloader in dataloaders2]

dataloaders3 = [loader.dataloader(dataset, batch_size, patch_size, shuffle=False, num_classes=3, num_channels=num_channels,
                                  transformations=[trf.train_base(patch_size,fixed=True),
                                                   trf.band_select(bands),
                                                   trf.class_merge(3, 4),
                                                   trf.class_merge(1, 2),
                                                   trf.white_noise(12.8/8)
                                                   ]) for dataset in datasets]
datagens3 = [dataloader() for dataloader in dataloaders3]

dataloaders4 = [loader.dataloader(dataset, batch_size, patch_size, shuffle=False, num_classes=3, num_channels=num_channels,
                                  transformations=[trf.train_base(patch_size,fixed=True),
                                                   trf.band_select(bands),
                                                   trf.class_merge(3, 4),
                                                   trf.class_merge(1, 2),
                                                   trf.white_noise(12.8/6)
                                                   ]) for dataset in datasets]
datagens4 = [dataloader() for dataloader in dataloaders4]

dataloaders5 = [loader.dataloader(dataset, batch_size, patch_size, shuffle=False, num_classes=3, num_channels=num_channels,
                                  transformations=[trf.train_base(patch_size,fixed=True),
                                                   trf.band_select(bands),
                                                   trf.class_merge(3, 4),
                                                   trf.class_merge(1, 2),
                                                   trf.white_noise(12.8/5)
                                                   ]) for dataset in datasets]
datagens5 = [dataloader() for dataloader in dataloaders5]

callback0 = custom_callbacks.foga_table5_Callback_no_thin(datasets, datagens0)
callback1 = custom_callbacks.foga_table5_Callback_no_thin(datasets, datagens1)
callback2 = custom_callbacks.foga_table5_Callback_no_thin(datasets, datagens2)
callback3 = custom_callbacks.foga_table5_Callback_no_thin(datasets, datagens3)
callback4 = custom_callbacks.foga_table5_Callback_no_thin(datasets, datagens4)
callback5 = custom_callbacks.foga_table5_Callback_no_thin(datasets, datagens5)


print('\n\nSNR: 50')
model.fit(*next(datagens0[0]), callbacks=[callback0])
print('\n\nSNR: 20')
model.fit(*next(datagens0[0]), callbacks=[callback1])
print('\n\nSNR: 10')
model.fit(*next(datagens0[0]), callbacks=[callback2])
print('\n\nSNR: 8')
model.fit(*next(datagens0[0]), callbacks=[callback3])
print('\n\nSNR: 6')
model.fit(*next(datagens0[0]), callbacks=[callback4])
print('\n\nSNR: 5')
model.fit(*next(datagens0[0]), callbacks=[callback5])
