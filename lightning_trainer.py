# ------------
# Author:       Philip Toma
# Description:  This file implements the training pipeline of the IFBID Model.
# Usage:        python3 trainer.py [--debug] [--epochs 10] [path]
#               where the []-brackets mean an entry is optional, but should be used.
#               For more info:      python3 trainer.py --help
# ------------

import pytorch_lightning as pl
import torch
import os
import pickle

from utils.lightning_wrapper import *
from utils.dataset import InputDataset
from models.Autoencoder import *

import argparse
from math import ceil
import csv
from utils.data_utils import *


# ------------------------------------------------
# Global Parameter Initialisation
# ------------------------------------------------

parser = argparse.ArgumentParser(description='Run Prediction Script.')
parser.add_argument('path', type=str, default='/home/phil/Documents/Studium/Courses/CIL/Project',
                    help='path to training data.')
parser.add_argument('--debug', action='store_true',
                    help='set --debug flag for lower computational impact.')
args = parser.parse_args()
path = args.path

if not os.path.exists(f'{path}/data_train.idx'):
    mat = read_data(path=f'{path}/data_train.csv')
    with open(f'{path}/data_train.idx', 'wb') as f:
        pickle.dump(mat, f)
else:
    print(f"Loading old matrix from {path}/data_train.idx")
    mat = np.load(f'{path}/data_train.idx', allow_pickle=True)

mat = np.nan_to_num(x=mat)

# Ensure Reproducability:
pl.seed_everything(2022, workers=True)

# Initialise model data. Set new_model=True to see how we train on the generalisation set
data, val_data = torch.utils.data.random_split(InputDataset(mat), [6000, 4000])

model = Simple_Autoencoder(in_dimension=1000, latent_size=50)

# Testing the Dataset
print(f"Dataset returns matrix of shape {data[0][0].shape}")

# Initialize training-loss
loss = torch.nn.MSELoss()

batch_size = 1

# Initialise test_data.  Set new_model=True to see how we train on the generalisation set.
# test_data = InputDataset()

# Initialise pl model and trainer
lightning_model = ModelWrapper(model_architecture=model, learning_rate=1e-3, loss=loss, datasets=[data, val_data],
                               batch_size=batch_size)

trainer = pl.Trainer(max_epochs=20, deterministic=True, reload_dataloaders_every_n_epochs=5)

# Train the model.
trainer.fit(lightning_model)

# Test the model.
# trainer.test(lightning_model)

# get and record accuracy obtained from test.
# test_accuracy = lightning_model._model.test_accuracy

"""print(f'Save model under name: {name}{tmp}-trainsize-{int(0.7*len(data))+int(0.7*len(test_data))}')

with open(os.path.join(args.path, f'{name}{tmp}-trainsize-{int(0.7*len(data))+int(0.7*len(test_data))}.csv'), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    writer.writerow([test_accuracy])

if not args.debug:
    os.makedirs(os.path.join(args.path, 'bias_classifiers'), exist_ok=True)
    torch.save(lightning_model._model.state_dict(),
               os.path.join(args.path, 'bias_classifiers',
                            f'{name}{tmp}-trainsize-{int(0.7*len(data))+int(0.7*len(test_data))}')
               )"""
