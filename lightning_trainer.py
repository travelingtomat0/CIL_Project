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
from utils.dataset import InputDataset, ValDataset
from models.Autoencoder import *
import numpy as np

import argparse
from math import ceil
import csv
from utils.data_utils import *


# ------------------------------------------------
# Function Definitions
# ------------------------------------------------

def mask(data, non_nan_users, non_nan_items):
    masking_ids = np.random.choice(
        np.array([i for i in range(0, len(non_nan_users))]), size=int(0.1 * len(non_nan_users))
    ).astype(int)

    user_masks = [non_nan_users[i] for i in masking_ids]
    item_masks = [non_nan_items[i] for i in masking_ids]

    mask = np.ones(mat.shape)
    for i in range(len(user_masks)):
        mask[item_masks[i], user_masks[i]] = 0

    # mask = np.random.rand(val_data.X.shape[0], val_data.X.shape[1]) < 0.5
    # mask = np.random.rand(val_data.X.shape[0], val_data.X.shape[1]) < 0.5
    data.masked_X = mask * val_data.X
    data.mask = mask



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

if not os.path.exists(f'data_train.idx'):
    mat, non_nan_users, non_nan_items = read_data(path=f'data_train.csv')
    with open(f'data_train.idx', 'wb') as f:
        pickle.dump(mat, f)
    with open(f'non_nan_users.idx', 'wb') as f:
        pickle.dump(non_nan_users, f)
    with open(f'non_nan_items.idx', 'wb') as f:
        pickle.dump(non_nan_items, f)
else:
    print(f"Loading old matrix from data_train.idx")
    mat = np.load(f'data_train.idx', allow_pickle=True)
    non_nan_users = np.load(f'non_nan_users.idx', allow_pickle=True)
    non_nan_items = np.load(f'non_nan_items.idx', allow_pickle=True)

mat = np.nan_to_num(x=mat)

# Ensure Reproducability:
pl.seed_everything(2022, workers=True)

# Initialise model data. Set new_model=True to see how we train on the generalisation set
#data, val_data = torch.utils.data.random_split(InputDataset(mat), [6000, 4000])
# train_data = InputDataset(mat[:, 1000:])
# val_data = InputDataset(mat[:, :1000])

train_data = InputDataset(mat)
val_data = InputDataset(mat)





masking_ids = np.random.choice(
    np.array([i for i in range(0, len(non_nan_users))]), size=int(0.1*len(non_nan_users))
).astype(int)

user_masks = [non_nan_users[i] for i in masking_ids]
item_masks = [non_nan_items[i] for i in masking_ids]

mask = np.ones(mat.shape)
for i in range(len(user_masks)):
    mask[item_masks[i], user_masks[i]] = 0

#mask = np.random.rand(val_data.X.shape[0], val_data.X.shape[1]) < 0.5
#mask = np.random.rand(val_data.X.shape[0], val_data.X.shape[1]) < 0.5
val_data.masked_X = mask * val_data.X
val_data.mask = mask

print("Finished Preprocessing the Input Dataset")

model = Simple_Autoencoder(in_dimension=1000, latent_size=15)

# Testing the Dataset
print(f"Dataset returns matrix of shape {train_data[0][0].shape}")

# Initialize training-loss
loss = torch.nn.MSELoss()

batch_size = 20

# Initialise pl model and trainer
lightning_model = ModelWrapper(model_architecture=model, learning_rate=1e-3, loss=loss, datasets=[train_data, val_data],
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
