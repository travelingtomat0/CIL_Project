import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning import LightningModule

from collections import OrderedDict
from utils.lightning_wrapper import ModelWrapper


################################################################
# -- Start --
################################################################

class Label_Classifier(nn.Module):

    def __init__(self, in_dimension, out_size=5, batch_size=1):
        super(Label_Classifier, self).__init__()
        self.layers = nn.Sequential(
            OrderedDict([
                ('Linear', nn.Linear(in_features=in_dimension, out_features=out_size)),
                ('ReLU1', nn.Softmax()),
                #('Linear2', nn.Linear(in_features=latent_size, out_features=in_dimension)),
                #('Final Activation', nn.ReLU())
            ])
        )

    def forward(self, x):
        # x = torch.from_numpy(x).float()
        return self.layers(x.float())


def run_model(model, train_data, val_data):
    # category-wise training.
    model = Label_Classifier(in_dimension=1000, out_size=5)


###############################################
# --- MAIN ---
###############################################

model = Label_Classifier(in_dimension=1000, out_size=5)
train_data = None
val_data = None

loss = torch.nn.BCELoss()
batch_size = 20
lightning_model = ModelWrapper(model_architecture=model, learning_rate=1e-3, loss=loss, datasets=[train_data, val_data],
                               batch_size=batch_size)
trainer = pl.Trainer(max_epochs=20, deterministic=True, reload_dataloaders_every_n_epochs=5)
# Train the model.
trainer.fit(lightning_model)

print("Finished")

