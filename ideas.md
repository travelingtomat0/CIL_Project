# Collection of Ideas

## Task

- Given ratings (between 1 and 5) of 10.000 users for 1.000 different items, predict user preferences (given by rating between 1 and 5).

## Preprocessing

- Data Augmentation to help with generalisation
- Centering per item or user

## Own Baselines

- PCA-Classifier
- SVM
- Autoencoder

## Affinity instead of Distance

- See https://doi.org/10.1016/j.cell.2021.04.048, Methods Section, "Weighted Nearest Neighbor Analysis"
  - Quick Access: https://www.sciencedirect.com/science/article/pii/S0092867421005833?via%3Dihub#sec4.4.9
  - Instead of using a simple distance metric such as Euclidean or Cosine, use Affinity using eg. Exponential Kernel


## General
The idea behind the recommender system is the imputation of user preferences in the preference matrix. In a classical
bioinformatics setting, this can be done with variants of a hidden markov model (HMM), Autoencoders etc. (see
ML for Genomics).

However, the utility of simply imputing data is to be questioned here.

Also question to be answered: is this a regression task?