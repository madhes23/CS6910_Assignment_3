# CS6910 Deep Learning Assignment 3

## Required Libraries
1. `torch`
2. `random`
3. `tqdm` (for progress bar visualization)
4. `numpy`
5. `matplotlib`
6. `pandas`
7. `sklearn.utils.shuffle` (for shuffling dataset)
8. `seaborn` (for attention heatmap visualization)
# File Structre
A breif description of the list of files, and their respective purposes.
1. `AksharantarDataset.py` - A class for handling data and all the data related methods
2. `Utils.py` - A genaral set of versatile utility functions that is used in this assignment.
3. `SequenceLearning.py` - This is the main file, containing the Encoder, Decoder, AttentionDecoder, and Seq2Seq model.
4. `BestModels.ipynb` - This file takes care of perfoming the WandB Sweeps for hyper-paramter tuning, and training, testing the best models found in the WandB sweeps.
5. `train.py` - Code to implement the command line interface to interact with the repository

# Classes
## Encoder
## Decoder
## AttentionDecoder
## Seq2Seq

# Training
## Early Stopping
Here Early Stopping is done to prevent extremely inefficient runs from happening.
## TFR method

# Usage Guidelines
## AksharantarDataset
Lang, tokens and path selection
## Model creation
### create models using train.py
### create models directly