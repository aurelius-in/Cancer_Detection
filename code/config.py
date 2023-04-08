import os

# This config.py file defines several configuration parameters for the skin cancer classification project:
# DATA_DIR: The directory path to the skin cancer dataset.
# TRAIN_DIR, VAL_DIR, TEST_DIR: The directory paths to the train, validation, and test data subsets within the dataset.
# IMG_SIZE: The size of the input images to the model (in pixels).
# BATCH_SIZE: The number of images to include in each training batch.
# LEARNING_RATE: The learning rate for the optimizer during model training.
# NUM_EPOCHS: The number of epochs to train the model for.
# PATIENCE: The number of epochs to wait before early stopping if the validation loss doesn't improve.
# WEIGHTS_DIR: The directory path to save the best performing model weights.
# METRICS: The evaluation metrics to calculate during model evaluation, including accuracy, precision, recall, F1-score, and AUC-ROC.

# Dataset paths
DATA_DIR = os.path.join('data', 'skin_cancer')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Model parameters
IMG_SIZE = 224
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
PATIENCE = 10
WEIGHTS_DIR = 'weights'

# Evaluation metrics
METRICS = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
