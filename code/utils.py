import os
import matplotlib.pyplot as plt

# This utils.py file defines two helper functions that can be used in the main.py file.
# The plot_history function takes a Keras history object as input and plots the training 
# and validation loss and accuracy over the epochs. This can be useful for visualizing the 
# training progress and identifying overfitting.

# The save_model function takes a trained Keras model and a directory path as input and saves 
# the model to disk in the specified directory. This can be useful for saving the best performing 
# model weights for later use or deployment.

def plot_history(history):
    """
    Plots the training and validation loss and accuracy from a Keras history object.

    Args:
        history (History): The history object returned from model.fit().
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot training and validation accuracy
    ax1.plot(history.history['accuracy'], label='train')
    ax1.plot(history.history['val_accuracy'], label='val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    # Plot training and validation loss
    ax2.plot(history.history['loss'], label='train')
    ax2.plot(history.history['val_loss'], label='val')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.show()

def save_model(model, model_dir):
    """
    Saves a Keras model to disk.

    Args:
        model (Model): The trained Keras model.
        model_dir (str): The directory path to save the model to.
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model.save(os.path.join(model_dir, 'model.h5'))
