import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# This data_preprocessing.py file defines the preprocess_data function, 
# which applies data augmentation, resizing, and normalization to the image data.

# The ImageDataGenerator class from TensorFlow is used to apply data augmentation, 
# which includes rotation, width and height shifts, shear, zoom, and horizontal flip. 
# The rescale parameter is also set to normalize the pixel values to be between 0 and 1.

# The function returns the generators for the training and validation data, which can be 
# used in the main.py file for training the model. The generators use the flow_from_directory 
# method to load the images from the specified directories and resize them to the target size. 
# The class_mode is set to 'categorical' since this is a multi-class classification problem.

def preprocess_data(data_dir, target_size, batch_size):
    """
    Preprocesses the image data by applying data augmentation, resizing, and normalization.

    Args:
        data_dir (str): The directory path containing the image data.
        target_size (tuple): The target size to resize the images to.
        batch_size (int): The batch size to use for training.

    Returns:
        train_generator (DirectoryIterator): The generator for training data.
        val_generator (DirectoryIterator): The generator for validation data.
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        os.path.join(data_dir, 'val'),
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, val_generator
