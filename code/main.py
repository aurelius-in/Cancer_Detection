import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model import build_model
from evaluation import evaluate_model
from testing import test_model

# This main file sets the data paths, image dimensions, batch size, and number of epochs 
# and early stopping criteria for the project. It then sets up the data generators using 
# the ImageDataGenerator class from TensorFlow and loads the training, validation, and test 
# data from their respective directories.

# The build_model function is called from the model.py file to create and compile the model 
# architecture using the specified number of classes.

# The ModelCheckpoint and EarlyStopping callbacks are set up to save the best model weights 
# and prevent overfitting during training.

# The model is trained using the fit method on the training data and validated on the validation 
# data. The best model weights are saved using the ModelCheckpoint callback.

# The best model weights are loaded and the model is evaluated on the test set using the 
# evaluate_model function from the evaluation.py file.

# Finally, the test_model function from the testing.py file is called to test the model on some random predictions.

# Set the data paths
train_data_dir = 'data/train'
val_data_dir = 'data/val'
test_data_dir = 'data/test'

# Set the image dimensions and batch size
img_width, img_height = 224, 224
batch_size = 32

# Set the number of epochs and early stopping criteria
num_epochs = 50
early_stopping_patience = 10

# Set up the data generators
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_width, img_height), batch_size=batch_size, class_mode='categorical')
val_generator = val_datagen.flow_from_directory(val_data_dir, target_size=(img_width, img_height), batch_size=batch_size, class_mode='categorical')
test_generator = test_datagen.flow_from_directory(test_data_dir, target_size=(img_width, img_height), batch_size=batch_size, class_mode='categorical')

# Build the model
model = build_model(num_classes=train_generator.num_classes)

# Set up the model callbacks
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, save_weights_only=True, monitor='val_loss', mode='min', verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=early_stopping_patience, mode='min', verbose=1)

# Train the model
model.fit(train_generator, epochs=num_epochs, validation_data=val_generator, callbacks=[model_checkpoint, early_stopping])

# Load the best model weights
model.load_weights('best_model.h5')

# Evaluate the model on the test set
evaluate_model(model, test_generator)

# Test the model on some random predictions
test_model(X_test, y_test, model)
