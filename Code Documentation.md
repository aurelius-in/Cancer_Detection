# Skin Cancer Classification Project Documentation

This documentation provides an overview of the code files and functions used in the skin cancer classification project.

## Code Files

### data_preprocessing.py

This file contains functions for data preprocessing, such as data augmentation, resizing, and normalization.

### model.py

This file contains the model architecture and training process, including hyperparameter tuning and saving the best model weights.

### evaluation.py

This file contains functions for evaluating the model's performance, including accuracy, precision, recall, F1-score, and AUC-ROC.

### testing.py

This file contains functions for testing the trained model, including generating predictions, calculating evaluation metrics, and visualizing the predictions.

### main.py

This file contains the main function that runs the entire skin cancer image classification pipeline, including data preprocessing, model training, model evaluation, and model testing.

### config.py

This file contains configuration parameters for the skin cancer classification project, such as dataset paths, model parameters, and evaluation metrics.

### utils.py

This file contains helper functions for the skin cancer classification project, such as plotting training history and saving trained model weights.

## Functions

### Data Preprocessing Functions

- `load_image(filepath, target_size)`: Loads an image from a file path and resizes it to the target size.
- `augment_image(image)`: Performs data augmentation on an image, such as random flips and rotations.
- `normalize_image(image)`: Normalizes an image by scaling the pixel values between 0 and 1.

### Model Functions

- `build_model(input_shape, num_classes)`: Builds a convolutional neural network (CNN) model with the specified input shape and number of output classes.
- `train_model(model, train_generator, val_generator, num_epochs, patience, weights_dir)`: Trains a CNN model on the training data using a data generator, with early stopping based on the validation loss and the best model weights saved to disk.
- `load_best_weights(model, weights_dir)`: Loads the best performing model weights from disk into the specified Keras model.

### Evaluation Functions

- `calculate_metrics(y_true, y_pred, metrics)`: Calculates the specified evaluation metrics for the predicted and true labels.
- `evaluate_model(model, data_generator, metrics)`: Evaluates a trained model on the specified data using a data generator and returns the calculated evaluation metrics.

### Testing Functions

- `generate_predictions(model, data_generator)`: Generates predictions for the specified data using a trained model and data generator.
- `calculate_metrics(y_true, y_pred, metrics)`: Calculates the specified evaluation metrics for the predicted and true labels.
- `test_model(model, data_generator, metrics)`: Tests a trained model on the specified data using a data generator and returns the calculated evaluation metrics and predicted labels.

### Utility Functions

- `plot_history(history)`: Plots the training and validation loss and accuracy from a Keras history object.
- `save_model(model, model_dir)`: Saves a Keras model to disk.
