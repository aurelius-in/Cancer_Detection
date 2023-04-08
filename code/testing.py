import numpy as np
import matplotlib.pyplot as plt

# This testing file contains two functions, visualize_predictions and test_model.
# The visualize_predictions function generates predictions for the test set, converts 
# the predicted probabilities to class labels, and converts the one-hot encoded test 
# labels to class labels. It then randomly selects and visualizes some of the predictions 
# with their true and predicted labels.

The test_model function also generates predictions for the test set, converts the predicted probabilities to class labels, and converts the one-hot encoded test labels to class labels. It then calculates the accuracy of the model on the test set and prints it to the console. Finally, it calls the visualize_predictions function to visualize some of the predictions.

def visualize_predictions(X_test, y_test, model):
    # Generate predictions for the test set
    y_pred = model.predict(X_test)

    # Convert the predicted probabilities to class labels
    y_pred = np.argmax(y_pred, axis=1)

    # Convert the one-hot encoded test labels to class labels
    y_test = np.argmax(y_test, axis=1)

    # Visualize some random predictions
    fig, axs = plt.subplots(3, 3, figsize=(10,10))
    axs = axs.flatten()
    for i in range(9):
        index = np.random.randint(0, len(X_test))
        img = X_test[index]
        true_label = y_test[index]
        pred_label = y_pred[index]
        axs[i].imshow(img)
        axs[i].set_title(f'True: {true_label}, Pred: {pred_label}')
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()

def test_model(X_test, y_test, model):
    # Generate predictions for the test set
    y_pred = model.predict(X_test)

    # Convert the predicted probabilities to class labels
    y_pred = np.argmax(y_pred, axis=1)

    # Convert the one-hot encoded test labels to class labels
    y_test = np.argmax(y_test, axis=1)

    # Calculate the accuracy of the model on the test set
    accuracy = np.mean(y_pred == y_test)

    # Print the accuracy to the console
    print('Test accuracy:', accuracy)

    # Visualize some random predictions
    visualize_predictions(X_test, y_test, model)
