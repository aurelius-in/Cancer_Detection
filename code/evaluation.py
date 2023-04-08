import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# This evaluation file contains a function called evaluate_model that takes in the trained model, 
# test data (X_test and y_test), and calculates several common evaluation metrics, including accuracy, 
# precision, recall, F1-score, and AUC-ROC. The function first generates predictions for the test set 
# using the trained model, then converts the predicted probabilities to class labels and the one-hot 
# encoded test labels to class labels. It then uses scikit-learn functions to calculate the evaluation 
# metrics and prints them to the console.

def evaluate_model(model, X_test, y_test):
    # Generate predictions for the test set
    y_pred = model.predict(X_test)

    # Convert the predicted probabilities to class labels
    y_pred = np.argmax(y_pred, axis=1)

    # Convert the one-hot encoded test labels to class labels
    y_test = np.argmax(y_test, axis=1)

    # Calculate the evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    auc_roc = roc_auc_score(y_test, y_pred, average='weighted', multi_class='ovo')

    # Print the evaluation metrics
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1-score:', f1)
    print('AUC-ROC:', auc_roc)
