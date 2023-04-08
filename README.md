# Skin Cancer Detection Project

This project is a skin cancer detection system that uses machine learning to classify skin lesion images as benign or malignant. The project consists of several components, including data preprocessing, model development, evaluation, and testing.

## Getting Started

To get started with this project, you will need to clone the repository and install the required dependencies. You can do this by running the following commands:

```
git clone https://github.com/yourusername/skin-cancer-detection.git
cd skin-cancer-detection
pip install -r requirements.txt
```

You will also need to download the skin cancer dataset and place it in the `data` folder.

## Project Structure

The project is organized into several files and directories:

- `data_preprocessing.py`: This file contains functions for data preprocessing, such as data augmentation, resizing, and normalization.
- `model.py`: This file contains the model architecture and training process, including hyperparameter tuning and saving the best model weights.
- `evaluation.py`: This file contains functions for evaluating the model's performance, including accuracy, precision, recall, F1-score, and AUC-ROC.
- `testing.py`: This file contains functions for testing the trained model, including generating predictions, calculating evaluation metrics, and visualizing the predictions.
- `main.py`: This file contains the main function that runs the entire skin cancer image classification pipeline, including data preprocessing, model training, model evaluation, and model testing.
- `utils.py`: This file contains utility functions used throughout the project, such as loading data and saving models.
- `config.py`: This file contains configuration variables used throughout the project, such as the dataset path and model hyperparameters.
- `requirements.txt`: This file lists all the required Python packages and their versions for the project.
- `README.md`: This file provides an overview of the project and instructions for getting started.
- `LICENSE`: This file specifies the license under which the project is released.

## Running the Project

To run the project, you can simply run the `main.py` file:

```
python main.py
```

This will run the entire pipeline, including data preprocessing, model training, evaluation, and testing.

You can also customize the project settings by modifying the variables in the `config.py` file. For example, you can change the dataset path or the model hyperparameters.

## Results

The results of the project will be saved in the `results` folder. This includes the trained model weights, evaluation metrics, and visualizations.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

