# User Guide

## Introduction

This user guide provides step-by-step instructions for using the skin cancer image classification model to classify images of skin lesions as benign or malignant. The model has been trained on a large dataset of skin lesion images and has achieved high accuracy in classifying these images.

## Requirements

To use the skin cancer image classification model, you will need:

- A computer running Python 3 with the necessary dependencies installed (see `requirements.txt` for the list of dependencies)
- A set of skin lesion images to classify

## Installation

To install the necessary dependencies, run the following command in the terminal:

```bash
pip install -r requirements.txt
```

## Usage
To classify skin lesion images using the skin cancer image classification model, follow these steps:

Prepare your skin lesion images in a folder on your computer. Ensure that each image is in either PNG or JPG format and has a file name that ends with either "_benign.png" or "_malignant.png", depending on whether the lesion is benign or malignant.

Open a terminal and navigate to the project directory.

Run the following command to classify the skin lesion images:

```
python main.py --data_dir /path/to/skin/lesion/images
```

Wait for the model to finish classifying the images. The predicted labels for each image will be saved in a file named predictions.csv in the project directory.

To visualize the predicted labels for a subset of the images, run the following command:

```
python testing.py --data_dir /path/to/skin/lesion/images --predictions_file predictions.csv --sample_size 10
```

This will display a grid of images with their predicted labels overlaid on top of them.


## Conclusion
This user guide provides clear instructions for installing and using the skin cancer image classification model to classify skin lesion images. With these instructions, users can easily classify their own skin lesion images and visualize the model's predictions.
