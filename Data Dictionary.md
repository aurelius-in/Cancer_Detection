# Data Dictionary

This data dictionary provides a description of the features in the skin cancer image dataset.

## Features

| Feature Name | Description | Type |
| --- | --- | --- |
| Image | A medical image of a skin lesion | Image file (.jpg, .png, etc.) |
| Diagnosis | The diagnosis of the skin lesion | Categorical (benign or malignant) |

## Data Integrity Checks

The following data integrity checks will be performed:

- Verify that all images are in a suitable format for processing (e.g., JPEG, PNG).
- Ensure that each image is only associated with one diagnosis.
- Verify that there are no missing values in the dataset.
- Check for any duplicate images to avoid biasing the model towards certain images.
- Ensure that the dataset is balanced between benign and malignant diagnoses.
