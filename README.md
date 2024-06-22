# Optic-Disc-and-Cup-Segmentation
This repository contains a deep learning-based solution for segmenting the optic disc and optic cup in fundus images using TensorFlow and Keras. The project utilizes advanced image processing techniques and incorporates a U-Net architecture for efficient segmentation. The model is trained on a dataset of fundus images and their corresponding ground truth masks.

### Note: This project is currently a work in progress and is being actively developed.

## Dataset
The dataset used in this project consists of fundus images and their corresponding ground truth masks for the optic disc and cup. The dataset is not included in this repository due to its size. You need to provide your own dataset and set the path to the dataset in the `dataset_path variable` in the `fundus_image_segmentation.py file`.

## Prerequisites
Python 3.6 or higher
TensorFlow 2.x
OpenCV
NumPy
Matplotlib

## Installation
 - Clone the repository: `git clone https://github.com/gmohor21/Optic-Disc-and-Cup-Segmentation.git`
 - Install the required packages using the `pip` command.

## Usage
 - Set the path to your local dataset in the `dataset_path` variable in the `fundus_image_segmentation.py file`.
 - Run the `fundus_image_segmentation.py` script: `python fundus_image_segmentation.py`

This script will load and preprocess the images and masks, define the U-Net model architecture, train the model, evaluate its performance, and visualize the results.

## Model Architecture
The model is based on the U-Net architecture, which consists of an encoder and a decoder part. The encoder part is responsible for capturing the context in the image, while the decoder part is responsible for the precise localization of the optic disc and cup.

The model uses convolutional blocks with batch normalization and ReLU activation functions. The decoder part uses transposed convolutions for upsampling and concatenates the features from the corresponding encoder block (skip connections).

## Loss Function and Metrics
The model is trained using a custom loss function called BCE-Jaccard loss, which combines the binary cross-entropy loss and the Jaccard loss. The Intersection over Union (IoU) metric is used to evaluate the model's performance during training.

After training, the model is evaluated using the Dice coefficient metric for both the optic disc and cup segmentation tasks.

## Results
The trained model's performance is evaluated on a test set, and the mean Dice coefficient for the optic disc and cup segmentation tasks is reported. Additionally, a visualization of the input image, ground truth masks, and predicted masks is provided.

## Contributing
Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.
