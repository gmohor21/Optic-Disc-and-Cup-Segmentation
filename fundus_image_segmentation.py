import os
import cv2 
import numpy as np 
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers 
from tensorflow.keras import backend as K 
#import tensorflow_addons as tfa 

# Define the path to the local dataset
dataset_path = 'dataset path'

# Load and preprocess images and masks
images = []
masks = []
for f in os.listdir(dataset_path):
    if f.endswith('_g.jpg'):
        # Load fundus images
        img_path = os.path.join(dataset_path, f)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        images.append(img)

        # Load corresponding ground truth masks
        mask_path = os.path.join(dataset_path, f.replace('_g.jpg', '_h.jpg'))
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.expand_dims(mask, axis=-1)
        masks.append(mask)

images = np.array(images)
masks = np.array(masks)

# Define data augmentation parameters
data_gen_args = dict(rotation_range=10,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     shear_range=0.1,
                     zoom_range=0.1,
                     horizontal_flip=True,
                     fill_mode='reflect')

# Create an image data generator
image_datagen = keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
mask_datagen = keras.preprocessing.image.ImageDataGenerator(**data_gen_args)

# Fit the image and mask data generators to the data
image_datagen.fit(images, augment=True, seed=42)
mask_datagen.fit(masks, augment=True, seed=42)

# Combine the image and mask generators
train_generator = zip(image_datagen.flow(images, batch_size=1, seed=42),
                       mask_datagen.flow(masks, batch_size=1, seed=42))

# Define BCE-Jaccard loss function
def bce_jaccard_loss(y_true, y_pred):
    """
    Calculate the BCE-Jaccard loss function.

    Args:
        y_true: Ground truth masks.
        y_pred: Predicted masks.

    Returns:
        The BCE-Jaccard loss value.
    """
    # Smooth value to avoid division by zero
    smooth = 1e-6

    # Flatten the ground truth and predicted masks
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    # Calculate the intersection and union of the masks
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection

    # Calculate the Jaccard score
    jaccard = (intersection + smooth) / (union + smooth)

    # Calculate the binary cross-entropy loss
    bce = K.binary_crossentropy(y_true, y_pred)

    # Calculate the BCE-Jaccard loss
    return bce - jaccard

# Define IoU metric function
def iou_metric(y_true, y_pred):
    """
    Calculate the Intersection over Union (IoU) metric.

    Args:
        y_true: Ground truth masks.
        y_pred: Predicted masks.

    Returns:
        The IoU score.
    """
    # Smooth value to avoid division by zero
    smooth = 1e-6

    # Flatten the ground truth and predicted masks
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    # Calculate the intersection and union of the masks
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection

    # Calculate the IoU score
    return (intersection + smooth) / (union + smooth)

# Build U-Net model
def conv_block(inputs, num_filters):
    """
    Convolutional block of the U-Net model.

    Args:
        inputs: Input tensor.
        num_filters: Number of filters in the convolutional layers.

    Returns:
        Output tensor after applying the convolutional block.
    """
    # First convolutional layer
    x = layers.Conv2D(num_filters, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)  # Normalize the activations
    x = layers.Activation('relu')(x)  # Apply ReLU activation

    # Second convolutional layer
    x = layers.Conv2D(num_filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    return x

def decoder_block(inputs, skip_features, num_filters):
    """
    Decoder block of the U-Net model.

    Args:
        inputs: Input tensor.
        skip_features: Features from the corresponding encoder block.
        num_filters: Number of filters in the convolutional layers.

    Returns:
        Output tensor after applying the decoder block.
    """
    # Upsample the input tensor
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(inputs)

    # Concatenate the upsampled tensor with the skip features
    x = layers.Concatenate()([x, skip_features])

    # Apply convolutional block to the concatenated tensor
    x = conv_block(x, num_filters)

    return x

inputs = keras.Input(shape=(None, None, 3))
skip_connections = []
x = inputs
for num_filters in [16, 32, 64, 128]:
    x = conv_block(x, num_filters)
    skip_connections.append(x)
    x = layers.MaxPooling2D((2, 2))(x)

x = conv_block(x, 256)

for num_filters, skip in zip([128, 64, 32, 16], reversed(skip_connections)):
    x = decoder_block(x, skip, num_filters)

outputs = layers.Conv2D(2, 3, padding='same', activation='softmax')(x)

model = keras.Model(inputs, outputs)

model.summary()

# Define optimizer and compile model
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=bce_jaccard_loss, metrics=[iou_metric])

# Train the model
epochs = 10
batch_size = 1
steps_per_epoch = len(images)
model.fit(list(zip(*train_generator)), epochs=epochs, steps_per_epoch=steps_per_epoch)

# Evaluate the model
def dice_coeff(y_true, y_pred, smooth=1):
    """
    Calculate the Dice coefficient metric.

    Args:
        y_true (ndarray): Ground truth masks.
        y_pred (ndarray): Predicted masks.
        smooth (float, optional): Smoothing value to avoid division by zero. Defaults to 1.

    Returns:
        ndarray: The Dice coefficient score.
    """
    # Calculate the intersection of ground truth and predicted masks
    intersection = np.sum(y_true * y_pred, axis=(1, 2, 3))

    # Calculate the union of ground truth and predicted masks
    union = np.sum(y_true, axis=(1, 2, 3)) + np.sum(y_pred, axis=(1, 2, 3))

    # Calculate the Dice coefficient score
    dice = np.mean((2.0 * intersection + smooth) / (union + smooth), axis=0)

    return dice

def eval_model(model, images, masks):
    """
    Evaluate the model on a set of images and masks.

    Args:
        model (keras.Model): The trained model.
        images (ndarray): Array of images.
        masks (ndarray): Array of masks.

    Returns:
        None
    """
    dice_disc = []
    dice_cup = []
    for img, mask in zip(images, masks):
        # Predict the mask for the image
        img = np.expand_dims(img, axis=0)
        pred_mask = model.predict(img)[0]

        # Extract the predicted and ground truth masks for the optic disc and cup
        disc_pred = pred_mask[:, :, 0]
        cup_pred = pred_mask[:, :, 1]
        disc_true = mask[:, :, 0]
        cup_true = mask[:, :, 1]

        # Calculate the Dice coefficient for the optic disc and cup
        dice_disc.append(dice_coeff(disc_true, disc_pred)[0])
        dice_cup.append(dice_coeff(cup_true, cup_pred)[0])

    # Print the mean Dice coefficient for the optic disc and cup
    print(f"Mean Dice Coefficient (Optic Disc): {np.mean(dice_disc):.4f}")
    print(f"Mean Dice Coefficient (Optic Cup): {np.mean(dice_cup):.4f}")

eval_model(model, images, masks)

# Visualize results
test_img = images[0]
test_mask = masks[0]
pred_mask = model.predict(np.expand_dims(test_img, axis=0))[0]

import matplotlib.pyplot as plt 

fig, ax = plt.subplots(1, 5, figsize=(20, 5))
ax[0].imshow(test_img)
ax[0].set_title('Input Image')
ax[1].imshow(test_mask[:, :, 0], cmap='gray')
ax[1].set_title('Ground Truth Disc')
ax[2].imshow(test_mask[:, :, 1], cmap='gray')
ax[2].set_title('Ground Truth Cup')
ax[3].imshow(pred_mask[:, :, 0], cmap='gray', vmin=0, vmax=1)
ax[3].set_title('Predicted Disc')
ax[4].imshow(pred_mask[:, :, 1], cmap='gray', vmin=0, vmax=1)
ax[4].set_title('Predicted Cup')
plt.show()
