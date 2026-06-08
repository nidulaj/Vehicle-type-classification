import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input



# Augmentations

def add_gaussian_noise(img):
    noise = np.random.normal(0, 0.02, img.shape)
    img = img + noise
    return np.clip(img, 0.0, 1.0)


def random_cutout(img, max_mask_size=50):
    img = img.copy()
    h, w, _ = img.shape
    mask_size = np.random.randint(20, max_mask_size)
    x = np.random.randint(0, w)
    y = np.random.randint(0, h)

    x1 = max(0, x - mask_size // 2)
    y1 = max(0, y - mask_size // 2)
    x2 = min(w, x + mask_size // 2)
    y2 = min(h, y + mask_size // 2)

    img[y1:y2, x1:x2, :] = 0
    return img


def combined_augmentation(img):
    img = add_gaussian_noise(img)
    img = random_cutout(img)
    return img



# Conditional resize + padding

def resize_if_smaller(img, target_size=(224, 224)):
    h, w, _ = img.shape

    if h < target_size[0] or w < target_size[1]:
        img = tf.image.resize_with_pad(
            img,
            target_size[0],
            target_size[1]
        )
        img = img.numpy()

    return img



# Full preprocessing pipeline

def full_preprocessing(img):
    img = resize_if_smaller(img, (224, 224))
    img = preprocess_input(img)
    return img



# Data generators

def create_generators(config):

    train_datagen = ImageDataGenerator(
        preprocessing_function=full_preprocessing,
        validation_split=config["validation_split"],
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=full_preprocessing,
        validation_split=config["validation_split"]
    )

    train_gen = train_datagen.flow_from_directory(
        directory=config["dataset_dir"],
        target_size=(config["image_height"], config["image_width"]),
        batch_size=config["batch_size"],
        class_mode="categorical",
        subset="training",
        shuffle=True
    )

    val_gen = val_datagen.flow_from_directory(
        directory=config["dataset_dir"],
        target_size=(config["image_height"], config["image_width"]),
        batch_size=config["batch_size"],
        class_mode="categorical",
        subset="validation",
        shuffle=False
    )

    return train_gen, val_gen