import os
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt

from constants import (
    BATCH_SIZE, DATASET_DIRECTORY, CROP_HEIGHT, CROP_WIDTH,
    DATASET_NAME, RAW_IMAGES_PATH, RGB_IMAGES_PATH)
from convert_data import convert_data
from dataset import Image_Dataset


def check_data(
    raw_images_path: str = RAW_IMAGES_PATH,
    rgb_images_path: str = RGB_IMAGES_PATH
) -> None:
    """Check if data folder exists and
    if number of raw files is the same as rgb files"""

    if not os.path.isdir('new_folder'):
        raise Exception(
            'All data must be in "data" folder. Raw images must be \
            inside "data/raw_images" and rgb images must be \
            inside "data/rgb_images"')

    if len(os.listdir(raw_images_path)) == 0:
        raise Exception('Check your data in data/raw_images')

    if len(os.listdir(rgb_images_path)) == 0:
        choice = input('data/rgb_images seems empty. \
        Convert rgb images from raw files? (y/n):')
        if choice == 'y':
            convert_data()
        else:
            raise Exception('Could not find rgb data in data/rgb_images')

    if not len(os.listdir(
            raw_images_path)) == len(os.listdir(rgb_images_path)):
        raise Exception('Number of raw files is not equal to rgb files. \
        Check your data')


def create_dataset(
    file_names: list,
    dataset_name: str = DATASET_NAME,
    crop_height: int = CROP_HEIGHT,
    crop_width: int = CROP_WIDTH,
    batch_size: int = BATCH_SIZE
) -> None:
    """Create dataset and save it locally."""
    dataset = Image_Dataset(
        file_names=file_names,
        crop_height=crop_height,
        crop_width=crop_width,
        batch_size=batch_size)
    if not os.path.exists(DATASET_DIRECTORY):
        os.mkdir(DATASET_DIRECTORY)
    with open(DATASET_DIRECTORY + f'{dataset_name}.pickle', 'wb') as f:
        pickle.dump(dataset, f)


def illustration(fig, no, image, title, fontsize=28):
    ax = fig.add_subplot(1, 2, no)
    ax.set_title(title, fontsize=fontsize)
    ax.imshow(image)
    plt.axis('off')


def imshow(image: tf.Tensor, fake: tf.Tensor) -> None:
    """Draws raw and generated rgb photos"""
    image_np = image.numpy()
    fake_np = fake.numpy()
    fig = plt.figure(figsize=(15, 10))
    illustration(fig, 1, image_np[0], 'Real Image')
    illustration(fig, 2, fake_np[0], 'Fake Image')
    plt.show()
