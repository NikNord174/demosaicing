import os
import pickle
import matplotlib.pyplot as plt

from constants import DATASET_DIRECTORY, DATASET_NAME
from convert_data import convert_data


def check_data(raw_images_path, rgb_images_path):
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


def create_dataset(dataset) -> None:
    """Create dataset and save it locally."""
    with open('.data/wiki_corpus.pickle', 'rb') as f:
        wiki_corpus = pickle.load(f)
    dataset = dataset(wiki_corpus[:2000])
    if not os.path.exists(DATASET_DIRECTORY):
        os.mkdir(DATASET_DIRECTORY)
    with open(f'.data/{DATASET_NAME}.pickle', 'wb') as f:
        pickle.dump(dataset, f)


def illustration(fig, no, image, title, fontsize=28):
    ax = fig.add_subplot(1, 2, no)
    ax.set_title(title, fontsize=fontsize)
    ax.imshow(image)
    plt.axis('off')


def imshow(image, fake):
    image_np = image.numpy()
    fake_np = fake.numpy()
    fig = plt.figure(figsize=(15, 10))
    illustration(fig, 1, image_np[0], 'Real Image')
    illustration(fig, 2, fake_np[0], 'Fake Image')
    plt.show()
