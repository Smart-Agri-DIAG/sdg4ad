"""
This script is used to generate the train/val splits for the dataset.
"""

import os
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold


def get_image_list(data_dir, class_label, extension='.jpg'):
    """
    Retrieves a list of file paths for images from a specified directory with a given class label and file extension.

    This function iterates over all files in a specified subdirectory of the data directory. The subdirectory
    is determined by the class label. It then filters these files based on the file extension and returns a list
    of the file paths for these files.

    Args:
        data_dir (str): The path to the data directory.
        class_label (str): The class label of the images.
        extension (str, optional): The extension of the images. Defaults to '.jpg'.

    Returns:
        list: A list of file paths that match the given class label and file extension.
    """
    try:
        return [os.path.join(data_dir, class_label, file) for file in os.listdir(os.path.join(data_dir, class_label)) if file.endswith(extension)]
    except FileNotFoundError:
        print(f"The directory {os.path.join(data_dir, class_label)} does not exist.")
        return []


def write_split(image_paths, class_labels, output_file):
    """
    Writes a split to a file.

    This function writes a split to a file. It iterates over the image paths and class labels and writes
    them to the file in the format 'image_path class_label'.

    Args:
        image_paths (list): A list of image paths.
        class_labels (list): A list of class labels.
        output_file (str): The path to the output file.
    """
    with open(output_file, 'w') as file:
        for image, label in zip(image_paths, class_labels):
            file.write(f'{image} {label}\n')


if __name__ == '__main__':
    data_dir = 'data/CroppedPatchesDataset/PN/train'
    output_dir = 'data/CroppedPatchesDataset/PN/'
    n_splits = 5
    output_prefix = 'split_'
    image_extension = '.jpg'

    # Set the seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Get the list of all the image paths in the good and bad folders
    good_images = get_image_list(data_dir, 'good', image_extension)
    bad_images = get_image_list(data_dir, 'bad', image_extension)

    # Print the number of good and bad images
    print(f'Number of good images: {len(good_images)}')
    print(f'Number of bad images: {len(bad_images)}')

    all_images = good_images + bad_images
    all_labels = [0] * len(good_images) + [1] * len(bad_images)
    print(f'Total images: {len(all_images)}')
    print(f'Total labels: {len(all_labels)}')

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for i, (train_index, val_index) in enumerate(skf.split(all_images, all_labels), start=1):
        train_images = [all_images[i] for i in train_index]
        train_labels = [all_labels[i] for i in train_index]
        val_images = [all_images[i] for i in val_index]
        val_labels = [all_labels[i] for i in val_index]

        print(f'Split {i}:')
        print(f'Number of train images: {len(train_images)}')
        print(f'Number of val images: {len(val_images)}')
        print(
            f'Number of good images in the train split: {len([image for image in train_images if image in good_images])}')
        print(
            f'Number of bad images in the train split: {len([image for image in train_images if image in bad_images])}')
        print(f'Number of good images in the val split: {len([image for image in val_images if image in good_images])}')
        print(f'Number of bad images in the val split: {len([image for image in val_images if image in bad_images])}')
        print("--------------------")

        # Write the train and val splits to the output files
        train_file = os.path.join(output_dir, f'{output_prefix}{i}_train.txt')
        val_file = os.path.join(output_dir, f'{output_prefix}{i}_val.txt')
        write_split(train_images, train_labels, train_file)
        write_split(val_images, val_labels, val_file)
