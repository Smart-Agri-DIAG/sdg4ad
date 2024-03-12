"""
This script is used to generate the train/val splits for the dataset with added synthetic images.
"""

import os
import random
import numpy as np

from generate_splits import get_image_list


def generate_splits_with_synthetic(data_dir, original_split, output_file, image_extension='jpg'):
    """
    Generates the train/val splits for the dataset.

    This function generates the train/val splits for the dataset. It first retrieves the list of all the image paths
    in the good and bad folders. It then uses the StratifiedKFold class from scikit-learn to generate the splits.
    It writes the train and val splits to the output files.

    Args:
        data_dir (str): The path to the data directory.
        output_dir (str): The path to the output directory.
        n_splits (int): The number of splits to generate.
        output_prefix (str): The prefix to use for the output files.
        image_extension (str): The extension of the images.
    """
    # Get the list of all the image paths
    synthetic_images = get_image_list(data_dir, image_extension)
    print(f'Number of synthetic images: {len(synthetic_images)}')

    labels = [1] * len(synthetic_images)

    with open(original_split, 'r') as f:
        original_split_lines = f.readlines()

    print(f'Number of real images: {len(original_split_lines)}')

    with open(output_file, 'w') as f:
        for line in original_split_lines:
            f.write(line)
        for image, label in zip(synthetic_images, labels):
            f.write(f'{image} {label}\n')

    print(f'Total images: {len(original_split_lines) + len(synthetic_images)}')


if __name__ == '__main__':
    data_dirs = ['data/synthetic_images/PN/split_1',
                 'data/synthetic_images/PN/split_2',
                 'data/synthetic_images/PN/split_3',
                 'data/synthetic_images/PN/split_4',
                 'data/synthetic_images/PN/split_5',
                 'data/synthetic_images/RG/split_1',
                 'data/synthetic_images/RG/split_2',
                 'data/synthetic_images/RG/split_3',
                 'data/synthetic_images/RG/split_4',
                 'data/synthetic_images/RG/split_5']
    original_splits = ['data/Splits/PN/split_1_train.txt',
                       'data/Splits/PN/split_2_train.txt',
                       'data/Splits/PN/split_3_train.txt',
                       'data/Splits/PN/split_4_train.txt',
                       'data/Splits/PN/split_5_train.txt',
                       'data/Splits/RG/split_1_train.txt',
                       'data/Splits/RG/split_2_train.txt',
                       'data/Splits/RG/split_3_train.txt',
                       'data/Splits/RG/split_4_train.txt',
                       'data/Splits/RG/split_5_train.txt']
    output_files = ['data/Splits/PN_with_synthetic/split_1_train.txt',
                    'data/Splits/PN_with_synthetic/split_2_train.txt',
                    'data/Splits/PN_with_synthetic/split_3_train.txt',
                    'data/Splits/PN_with_synthetic/split_4_train.txt',
                    'data/Splits/PN_with_synthetic/split_5_train.txt',
                    'data/Splits/RG_with_synthetic/split_1_train.txt',
                    'data/Splits/RG_with_synthetic/split_2_train.txt',
                    'data/Splits/RG_with_synthetic/split_3_train.txt',
                    'data/Splits/RG_with_synthetic/split_4_train.txt',
                    'data/Splits/RG_with_synthetic/split_5_train.txt']

    # Set the seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    for data_dir, original_split, output_file in zip(data_dirs, original_splits, output_files):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        print(f'Generating splits for {data_dir}...')
        generate_splits_with_synthetic(data_dir, original_split, output_file)
        print('Done!\n')
