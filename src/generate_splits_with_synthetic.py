"""
This script is used to generate the train/val splits for the dataset with added synthetic images.
"""

import os
import random
import numpy as np

from generate_splits import get_image_list


def generate_splits_with_synthetic(data_dir, original_split, output_root, image_extension='jpg'):
    """
    Generates the train/val splits for the dataset.

    This function generates the train/val splits for the dataset. It first retrieves the list of all the image paths
    in the good and bad folders. It then uses the StratifiedKFold class from scikit-learn to generate the splits.
    It writes the train and val splits to the output files.

    Args:
        data_dir (str): The path to the data directory.
        original_split (str): The path to the original split file.
        output_root (str): The root path to the output directory.
        image_extension (str): The extension of the synthetic images.
    """
    split_filename = original_split.split('/')[-1]
    folder_name = data_dir.split('/')[-2]

    with open(original_split, 'r') as f:
        original_split_lines = f.readlines()
    print(f'Number of real images: {len(original_split_lines)}')

    good_lines = [line for line in original_split_lines if line.rstrip().split(' ')[1] == '0']
    num_good = len(good_lines)
    print(f'Number of good images: {num_good}')

    bad_lines = [line for line in original_split_lines if line.rstrip().split(' ')[1] == '1']
    num_bad = len(bad_lines)
    print(f'Number of bad images: {num_bad}')

    assert num_good + num_bad == len(original_split_lines)

    # Get the list of all the image paths in the synthetic images folder
    synthetic_images = get_image_list(data_dir, image_extension)
    num_synthetic = len(synthetic_images)
    print(f'Number of synthetic bad images: {num_synthetic}')

    # Addition 100%
    output_file = os.path.join(output_root, f"{folder_name}_addition_100", split_filename)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.writelines(good_lines)
        f.writelines(bad_lines)
        for image in synthetic_images:
            f.write(f'{image} 1\n')
    assert num_good + num_bad + num_synthetic == len(open(output_file).readlines())

    # Addition 50%
    output_file = os.path.join(output_root, f"{folder_name}_addition_50", split_filename)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.writelines(good_lines)
        f.writelines(bad_lines)
        for image in random.sample(synthetic_images, num_synthetic // 2):
            f.write(f'{image} 1\n')

    # Addition 25%
    output_file = os.path.join(output_root, f"{folder_name}_addition_25", split_filename)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.writelines(good_lines)
        f.writelines(bad_lines)
        for image in random.sample(synthetic_images, num_synthetic // 4):
            f.write(f'{image} 1\n')

    # Addition 10%
    output_file = os.path.join(output_root, f"{folder_name}_addition_10", split_filename)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.writelines(good_lines)
        f.writelines(bad_lines)
        for image in random.sample(synthetic_images, num_synthetic // 10):
            f.write(f'{image} 1\n')

    # Substition 100%
    output_file = os.path.join(output_root, f"{folder_name}_substitution_100", split_filename)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.writelines(good_lines)
        for image in random.sample(synthetic_images, num_bad):
            f.write(f'{image} 1\n')
    assert num_good + num_bad == len(open(output_file).readlines())

    # Substition 50%
    output_file = os.path.join(output_root, f"{folder_name}_substitution_50", split_filename)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.writelines(good_lines)
        for line in random.sample(bad_lines, num_bad // 2):
            f.write(line)
        for image in random.sample(synthetic_images, num_bad - num_bad // 2):
            f.write(f'{image} 1\n')
    assert num_good + num_bad == len(open(output_file).readlines())

    # Substition 25%
    output_file = os.path.join(output_root, f"{folder_name}_substitution_25", split_filename)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.writelines(good_lines)
        for line in random.sample(bad_lines, num_bad // 4):
            f.write(line)
        for image in random.sample(synthetic_images, num_bad - num_bad // 4):
            f.write(f'{image} 1\n')
    assert num_good + num_bad == len(open(output_file).readlines())

    # Substition 10%
    output_file = os.path.join(output_root, f"{folder_name}_substitution_10", split_filename)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.writelines(good_lines)
        for line in random.sample(bad_lines, num_bad // 10):
            f.write(line)
        for image in random.sample(synthetic_images, num_bad - num_bad // 10):
            f.write(f'{image} 1\n')
    assert num_good + num_bad == len(open(output_file).readlines())


if __name__ == '__main__':
    data_dirs = ['data/synthetic_images_PN_paste1/split_1',
                 'data/synthetic_images_PN_paste1/split_2',
                 'data/synthetic_images_PN_paste1/split_3',
                 'data/synthetic_images_PN_paste3/split_1',
                 'data/synthetic_images_PN_paste3/split_2',
                 'data/synthetic_images_PN_paste3/split_3']
    original_splits = ['data/Splits/PN/split_1_train.txt',
                       'data/Splits/PN/split_2_train.txt',
                       'data/Splits/PN/split_3_train.txt',
                       'data/Splits/PN/split_1_train.txt',
                       'data/Splits/PN/split_2_train.txt',
                       'data/Splits/PN/split_3_train.txt']
    output_root = 'data/Splits/'
    img_extension = '.jpg'  # The extension of the synthetic images

    # Set the seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    for data_dir, original_split in zip(data_dirs, original_splits):
        print(f'Generating splits for {data_dir}...')
        generate_splits_with_synthetic(data_dir, original_split, output_root, img_extension)
        print("-----------------------------------\n")
    print('Done!\n')
