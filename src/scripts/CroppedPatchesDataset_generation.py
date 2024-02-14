"""
This script computes the crops starting from the images in the AnomalyDataset and using the original yolo
crop coordinates computed the first time it was created.

The directory structure is the following:
CroppedPatchesDataset
│
└───PN
│   └───train
|   |   └───good
|   |   └───bad
│   └───test
|   |   └───good
|   |   └───bad
└───RG
    └───train
    |   └───good
    |   └───bad
    └───test
        └───good
        └───bad

In each "good" and "bad" folder there are the yolo txt files with the same name of the source image file contained in
the AnomalyDataset. The structure of the AnomalyDataset is similar to the previous one:
AnomalyDataset
│
└───PN
│   └───train
|   |   └───good
|   |   └───bad
│   └───test
|   |   └───good
|   |   └───bad
|   └───raw_test
└───RG
    └───train
    |   └───good
    |   └───bad
    └───test
    |   └───good
    |   └───bad
    └───raw_test
    └───raw_train

The script will read the yolo txt files and extrapolate the relative path to the source image file. Thei it will crop
the image and save it in the correct folder in the CroppedPatchesDataset.
"""
import os
import cv2
from config.definitions import ROOT_DIR
from pathlib import Path
from crop_dataset_generation import crop_image, parse_yolo_format


CROPPED_PATCHES_DATASET_DIR = Path(ROOT_DIR) / 'data/CroppedPatchesDataset'
ANOMALY_DATASET_DIR = Path(ROOT_DIR) / 'data/AnomalyDataset'

if __name__ == '__main__':
    # Iterate over the folders in the CroppedPatchesDataset and read each yolo txt file
    for dataset in CROPPED_PATCHES_DATASET_DIR.iterdir():
        if dataset.is_dir():
            for split in dataset.iterdir():
                if split.is_dir():
                    for label in split.iterdir():
                        if label.is_dir():
                            for yolo_file in label.glob('*.txt'):
                                # Read the yolo txt file and extract the image path
                                with open(yolo_file, 'r') as file:
                                    lines = file.readlines()
                                    image_path = ANOMALY_DATASET_DIR / os.path.relpath(yolo_file.parents[0], yolo_file.parents[3]) / yolo_file.with_suffix('.jpg').name
                                    # open the image to get the width and height
                                    image = cv2.imread(str(image_path))
                                    # Extract the bounding box coordinates
                                    bboxes = parse_yolo_format(lines, image.shape[1], image.shape[0])
                                    # Crop the image and save it in the correct folder
                                    cropped_images = crop_image(image_path, bboxes)
                                    for i, cropped_image in enumerate(cropped_images):
                                        cropped_image_path = CROPPED_PATCHES_DATASET_DIR / dataset / split / label / f'{image_path.stem}_{i:02d}.jpg'
                                        cv2.imwrite(str(cropped_image_path), cropped_image)
                                        print(f'Image saved at {cropped_image_path}')