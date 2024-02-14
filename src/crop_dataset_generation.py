import cv2
from pathlib import Path
import numpy as np


# Define a function to parse YOLO format bounding boxes and convert them to pixel coordinates
def parse_yolo_format(bbox_data, image_width, image_height):
    """
    Parse YOLO format bounding boxes and convert normalized coords to pixel coords.

    Parameters:
    bbox_data (list of str): Bounding box data in YOLO format.
    image_width (int): Width of the image.
    image_height (int): Height of the image.

    Returns:
    list of tuple: List containing pixel coordinates for bounding boxes.
    """
    bboxes = []
    for line in bbox_data:
        _, x_center_norm, y_center_norm, width_norm, height_norm = map(float, line.split())
        x_center, y_center = round(x_center_norm * image_width), round(y_center_norm * image_height)
        width, height = round(width_norm * image_width), round(height_norm * image_height)
        # Convert to pixel coordinates: top left x, top left y, bottom right x, bottom right y
        x1 = int(x_center - (width / 2))
        y1 = int(y_center - (height / 2))
        x2 = int(x_center + (width / 2))
        y2 = int(y_center + (height / 2))
        bboxes.append((x1, y1, x2, y2))
    return bboxes


# # Test the function using the image width and height from the sample image uploaded
# sample_image_path = '/mnt/data/IMG_20210924_124738392_HDR.jpg'
# with Image.open(sample_image_path) as img:
#     image_width, image_height = img.size
#
# # Now, let's parse the YOLO bounding boxes to pixel coordinates
# pixel_bboxes = parse_yolo_format(bbox_data, image_width, image_height)
# pixel_bboxes


# Define a function to crop images based on the bounding box coordinates
def crop_image(image_input, bbox_coordinates):
    """
    Crop the image based on the bounding box coordinates.

    Parameters:
    image_input: Can be PAth, str or the image itself (opened with opencv)
    bbox_coordinates (list of tuple): List containing pixel coordinates for bounding boxes.

    Returns:
    list of Image: List containing cropped Image objects.
    """
    if not isinstance(bbox_coordinates, list):
        raise ValueError('bbox_coordinates should be a list of tuples')
    if isinstance(image_input, Path):
        image_input = str(image_input)
    if isinstance(image_input, np.ndarray):
        image = image_input
    else:
        image = cv2.imread(image_input)
    cropped_images = []
    for bbox in bbox_coordinates:
        cropped_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        cropped_images.append(cropped_image)
    return cropped_images

