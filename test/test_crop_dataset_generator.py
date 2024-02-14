import os
import cv2
from pathlib import Path
from src.crop_dataset_generation import crop_image, parse_yolo_format


def test_cropped_images():
    """
    Test function that compares cropped images with ground truth images pixel by pixel.

    """
    ref_img_path = 'data'
    ref_img_name = 'IMG_20210924_124738392_HDR'
    ref_bbox_file = 'data/IMG_20210924_124738392_HDR.txt'

    # load the reference image with opencv
    ref_img = cv2.imread(os.path.join(ref_img_path, ref_img_name) + '.jpg')

    # load the bounding box data
    with open(ref_bbox_file, 'r') as f:
        bbox_data = f.readlines()

    # parse the bounding box data
    image_width, image_height = ref_img.shape[1], ref_img.shape[0]
    pixel_bboxes = parse_yolo_format(bbox_data, image_width, image_height)

    # crop the image
    cropped_imgs = crop_image(os.path.join(ref_img_path, ref_img_name) + '.jpg', pixel_bboxes)
    #save the cropped images temporarily
    for i, cropped_img in enumerate(cropped_imgs):
        cropped_img_path = Path(ref_img_path) / f'{ref_img_name}_{i:02d}_r.jpg'
        cv2.imwrite(str(cropped_img_path), cropped_img)
    # Iterate over the cropped images and their corresponding ground truth
    for i in range(len(cropped_imgs)):
        ref_cropped_img_path = Path(ref_img_path) / f'{ref_img_name}_{i:02d}.jpg'
        ref_cropped_img = cv2.imread(str(ref_cropped_img_path))
        cropped_img_path = Path(ref_img_path) / f'{ref_img_name}_{i:02d}_r.jpg'
        cropped_img = cv2.imread(str(cropped_img_path))

        # Compare the cropped image with the ground truth image pixel by pixel
        assert cropped_img.shape == ref_cropped_img.shape
        assert (cropped_img == ref_cropped_img).all()



