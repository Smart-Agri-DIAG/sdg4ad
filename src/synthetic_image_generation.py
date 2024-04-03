from utils import load_config, print_config, set_seed

import os
import cv2
import imageio
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA

# !{sys.executable} -m pip install 'git+https://github.com/facebookresearch/segment-anything.git'
# !wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


### Canny ###

def get_indices_of_edgiest_grapes(cfg, imgs, N):
    """
    Returns the indices of the N edgiest grapes in the list of images.

    It uses two Canny edge detectors with different thresholds and returns the index of the image
    with the highest difference in edge count. The edge count is normalized by the number of non-black pixels.

    Args:
        cfg (dict): Configuration dictionary.
        imgs (list): List of images.
        N (int): Number of edgiest grapes to return.

    Returns:
        list: Indices of the N edgiest grapes.
    """
    edge_counts = []
    for img in imgs:
        img = cv2.GaussianBlur(img, (cfg["kernel_size"], cfg["kernel_size"]), 0)
        num_edges_wide = cv2.countNonZero(cv2.Canny(img, cfg["lower_th_wide"], cfg["upper_th_wide"]))
        num_edges_narrow = cv2.countNonZero(cv2.Canny(img, cfg["lower_th_narrow"], cfg["upper_th_narrow"]))
        non_black_pixels = np.count_nonzero(np.any(img != [0, 0, 0], axis=-1))
        normalized_edge_count = abs(num_edges_wide - num_edges_narrow) / (non_black_pixels)
        edge_counts.append(normalized_edge_count)

    indices = np.argsort(edge_counts)[::-1][:N]
    indices = np.resize(indices, N)  # In case there are less than N grapes
    return indices.tolist()


### SAM ###

def get_mask_generator(cfg):
    """
    Returns the SAM mask generator.

    Args:
        cfg (dict): The configuration dictionary.

    Returns:
        SamAutomaticMaskGenerator: The mask generator.
    """
    sam = sam_model_registry[cfg["model_type"]](checkpoint=cfg["sam_checkpoint"]).to(device=cfg["device"])
    mask_generator = SamAutomaticMaskGenerator(model=sam, points_per_side=cfg["points_per_side"])
    return mask_generator


def filter_masks_by_area(masks, keep_ratio):
    """
    Filters a list of masks based on their area. It returns the top masks based on the keep_ratio,
    but only if their area is greater than the mean.

    Args:
        masks (list): A list of dictionaries where each dictionary represents a mask and has an 'area' key.
        keep_ratio (float): The ratio of masks to keep. The value should be between 0 (exclusive) and 1 (inclusive).

    Returns:
        list: The filtered masks.
    """
    assert 0 < keep_ratio <= 1, "keep_ratio should be between 0 (exclusive) and 1 (inclusive)."

    # Sort masks by area
    masks = sorted(masks, key=lambda x: x['area'])

    # Find the index of the first mask with area >= mean area
    areas = [mask['area'] for mask in masks]
    mean_area = np.mean(areas)
    index = np.searchsorted(areas, mean_area)

    # Keep the top masks only if their area is greater than the mean area
    index = max(index, int(len(masks) * (1-keep_ratio)))
    return masks[index:]


def generate_masks(image, mask_generator, keep_ratio):
    """
    Generates masks for the input image. The masks are filtered by area.

    Args:
        image (np.array): The input image.
        mask_generator (SamAutomaticMaskGenerator): The mask generator.
        keep_ratio (float): The ratio of masks to keep. The value should be between 0 (exclusive) and 1 (inclusive).

    Returns:
        list: The generated masks.
    """
    masks = mask_generator.generate(image)
    masks = filter_masks_by_area(masks, keep_ratio)
    masks = [mask['segmentation'] for mask in masks]
    return masks


### Utils ###

def read_file_list(file_path):
    """
    Reads the file list and returns the paths of the good and bad grapes.

    Args:
        file_path (str): The path to the file list.

    Returns:
        list: The paths of the good grapes.
        list: The paths of the bad grapes.
    """
    good_paths = []
    bad_paths = []
    with open(file_path, 'r') as file:
        for line in file:
            path, label = line.strip().split()
            if label == '0':  # Good grape
                good_paths.append(path)
            else:  # Grape with anomaly
                bad_paths.append(path)
    return good_paths, bad_paths


def write_log(index, good_image_path, bad_image_path, log_folder):
    """
    Writes the log of the generated anomaly.

    Args:
        index (int): The index of the anomaly.
        good_image_path (str): The path of the good grape image.
        bad_image_path (str): The path of the bad grape image.
        log_folder (str): The path of the log folder.
    """
    with open(os.path.join(log_folder, f"log_anomaly_{index}.txt"), "a") as f:
        f.write(f"Anomaly {index} generated with:\n")
        f.write(f"Good image: {os.path.basename(good_image_path)}\n")
        f.write(f"Bad image: {os.path.basename(bad_image_path)}\n")


def get_biggest_component(mask):
    """
    Returns the biggest component of the mask.

    Args:
        mask (np.array): The input mask.

    Returns:
        np.array: The biggest component of the mask.
    """
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_label).astype(np.uint8)


### Rotation and Scaling ###

def get_rotation_matrix(mask, reference_mask):
    """
    Returns the rotation matrix to align the principal axes of the mask with the reference mask.

    Args:
        mask (np.array): The mask to align.
        reference_mask (np.array): The reference mask.

    Returns:
        np.array: The rotation matrix.
    """
    # Find indices of non-zero elements (row, column) -> (y, x)
    mask_points = np.argwhere(mask)
    ref_points = np.argwhere(reference_mask)

    # Fit PCA on mask and reference mask points
    pca = PCA(n_components=2)
    pca.fit(mask_points)
    ref_pca = PCA(n_components=2)
    ref_pca.fit(ref_points)

    # Get primary principal components from PCA (y,x)
    principal_axis = pca.components_[0]
    ref_principal_axis = ref_pca.components_[0]

    # Only consider the direction of the principal axis in the positive y direction
    if principal_axis[0] < 0:
        principal_axis = -principal_axis
    if ref_principal_axis[0] < 0:
        ref_principal_axis = -ref_principal_axis

    # Compute angles of principal axes (positive -> clockwise)
    angle = np.rad2deg(np.arctan2(principal_axis[0], principal_axis[1]))
    ref_angle = np.rad2deg(np.arctan2(ref_principal_axis[0], ref_principal_axis[1]))

    # Compute rotation angle
    rotation_angle = ref_angle - angle

    # Get rotation matrix
    moments = cv2.moments(mask)
    center = (moments['m10'] / moments['m00'], moments['m01'] / moments['m00'])
    rotation_matrix = cv2.getRotationMatrix2D(center, -rotation_angle, 1.0)  # positive angle -> counter-clockwise
    return rotation_matrix


def rotate_grape(image, mask, reference_mask):
    """
    Rotates the grape (image and mask) to align the principal axes of the mask with the reference mask.

    Args:
        image (np.array): The image of the grape.
        mask (np.array): The mask of the grape.
        reference_mask (np.array): The reference mask.

    Returns:
        np.array: The rotated image.
        np.array: The rotated mask.
    """
    # Get rotation matrix
    rotation_matrix = get_rotation_matrix(mask, reference_mask)

    # Rotate the image
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC)

    # Rotate the mask
    rotated_mask = cv2.warpAffine(mask, rotation_matrix, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST)
    return rotated_image, rotated_mask


def scale_grape(image, mask, reference_mask):
    """
    Scales the grape (image and mask) to match the size of the reference mask.

    Args:
        image (np.array): The image of the grape.
        mask (np.array): The mask of the grape.
        reference_mask (np.array): The reference mask.

    Returns:
        np.array: The scaled image.
        np.array: The scaled mask.
    """
    # Compute moments
    moments = cv2.moments(mask)
    ref_moments = cv2.moments(reference_mask)

    # Compute the scale factor
    scale_factor = np.sqrt(ref_moments['m00'] / moments['m00'])

    # Scale the image
    img_interp = cv2.INTER_CUBIC if scale_factor > 1 else cv2.INTER_AREA
    scaled_image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=img_interp)

    # Scale the mask
    scaled_mask = cv2.resize(mask, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)

    return scaled_image, scaled_mask


### Blending ###

def PoissonBlending(source, destination, bad_mask, good_mask, cloning_type):
    """
    Blends the source image with the destination image using Poisson blending.

    Args:
        source (np.array): The source image.
        destination (np.array): The destination image.
        bad_mask (np.array): The mask of the bad grape.
        good_mask (np.array): The mask of the good grape.
        cloning_type (int): The type of cloning.

    Returns:
        np.array: The blended image.
    """
    # Get dimensions of the bad mask
    x_bad, y_bad, w_bad, h_bad = cv2.boundingRect(bad_mask)
    center_bad = (x_bad + w_bad // 2, y_bad + h_bad // 2)

    # Set the center for the blending to be the centroid of the good grape
    moments = cv2.moments(good_mask)
    center_good = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))

    # Calculate offset for aligning bad mask with good mask centroid
    offset = (center_good[0] - center_bad[0], center_good[1] - center_bad[1])

    # # Move the bad grape to the centroid of the good mask
    M = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]])
    bad_mask = cv2.warpAffine(bad_mask, M, (destination.shape[1], destination.shape[0]), flags=cv2.INTER_NEAREST)
    source = cv2.warpAffine(source, M, (destination.shape[1], destination.shape[0]))

    # # Get the intersection of the aligned masks
    intersection = cv2.bitwise_and(bad_mask, good_mask)

    # Get the bounding box of the intersection
    _, _, w_inter, h_inter = cv2.boundingRect(intersection)

    # Add padding around destination to make sure that the source will not be outside the image
    destination = cv2.copyMakeBorder(destination, h_inter, h_inter, w_inter, w_inter, cv2.BORDER_REFLECT101)
    center_good = (center_good[0] + w_inter, center_good[1] + h_inter)

    # Poisson blending
    blended = cv2.seamlessClone(source, destination, intersection*255, center_good, cloning_type)

    # Remove the padding
    blended = blended[h_inter:-h_inter, w_inter:-w_inter]

    return blended


def generate_synthetic_image(cfg, img_good, img_bad, mask_generator):
    """
    Generates a synthetic image by blending a bad grape onto an image of good grapes.

    Args:
        cfg (dict): Configuration dictionary.
        img_good (np.array): The image of good grapes.
        img_bad (np.array): The image of bad grapes.
        mask_generator (SamAutomaticMaskGenerator): The mask generator.

    Returns:
        np.array: The synthetic image.
    """
    # Generate masks
    good_masks = generate_masks(img_good, mask_generator, cfg["keep_ratio"])
    bad_masks = generate_masks(img_bad, mask_generator, cfg["keep_ratio"])

    # Sample 3 good masks
    good_indices = random.choices(range(len(good_masks)), k=1)
    good_masks = [good_masks[i].astype(np.uint8) for i in good_indices]

    # Sample 3 bad masks
    bad_grapes = [img_bad * mask[:, :, None] for mask in bad_masks]
    bad_indices = get_indices_of_edgiest_grapes(cfg, bad_grapes, 1)
    bad_masks = [bad_masks[i].astype(np.uint8) for i in bad_indices]
    bad_grapes = [bad_grapes[i] for i in bad_indices]

    for good_mask, bad_mask, bad_grape in zip(good_masks, bad_masks, bad_grapes):
        # Check if the masks are made of more than one connected component
        bad_mask = get_biggest_component(bad_mask)
        good_mask = get_biggest_component(good_mask)

        # Rotate the bad grape to match the orientation of the good grape
        bad_grape, bad_mask = rotate_grape(bad_grape, bad_mask, good_mask)

        # Scale the bad grape to match the size of the good grape
        bad_grape, bad_mask = scale_grape(bad_grape, bad_mask, good_mask)

        # Blend the bad grape onto the good image
        img_good = PoissonBlending(bad_grape, img_good, bad_mask, good_mask, cv2.NORMAL_CLONE)
    return img_good


if __name__ == "__main__":
    cfg_paths = ["config/synthetic_generation/config_synthetic_generation_split_1.yaml",
                 "config/synthetic_generation/config_synthetic_generation_split_2.yaml",
                 "config/synthetic_generation/config_synthetic_generation_split_3.yaml"]

    for cfg_path in cfg_paths:
        cfg = load_config(cfg_path)
        print_config(cfg)
        set_seed(cfg['seed'])

        mask_generator = get_mask_generator(cfg)

        print(f"Generating synthetic images for {cfg['file_path']}...")
        good_image_paths, bad_image_paths = read_file_list(cfg['file_path'])
        log_folder = os.path.join(cfg['output_path'], "logs")
        os.makedirs(log_folder, exist_ok=True)

        num_good_images = len(good_image_paths)
        num_bad_images = len(bad_image_paths)
        for index, bad_image_path in tqdm(enumerate(bad_image_paths), desc="Processing images", total=num_bad_images):
            good_image_path = random.choice(good_image_paths)

            img_good = cv2.imread(good_image_path)
            img_good = cv2.cvtColor(img_good, cv2.COLOR_BGR2RGB)

            img_bad = cv2.imread(bad_image_path)
            img_bad = cv2.cvtColor(img_bad, cv2.COLOR_BGR2RGB)

            new_img = generate_synthetic_image(cfg, img_good, img_bad, mask_generator)

            write_log(index, good_image_path, bad_image_path, log_folder)
            imageio.imsave(f"{cfg['output_path']}/anomaly_{index}.jpg", new_img)

        print("Synthetic images generated successfully!")
