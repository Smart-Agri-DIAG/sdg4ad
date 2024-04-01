"""
compute_dced_params: finds the best parameters for the double canny edge detection so that it best classifies
the patches into bad and good.

It iterates over the parameter space and computes the best threshold for the train set. The best threshold is the one
that achieves the best balanced accuracy. The threshold is then applied to the validation set and the balanced accuracy
is computed. The best parameters are saved to a configuration file. This process is repeated for each split.
"""

import cv2
import numpy as np
from tqdm import tqdm
import yaml
from multiprocessing import Pool


def apply_dced(img, kernel_size, lower_th_wide, upper_th_wide, lower_th_narrow, upper_th_narrow):
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    edges_wide = cv2.Canny(img, lower_th_wide, upper_th_wide)
    edges_narrow = cv2.Canny(img, lower_th_narrow, upper_th_narrow)
    edges_diff = cv2.absdiff(edges_wide, edges_narrow)
    return edges_diff


def get_edge_percentages(img_paths, kernel_size, lower_th_wide, upper_th_wide, lower_th_narrow, upper_th_narrow, verbose):
    edge_percentage_list = []
    for img_path in tqdm(img_paths, disable=(not verbose)):
        img = cv2.imread(img_path)
        edges = apply_dced(img, kernel_size, lower_th_wide, upper_th_wide, lower_th_narrow, upper_th_narrow)
        edge_percentage_list.append(np.count_nonzero(edges) / (img.shape[0] * img.shape[1]) * 100)
    return np.array(edge_percentage_list)


def compute_metrics(good_edge_percentages, bad_edge_percentages, threshold):
    TP = np.sum(bad_edge_percentages >= threshold)
    TN = np.sum(good_edge_percentages < threshold)
    FP = np.sum(good_edge_percentages >= threshold)
    FN = np.sum(bad_edge_percentages < threshold)
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    specificity = TN / (TN + FP) if TN + FP > 0 else 0
    balanced_accuracy = (recall + specificity) / 2
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'specificity': specificity,
        'balanced_accuracy': balanced_accuracy
    }


def print_metrics(metrics):
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")
    print(f"F1 score: {metrics['f1_score']}")
    print(f"Specificity: {metrics['specificity']}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']}\n")


def get_best_threshold(good_img_paths, bad_img_paths, kernel_size, lower_th_wide, upper_th_wide, lower_th_narrow,
                       upper_th_narrow, verbose):
    good_edge_percentages = get_edge_percentages(good_img_paths, kernel_size, lower_th_wide, upper_th_wide,
                                                 lower_th_narrow, upper_th_narrow, verbose)
    bad_edge_percentages = get_edge_percentages(bad_img_paths, kernel_size, lower_th_wide, upper_th_wide,
                                                lower_th_narrow, upper_th_narrow, verbose)

    # Try different thresholds at 0.01% intervals and compute the best threshold and best balanced accuracy
    thresholds = np.arange(0, 100, 0.01)
    best_threshold = None
    best_metrics = None
    best_balanced_accuracy = 0
    for threshold in thresholds:
        metrics = compute_metrics(good_edge_percentages, bad_edge_percentages, threshold)
        if metrics['balanced_accuracy'] > best_balanced_accuracy:
            best_balanced_accuracy = metrics['balanced_accuracy']
            best_threshold = threshold
            best_metrics = metrics

    if verbose:
        print(f"Good average edge percentage: {np.mean(good_edge_percentages)}")
        print(f"Bad average edge percentage: {np.mean(bad_edge_percentages)}")
        print(f"Best threshold: {best_threshold}")
        print_metrics(best_metrics)

    return best_threshold


def worker(split, verbose=False):
    train_file_list_path = f"data/Splits/PN/split_{split}_train.txt"
    with open(train_file_list_path, 'r') as file:
        file_list = file.readlines()
    good_img_paths = [line.strip().split()[0] for line in file_list if line.strip().split()[1] == '0']
    bad_img_paths = [line.strip().split()[0] for line in file_list if line.strip().split()[1] == '1']

    val_file_list_path = f"data/Splits/PN/split_{split}_val.txt"
    with open(val_file_list_path, 'r') as file:
        val_file_list = file.readlines()
    val_good_img_paths = [line.strip().split()[0] for line in val_file_list if line.strip().split()[1] == '0']
    val_bad_img_paths = [line.strip().split()[0] for line in val_file_list if line.strip().split()[1] == '1']

    if verbose:
        print(f"Finding best parameters for split {split}\n")

    best_kernel_size = 0
    best_val_balanced_accuracy = 0
    best_lower_th_wide = 0
    best_upper_th_wide = 0
    best_lower_th_narrow = 0
    best_upper_th_narrow = 0
    best_val_metrics = None

    # Parameter sweep for the best thresholds
    for kernel_size in [3, 5, 7]:
        for lower_th_wide in range(0, 255, 25):
            for upper_th_wide in range(lower_th_wide + 25, 255, 25):
                for lower_th_narrow in range(lower_th_wide, 255, 25):
                    for upper_th_narrow in range(max(upper_th_wide + 25, lower_th_narrow + 25), 255, 25):
                        if verbose:
                            print(f"Trying parameters:")
                            print(f"kernel_size={kernel_size}")
                            print(f"lower_th_wide={lower_th_wide}, upper_th_wide={upper_th_wide}")
                            print(f"lower_th_narrow={lower_th_narrow}, upper_th_narrow={upper_th_narrow}\n")
                        threshold = get_best_threshold(good_img_paths, bad_img_paths, kernel_size, lower_th_wide,
                                                       upper_th_wide, lower_th_narrow, upper_th_narrow, verbose=verbose)

                        # Apply the best threshold to the validation set
                        val_good_edge_percentages = get_edge_percentages(val_good_img_paths, kernel_size, lower_th_wide,
                                                                         upper_th_wide, lower_th_narrow, upper_th_narrow,
                                                                         verbose)
                        val_bad_edge_percentages = get_edge_percentages(val_bad_img_paths, kernel_size, lower_th_wide,
                                                                        upper_th_wide, lower_th_narrow, upper_th_narrow,
                                                                        verbose)
                        val_metrics = compute_metrics(val_good_edge_percentages,
                                                      val_bad_edge_percentages, threshold)
                        if verbose:
                            print(f"Validation results:")
                            print_metrics(val_metrics)

                        if val_metrics['balanced_accuracy'] > best_val_balanced_accuracy:
                            best_val_balanced_accuracy = val_metrics['balanced_accuracy']
                            best_kernel_size = kernel_size
                            best_lower_th_wide = lower_th_wide
                            best_upper_th_wide = upper_th_wide
                            best_lower_th_narrow = lower_th_narrow
                            best_upper_th_narrow = upper_th_narrow
                            best_val_metrics = val_metrics

    if verbose:
        print(f"Best validation balanced accuracy: {best_val_balanced_accuracy}")
        print(f"Best kernel size: {best_kernel_size}")
        print(f"Lower threshold wide: {best_lower_th_wide}")
        print(f"Upper threshold wide: {best_upper_th_wide}")
        print(f"Lower threshold narrow: {best_lower_th_narrow}")
        print(f"Upper threshold narrow: {best_upper_th_narrow}")

    # Save the best parameters to a configuration file
    config = {
        'kernel_size': best_kernel_size,
        'lower_th_wide': best_lower_th_wide,
        'upper_th_wide': best_upper_th_wide,
        'lower_th_narrow': best_lower_th_narrow,
        'upper_th_narrow': best_upper_th_narrow,
        'val_precision': str(best_val_metrics['precision']),
        'val_recall': str(best_val_metrics['recall']),
        'val_f1_score': str(best_val_metrics['f1_score']),
        'val_specificity': str(best_val_metrics['specificity']),
        'val_balanced_accuracy': str(best_val_metrics['balanced_accuracy'])
    }
    with open(f"config/edge_statistics_split_{split}.yaml", 'w') as file:
        yaml.dump(config, file)


if __name__ == '__main__':
    splits = [1, 2, 3]
    verbose = False
    with Pool() as p:
        p.starmap(worker, [(split, verbose) for split in splits])
    print("Done!")
