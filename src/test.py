
import sys
from tqdm import tqdm
import wandb
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import torch
import os
from datetime import datetime

from utils import load_config, print_config, set_seed
from data import BinaryClassificationDataset
from models import BinaryClassifier


def train(cfg):

    val_split_path = os.path.join(cfg["val_splits_path"], f"split_{cfg['split']}_val.txt")
    val_dataset = BinaryClassificationDataset(val_split_path, train=False, resize=(224, 224))
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BinaryClassifier().to(device)
    scaler = GradScaler() if cfg["mixed_precision"] else None


def test(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast():
                outputs = model(inputs).squeeze(dim=1)

            preds = torch.sigmoid(outputs) > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    TP = np.sum((all_preds == 1) & (all_labels == 1))
    TN = np.sum((all_preds == 0) & (all_labels == 0))
    FP = np.sum((all_preds == 1) & (all_labels == 0))
    FN = np.sum((all_preds == 0) & (all_labels == 1))

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)
    specificity = TN / (TN + FP)
    balanced_accuracy = (recall + specificity) / 2

    results = {
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tp": TP,
        "tn": TN,
        "fp": FP,
        "fn": FN
    }
    return results


if __name__ == "__main__":
    cfg = load_config("config/config_train.yaml")
    print_config(cfg)
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BinaryClassifier()
    model.load_state_dict(torch.load("weights/pn_addition_10_split_3.pth"))
    model.to(device)

    dataset = BinaryClassificationDataset("data/Splits/PN_test.txt", train=False, resize=(600, 600))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    results = test(model, dataloader, device)

    print(results)
