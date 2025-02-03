from torch.cuda.amp import autocast
import numpy as np
import torch
import os
from PIL import Image
from tqdm import tqdm

from utils import load_config, print_config, set_seed
from data import BinaryClassificationDataset
from models import BinaryClassifier


def save_predictions(inputs, preds, labels, batch_num):
    for i, (input, pred, label) in enumerate(zip(inputs, preds, labels)):
        # Save images with wrong predictions
        pred = int(pred.cpu().numpy())
        label = int(label.cpu().numpy())
        image = input.permute(1, 2, 0).cpu().numpy()
        image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image)
        if pred != label:
            image.save(
                f"predictions/wrong_predictions/pred{str(pred)}_label{str(label)}_batch{batch_num}_{i}.png")
        else:
            image.save(f"predictions/correct_predictions/pred{str(pred)}_label{str(label)}_batch{batch_num}_{i}.png")


def test(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_num, (inputs, labels) in enumerate(tqdm(dataloader, desc="Testing")):
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast():
                outputs = model(inputs).squeeze(dim=1)

            preds = torch.sigmoid(outputs) > 0.5

            save_predictions(inputs, preds, labels, batch_num)

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
    model.load_state_dict(torch.load("weights/best_model.pth"))
    model.to(device)

    dataset = BinaryClassificationDataset("data/Splits/PN/split_2_val.txt", train=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    os.makedirs("predictions/wrong_predictions", exist_ok=True)
    os.makedirs("predictions/correct_predictions", exist_ok=True)

    results = test(model, dataloader, device)

    print(results)
