import os
from matplotlib import pyplot as plt
import torch
import numpy as np
from torch.cuda.amp import autocast, GradScaler

from src.models import CAE
from src.data import BinaryClassificationDataset, AutoEncoderDataset
from src.utils import load_config, print_config, set_seed


def get_mean_std_losses(model, dataloader, loss_fn, device, scaler=None):
    assert dataloader.batch_size == 1
    model.eval()

    good_losses = []
    bad_losses = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            with autocast(enabled=scaler is not None):
                outputs = model(inputs)
                loss = loss_fn(outputs, inputs)

            if labels[0] == 0:
                good_losses.append(loss.item())
            else:
                bad_losses.append(loss.item())

    good_losses = np.array(good_losses)
    mean_good_loss = np.mean(good_losses)
    std_good_loss = np.std(good_losses)

    bad_losses = np.array(bad_losses)
    mean_bad_loss = np.mean(bad_losses)
    std_bad_loss = np.std(bad_losses)

    return mean_good_loss, std_good_loss, mean_bad_loss, std_bad_loss


def evaluate(model, dataloader, loss_fn, threshold, device, scaler):
    model.eval()
    good_losses = []
    bad_losses = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast():
                outputs = model(inputs).squeeze(dim=1)

            loss = loss_fn(outputs, inputs)
            if labels[0] == 0:
                good_losses.append(loss.item())
            else:
                bad_losses.append(loss.item())

            pred = loss.item() > threshold
            all_preds.append(pred)
            all_labels.append(labels[0].cpu().numpy())

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

    good_losses = np.array(good_losses)
    mean_good_loss = np.mean(good_losses)
    std_good_loss = np.std(good_losses)

    bad_losses = np.array(bad_losses)
    mean_bad_loss = np.mean(bad_losses)
    std_bad_loss = np.std(bad_losses)

    results = {
        "mean good loss": mean_good_loss,
        "std good loss": std_good_loss,
        "mean bad loss": mean_bad_loss,
        "std bad loss": std_bad_loss,
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
    cfg = load_config("config/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_split_path = os.path.join(cfg["splits_path"], f"split_{cfg['split']}_train.txt")
    train_dataset = BinaryClassificationDataset(train_split_path, train=False, resize=(256, 256))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)

    model = CAE().to(device)
    model.load_state_dict(torch.load("weights/cae.pth"))
    model.eval()

    loss_fn = torch.nn.MSELoss()
    scaler = GradScaler() if cfg["mixed_precision"] else None

    mean_good_loss, std_good_loss, mean_bad_loss, std_bad_loss = get_mean_std_losses(
        model, train_dataloader, loss_fn, device, scaler)

    threshold1 = (mean_good_loss + mean_bad_loss) / 2
    threshold2 = mean_good_loss + std_good_loss

    print(f"Mean good loss: {mean_good_loss:.4f}, Std good loss: {std_good_loss:.4f}")
    print(f"Mean bad loss: {mean_bad_loss:.4f}, Std bad loss: {std_bad_loss:.4f}")
    print(f"Threshold 1 (mean_good_loss + mean_bad_loss) / 2: {threshold1:.4f}")
    print(f"Threshold 2 (mean good loss + std good loss): {threshold2:.4f}")

    val_split_path = os.path.join(cfg["splits_path"], f"split_{cfg['split']}_val.txt")
    val_dataset = BinaryClassificationDataset(val_split_path, train=False, resize=(256, 256))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    results1 = evaluate(model, val_dataloader, loss_fn, threshold1, device, scaler)
    results2 = evaluate(model, val_dataloader, loss_fn, threshold2, device, scaler)

    print(f"Results with threshold 1: {threshold1:.4f}")
    print(results1)
    print(f"Results with threshold 2: {threshold2:.4f}")
    print(results2)
