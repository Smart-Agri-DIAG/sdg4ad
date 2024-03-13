
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


def train_1_epoch(model, optimizer, loss_fn, dataloader, device, scaler=None):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(total=len(dataloader), desc="Training")

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        with autocast(enabled=scaler is not None):
            outputs = model(inputs).squeeze(dim=1)
            loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"Batch loss": f"{loss.item():.3f}"})
        progress_bar.update()

    return total_loss / len(dataloader)


def train(cfg):
    train_split_path = os.path.join(cfg["train_splits_path"], f"split_{cfg['split']}_train.txt")
    train_dataset = BinaryClassificationDataset(train_split_path, resize=(512, 512))
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])

    val_split_path = os.path.join(cfg["val_splits_path"], f"split_{cfg['split']}_val.txt")
    val_dataset = BinaryClassificationDataset(val_split_path, train=False, resize=(512, 512))
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BinaryClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(train_dataset.pos_weight).to(device))
    scaler = GradScaler() if cfg["mixed_precision"] else None
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg["epochs"]//3+1)

    best_val_loss = float("inf")
    best_epoch = 0
    patience = 0

    for epoch in range(cfg["epochs"]):
        train_loss = train_1_epoch(model, optimizer, loss_fn, train_dataloader, device, scaler)
        val_metrics = validate(model, loss_fn, val_dataloader, device)
        scheduler.step()

        print(f"\nEpoch {epoch + 1}, Train Loss: {train_loss:.4f}")
        print("Validation metrics:")
        print(f"    Loss: {val_metrics['loss']:.4f}")
        print(f"    Precision: {val_metrics['precision']:.4f}")
        print(f"    Recall: {val_metrics['recall']:.4f}")
        print(f"    F1-score: {val_metrics['f1_score']:.4f}")
        print(f"    Balanced Accuracy: {val_metrics['balanced_accuracy']:.4f}")
        print(f"    TP: {val_metrics['tp']}, TN: {val_metrics['tn']}, FP: {val_metrics['fp']}, FN: {val_metrics['fn']}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']}")

        wandb.log({
            "Train Loss": train_loss,
            "Validation Loss": val_metrics["loss"],
            "Validation Balanced Accuracy": val_metrics["balanced_accuracy"],
            "Validation Precision": val_metrics["precision"],
            "Validation Recall": val_metrics["recall"],
            "Validation F1-score": val_metrics["f1_score"],
            "Validation TP": val_metrics["tp"],
            "Validation TN": val_metrics["tn"],
            "Validation FP": val_metrics["fp"],
            "Validation FN": val_metrics["fn"],
            "Learning Rate": optimizer.param_groups[0]["lr"],
            "Best Epoch": best_epoch,
        }, step=epoch + 1)

        if val_metrics["loss"] < best_val_loss:
            patience = 0
            best_epoch = epoch + 1
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), os.path.join(cfg["checkpoint_dir"], "best_model.pth"))
            wandb.save(os.path.join(cfg["checkpoint_dir"], "best_model.pth"))
        elif cfg["early_stopping"]:
            patience += 1
            print(f"Validation loss did not improve. Patience ({patience}/{cfg['patience']})")
            if patience == cfg["patience"]:
                print(f"Early stopping after {epoch + 1} epochs.")
                break

    wandb.finish()


def validate(model, loss_fn, dataloader, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast():
                outputs = model(inputs).squeeze(dim=1)

            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
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
        "loss": total_loss / len(dataloader),
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

    # Add subfolder to checkpoint dir with current date and time
    cfg["checkpoint_dir"] = os.path.join(cfg["checkpoint_dir"], datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    # Setup Weight and Biases
    wandb.init(project="sdg4ad", entity="canopies-diag", config=cfg, mode=cfg["wandb_mode"])
    wandb.run.log_code(".")
    cfg = wandb.config  # Needed when running sweeps

    print_config(cfg)
    set_seed(cfg["seed"])

    train(cfg)
