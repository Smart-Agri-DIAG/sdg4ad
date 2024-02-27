from datetime import datetime
import os
from matplotlib import pyplot as plt
import torch
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import wandb
from tqdm import tqdm

from src.models import CAE
from src.data import BinaryClassificationDataset, AutoEncoderDataset
from src.utils import load_config, print_config, set_seed


def train_1_epoch(model, optimizer, loss_fn, dataloader, device, scaler=None):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(total=len(dataloader), desc="Training")

    for inputs in dataloader:
        inputs = inputs.to(device)

        with autocast(enabled=scaler is not None):
            outputs = model(inputs)
            loss = loss_fn(outputs, inputs)

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
    train_split_path = os.path.join(cfg["splits_path"], f"split_{cfg['split']}_train.txt")
    train_dataset = AutoEncoderDataset(train_split_path)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=4)

    val_split_path = os.path.join(cfg["splits_path"], f"split_{cfg['split']}_val.txt")
    val_dataset = AutoEncoderDataset(val_split_path, train=False)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = torch.nn.MSELoss()
    scaler = GradScaler() if cfg["mixed_precision"] else None
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg["epochs"]//3+1)

    best_val_loss = float("inf")
    best_epoch = 0
    patience = 0

    for epoch in range(cfg["epochs"]):
        train_loss = train_1_epoch(model, optimizer, loss_fn, train_dataloader, device, scaler)
        val_loss = validate(model, loss_fn, val_dataloader, device)
        scheduler.step()

        print(f"\nEpoch {epoch + 1}")
        print(f"Train Loss: {train_loss}")
        print(f"Validation loss: {val_loss}")

        print(f"Learning Rate: {optimizer.param_groups[0]['lr']}")

        wandb.log({
            "Train Loss": train_loss,
            "Validation Loss": val_loss,
            "Learning Rate": optimizer.param_groups[0]["lr"],
            "Best Epoch": best_epoch,
        }, step=epoch + 1)

        if cfg["cae_inference_interval"] > 0 and (epoch + 1) % cfg["cae_inference_interval"] == 0:
            # Visualization
            model.eval()
            with torch.no_grad():
                data = next(iter(val_dataloader))
                data = data.to(device)
                output = model(data)

                fig, axs = plt.subplots(1, 2, figsize=(6, 12))
                axs[0].imshow(data[0].cpu().numpy().transpose((1, 2, 0)))  # Input image
                axs[1].imshow(output[0].cpu().numpy().transpose((1, 2, 0)))  # Predicted image
                plt.show()
                wandb.log({"Reconstruction": [wandb.Image(fig, caption="Reconstruction")]}, step=epoch + 1)

        if val_loss < best_val_loss:
            patience = 0
            best_epoch = epoch + 1
            best_val_loss = val_loss
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

    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs.to(device)
            with autocast():
                outputs = model(inputs)

            loss = loss_fn(outputs, inputs)
            total_loss += loss.item()

    return total_loss / len(dataloader)


if __name__ == "__main__":
    cfg = load_config("config/config.yaml")

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
