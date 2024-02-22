import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


def get_train_transforms():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30, interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


def get_val_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])


class BinaryClassificationDataset(Dataset):
    def __init__(self, file_list_path, train=True):
        self.file_list = []
        with open(file_list_path, "r") as file:
            for line in file:
                img_path, class_label = line.strip().split()
                self.file_list.append((img_path, int(class_label)))

        if train:
            self.transforms = get_train_transforms()
        else:
            self.transforms = get_val_transforms()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path, class_label = self.file_list[idx]
        image = Image.open(img_path)
        image = self.transforms(image)
        class_label = torch.tensor(class_label)
        return image, class_label


if __name__ == '__main__':
    train_split_path = "data/CroppedPatchesDataset/PN/split_1_train.txt"
    val_split_path = "data/CroppedPatchesDataset/PN/split_1_val.txt"

    dataset = BinaryClassificationDataset(train_split_path)
    image, label = dataset[0]
    print(f'Image shape: {image.shape}, Label: {label}')
    print(f'Length of dataset: {len(dataset)}')

    val_dataset = BinaryClassificationDataset(val_split_path, train=False)
    val_image, val_label = val_dataset[0]
    print(f'Validation Image shape: {val_image.shape}, Validation Label: {val_label}')
    print(f'Length of validation dataset: {len(val_dataset)}')
