import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


def get_train_transforms(resize=None, normalize=True):

    transform_list = [
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
    ]
    if resize is not None:
        transform_list.insert(0, transforms.Resize(resize))
    if normalize:
        transform_list.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    return transforms.Compose(transform_list)


def get_val_transforms(resize=None, normalize=True):
    transform_list = [
        transforms.ToTensor(),
    ]
    if resize is not None:
        transform_list.insert(0, transforms.Resize(resize))
    if normalize:
        transform_list.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    return transforms.Compose(transform_list)


class BinaryClassificationDataset(Dataset):
    def __init__(self, file_list_path, train=True, resize=None):
        pos_count = 0
        self.file_list = []
        with open(file_list_path, "r") as file:
            for line in file:
                img_path, class_label = line.strip().split()
                self.file_list.append((img_path, int(class_label)))
                if int(class_label) == 1:
                    pos_count += 1
        self.pos_weight = (len(self.file_list) - pos_count) / pos_count

        if train:
            self.transforms = get_train_transforms(resize)
        else:
            self.transforms = get_val_transforms(resize)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path, class_label = self.file_list[idx]
        image = Image.open(img_path)
        image = self.transforms(image)
        class_label = torch.tensor(class_label).float()
        return image, class_label


class AutoEncoderDataset(Dataset):
    def __init__(self, file_list_path, train=True):
        self.file_list = []
        with open(file_list_path, "r") as file:
            for line in file:
                img_path, class_label = line.strip().split()
                if int(class_label) == 0:
                    self.file_list.append(img_path)

        if train:
            self.transforms = get_train_transforms(resize=(256, 256), normalize=False)
        else:
            self.transforms = get_val_transforms(resize=(256, 256), normalize=False)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        image = Image.open(img_path)
        image = self.transforms(image)
        return image


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
