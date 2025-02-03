import random
import numpy as np
import torch
import yaml


# Function to load yaml configuration file
def load_config(file_path):
    with open(file_path, 'r') as file:
        cfg = yaml.safe_load(file)
    return cfg


def print_config(cfg):
    print("Configuration:")
    for key, value in cfg.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
    print("")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
