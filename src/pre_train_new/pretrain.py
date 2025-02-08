
import os
import sys

# Add parent directory to path
sys.path.append(".")

import random
from config import parse_args
import numpy as np
import torch
from tqdm import tqdm
from src.pre_train_new.utils import find_learning_rate, pre_train_model

if __name__ == "__main__":
    config = parse_args()

    print("Configurations:")
    for key, value in vars(config).items():
        print(f"{key}: {value}")

    # Fix seed
    seed = config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Enable CUDA
    use_cuda = config.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Device: {device}")
    learning_rate = find_learning_rate(config, device)

    print("Suggested Learning Rate:", learning_rate)
    pre_train_model(config, device, learning_rate)