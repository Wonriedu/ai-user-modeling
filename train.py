import os
import json
import pickle

import torch

from torch.utils.data import DataLoader, random_split
from torch.optim import Adam

from data_loaders.pulja_data_loader import PuljaDataLoader
from models.utils import collate_fn


def main():
    ckpt_path = "ckpts"
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    batch_size = 256
    seq_len = 100

    dataset = PuljaDataLoader(seq_len)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
