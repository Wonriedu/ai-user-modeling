import os
import json
import pickle

import torch

from torch.utils.data import DataLoader, random_split
from torch.optim import Adam

from data_loaders.pulja_data_loader import PuljaDataLoader
from models._20220520_00 import UserModel
from models.utils import collate_fn


def main():
    ckpt_path = "ckpts"
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    batch_size = 256
    train_ratio = 0.9

    seq_len = 100

    dim_v = 50

    dataset = PuljaDataLoader(seq_len)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = UserModel(dataset.num_c, dataset.num_d, dim_v).to(device)

    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=test_size, shuffle=True,
        collate_fn=collate_fn
    )

    opt = Adam(model.parameters())


if __name__ == "__main__":
    main()
