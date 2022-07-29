import os
import pickle

import torch

from torch.utils.data import DataLoader, random_split
from torch.optim import Adam

from data_loaders.pulja_data_loader_02 import PuljaDataLoader
from models._20220729_10 import UserModel
from models.utils_02 import collate_fn


def main():
    ckpt_path = "ckpts"
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    batch_size = 256
    num_epochs = 1000
    train_ratio = 0.9

    seq_len = 1000

    dim_v = 20

    dataset = PuljaDataLoader(seq_len)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = UserModel(
        dataset.num_c1,
        dataset.num_c2,
        dataset.num_c3,
        dataset.num_d,
        dim_v
    ).to(device)

    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size]
    )

    if os.path.exists(os.path.join(dataset.dataset_dir, "indices.pkl")):
        with open(os.path.join(dataset.dataset_dir, "indices.pkl"), "rb") as f:
            indices = pickle.load(f)
        train_dataset.indices = indices[0]
        test_dataset.indices = indices[1]
    else:
        with open(os.path.join(dataset.dataset_dir, "indices.pkl"), "wb") as f:
            pickle.dump((train_dataset.indices, test_dataset.indices), f)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=test_size, shuffle=True,
        collate_fn=collate_fn
    )

    opt = Adam(model.parameters())

    train_loss_means, test_loss_means, aucs = model.train_model(
        train_loader, test_loader, num_epochs, opt, ckpt_path
    )
    import numpy as np
    print(np.min(test_loss_means))
    print(np.max(aucs))

    with open(os.path.join(ckpt_path, "train_loss_means.pkl"), "wb") as f:
        pickle.dump(train_loss_means, f)
    with open(os.path.join(ckpt_path, "test_loss_means.pkl"), "wb") as f:
        pickle.dump(test_loss_means, f)
    with open(os.path.join(ckpt_path, "aucs.pkl"), "wb") as f:
        pickle.dump(aucs, f)


if __name__ == "__main__":
    main()
