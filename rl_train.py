import os
import json
import pickle

import torch

from rl_models.pg import PolicyGradient
from rl_models.ac import ActorCritic
from rl_models.trpo import TRPO
from rl_models.gae import GAE
from rl_models.ppo import PPO

from data_loaders.pulja_data_loader import PuljaDataLoader
from models._20220615_00 import UserModel
from models.user_simulator import UserSimulator


def main():
    ckpt_path = "ckpts"
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    ckpt_path = os.path.join(ckpt_path, "rl")
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    model_name = "trpo"
    if model_name not in ["pg", "ac", "trpo", "gae", "ppo"]:
        print("The model name is wrong!")
        return

    ckpt_path = os.path.join(ckpt_path, model_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    with open(os.path.join("rl_models", "config.json")) as f:
        config = json.load(f)[model_name]

    with open(os.path.join(ckpt_path, "model_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    seq_len = 100

    dim_v = 20

    dataset = PuljaDataLoader(seq_len)

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = UserModel(dataset.num_c1, dataset.num_c2, dataset.num_d, dim_v)\
        .to(device)

    env = UserSimulator(model)
    env.reset()

    discrete = True
    state_dim = env.state_dim
    action_dim = env.action_dim

    if model_name == "pg":
        model = PolicyGradient(
            state_dim, action_dim, discrete, **config
        ).to(device)
    elif model_name == "ac":
        model = ActorCritic(
            state_dim, action_dim, discrete, **config
        ).to(device)
    elif model_name == "trpo":
        model = TRPO(
            state_dim, action_dim, discrete, **config
        ).to(device)
    elif model_name == "gae":
        model = GAE(
            state_dim, action_dim, discrete, **config
        ).to(device)
    elif model_name == "ppo":
        model = PPO(
            state_dim, action_dim, discrete, **config
        ).to(device)

    results = model.train(env)

    env.close()

    with open(os.path.join(ckpt_path, "results.pkl"), "wb") as f:
        pickle.dump(results, f)

    if hasattr(model, "pi"):
        torch.save(
            model.pi.state_dict(), os.path.join(ckpt_path, "policy.ckpt")
        )
    if hasattr(model, "v"):
        torch.save(
            model.v.state_dict(), os.path.join(ckpt_path, "value.ckpt")
        )


if __name__ == "__main__":
    main()
