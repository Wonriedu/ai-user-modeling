import os
import pickle

import numpy as np
import pandas as pd

from torch.utils.data import Dataset

from models.utils import match_seq_len


DATASET_DIR = "datasets"
TABLE_DIR = "tables"


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class PuljaDataLoader(Dataset):
    def __init__(
        self, seq_len, dataset_dir=DATASET_DIR, table_dir=TABLE_DIR
    ) -> None:
        super().__init__()

        self.seq_len = seq_len
        self.dataset_dir = dataset_dir
        self.table_dir = table_dir

        self.tb_user_curriculum_lecture_unit_solve = pd.read_csv(
            os.path.join(
                self.table_dir, "tb_user_curriculum_lecture_unit_solve.csv"
            )
        )
        self.tb_problems = pd.read_csv(
            os.path.join(self.table_dir, "tb_problems.csv")
        )
        self.tb_curriculum_unit = pd.read_csv(
            os.path.join(self.table_dir, "tb_curriculum_unit.csv")
        )

        self.difficulty2duration = {
            1.0: 1.5,
            2.0: 2.0,
            3.0: 2.5,
            4.0: 2.5,
            5.0: 3.0,
            6.0: 5.0,
            7.0: 7.0,
        }
        self.difficulty2duration = {
            k: v * 60 for k, v in self.difficulty2duration.items()
        }

        self.df = self.tb_user_curriculum_lecture_unit_solve\
            .merge(self.tb_problems)\
            .sort_values("seq")\
            .reset_index(drop=True)
        self.df = self.df[self.df.notnull()["category1"]]\
            .reset_index(drop=True)
        self.df = self.df[self.df.notnull()["category2"]]\
            .reset_index(drop=True)

        self.u_list = np.unique(self.df["userSeq"].values)
        self.u2idx = {u: i for i, u in enumerate(self.u_list)}

        self.num_u = self.u_list.shape[0]

        self.c1_list = np.unique(self.df["category1"].values)
        self.c12idx = {c1: i for i, c1 in enumerate(self.c1_list)}

        self.num_c1 = self.c1_list.shape[0]

        self.c2_list = np.unique(self.df["category2"].values)
        self.c22idx = {c2: i for i, c2 in enumerate(self.c2_list)}

        self.num_c2 = self.c2_list.shape[0]

        self.d_list = np.unique(self.df["difficulty_cd"].values)
        self.d2idx = {d: i for i, d in enumerate(self.d_list)}

        self.num_d = self.d_list.shape[0]

        if os.path.exists(
            os.path.join(self.dataset_dir, "dataset.pkl")
        ):
            with open(
                os.path.join(self.dataset_dir, "dataset.pkl"), "rb"
            ) as f:
                self.c1_seqs, self.c2_seqs, self.d_seqs, self.r_seqs = \
                    pickle.load(f)
        else:
            self.preprocess()

            with open(
                os.path.join(self.dataset_dir, "dataset.pkl"), "wb"
            ) as f:
                pickle.dump(
                    (self.c1_seqs, self.c2_seqs, self.d_seqs, self.r_seqs), f
                )

        self.len = len(self.r_seqs)

    def __getitem__(self, idx):
        return self.c1_seqs[idx], self.c2_seqs[idx], self.d_seqs[idx], \
            self.r_seqs[idx]

    def __len__(self):
        return self.len

    def preprocess(self):
        self.c1_seqs = []
        self.c2_seqs = []
        self.d_seqs = []
        self.r_seqs = []

        for u in self.u_list:
            df_u = self.df[self.df["userSeq"] == u]

            c1_seq = np.array(
                [self.c12idx[c1] for c1 in df_u["category1"].values]
            )
            c2_seq = np.array(
                [self.c22idx[c2] for c2 in df_u["category2"].values]
            )
            d_seq = np.array(
                [self.d2idx[d] for d in df_u["difficulty_cd"].values]
            )

            T_seq = df_u["duration"].values
            T_hat_seq = np.array(
                [self.difficulty2duration[self.d_list[d]] for d in d_seq]
            )
            TR_seq = T_seq / T_hat_seq

            r_seq = (df_u["isCorrect"].values == "Y").astype(float)
            r_seq = self.get_response(r_seq, TR_seq)

            self.c1_seqs.append(c1_seq)
            self.c2_seqs.append(c2_seq)
            self.d_seqs.append(d_seq)
            self.r_seqs.append(r_seq)

        if self.seq_len:
            self.c1_seqs, self.c2_seqs, self.d_seqs, self.r_seqs = \
                match_seq_len(
                    self.c1_seqs,
                    self.c2_seqs,
                    self.d_seqs,
                    self.r_seqs,
                    self.seq_len
                )

    def get_response(self, r_seq, TR_seq):
        return r_seq * (TR_seq <= 1.0).astype(float)
