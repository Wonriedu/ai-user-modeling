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

        # 3등급 커리큘럼 예상 풀이시간 기준으로 작성
        # self.difficulty2duration = {
        #     1.0: 0.75,
        #     2.0: 0.75,
        #     3.0: 1.0,
        #     4.0: 1.5,
        #     5.0: 2.5,
        #     6.0: 7,
        #     7.0: 10,
        # }
        # 임의 기준
        # self.difficulty2duration = {
        #     1.0: 0.75,
        #     2.0: 0.75,
        #     3.0: 1.0,
        #     4.0: 1.0,
        #     5.0: 1.5,
        #     6.0: 2.0,
        #     7.0: 4.0,
        # }
        self.difficulty2duration = {
            1.0: 0.75,
            2.0: 0.75,
            3.0: 1.0,
            4.0: 1.0,
            5.0: 1.0,
            6.0: 1.5,
            7.0: 1.5,
        }
        self.difficulty2duration = {
            k: v * 60 for k, v in self.difficulty2duration.items()
        }

        self.df = self.tb_user_curriculum_lecture_unit_solve\
            .merge(self.tb_problems)\
            .sort_values("seq")\
            .reset_index(drop=True)
        self.df = self.df[self.df.notnull()["category3"]]\
            .reset_index(drop=True)

        self.u_list = np.unique(self.df["userSeq"].values)
        self.u2idx = {u: i for i, u in enumerate(self.u_list)}

        self.num_u = self.u_list.shape[0]

        self.c_list = np.unique(self.df["category2"].values)
        self.c2idx = {c: i for i, c in enumerate(self.c_list)}

        self.num_c = self.c_list.shape[0]

        self.d_list = np.unique(self.df["difficulty_cd"].values)
        self.d2idx = {d: i for i, d in enumerate(self.d_list)}

        self.num_d = self.d_list.shape[0]

        if os.path.exists(
            os.path.join(self.dataset_dir, "dataset.pkl")
        ):
            with open(
                os.path.join(self.dataset_dir, "dataset.pkl"), "rb"
            ) as f:
                self.c_seqs, self.d_seqs, self.r_seqs = pickle.load(f)
        else:
            self.preprocess()

            with open(
                os.path.join(self.dataset_dir, "dataset.pkl"), "wb"
            ) as f:
                pickle.dump((self.c_seqs, self.d_seqs, self.r_seqs), f)

        self.len = len(self.r_seqs)

    def __getitem__(self, idx):
        return self.c_seqs[idx], self.d_seqs[idx], self.r_seqs[idx]

    def __len__(self):
        return self.len

    def preprocess(self):
        self.c_seqs = []
        self.d_seqs = []
        self.r_seqs = []

        for u in self.u_list:
            df_u = self.df[self.df["userSeq"] == u]

            c_seq = np.array(
                [self.c2idx[c] for c in df_u["category2"].values]
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

            self.c_seqs.append(c_seq)
            self.d_seqs.append(d_seq)
            self.r_seqs.append(r_seq)

        if self.seq_len:
            self.c_seqs, self.d_seqs, self.r_seqs = match_seq_len(
                self.c_seqs, self.d_seqs, self.r_seqs, self.seq_len
            )

    def get_response(self, r_seq, TR_seq):
        return r_seq * (TR_seq <= 1.0).astype(float)

    # def get_response(self, r_seq, TR_seq):
    #     result = []
    #     for r, TR in zip(r_seq, TR_seq):
    #         if r == 1:
    #             basis = 0.5
    #             TR = np.minimum(TR, 1.5)

    #             result.append(sigmoid(
    #                 basis *
    #                 (np.tan((np.pi / 2) - (TR * np.pi / 3)))
    #             ))
    #         else:
    #             basis = -0.5
    #             TR = np.maximum(TR, 0.65)

    #             result.append(sigmoid(
    #                 basis *
    #                 (np.tan((np.pi / 2) - ((1 / TR) * np.pi * 0.65 / 2)))
    #             ))

    #     return np.array(result)

    # def get_response(self, r, TR):
    #     return r * \
    #         (
    #             1.50 - np.minimum(
    #                 np.maximum(TR, 0.65), 1.50
    #             )
    #         ) / \
    #         (1.50 - 0.65)
