import os

import numpy as np
import torch

from torch.nn import Module, Embedding, Parameter, GRU, Sequential, Linear, \
    ReLU, Dropout
from torch.nn.functional import one_hot, binary_cross_entropy_with_logits
from sklearn import metrics


class UserModel(Module):
    def __init__(self, num_c1, num_c2, num_d, dim_v):
        super().__init__()

        self.num_c1 = num_c1
        self.num_c2 = num_c2
        self.num_d = num_d

        self.dim_v = dim_v

        self.D = Embedding(self.num_d, 1)

        self.v_d = Parameter(torch.Tensor(self.dim_v))

        self.v_r = Parameter(torch.Tensor(self.dim_v))
        self.v_c = Parameter(torch.Tensor(self.dim_v))

        self.gru = GRU(self.dim_v * 2, self.dim_v, batch_first=True)
        self.linear_1 = Sequential(
            Linear(self.dim_v, self.dim_v),
            ReLU(),
            Dropout(),
            Linear(self.dim_v, 1),
            Dropout(),
        )

        self.linear_2 = Sequential(
            Linear(self.dim_v * 3, self.dim_v),
            ReLU(),
            Dropout(),
            Linear(self.dim_v, 1),
            Dropout(),
        )

    def forward(
        self, c_seq, d_seq, r_seq, h_0=None, C_0=None,
    ):
        '''
            Args:
                c_seq: [batch_size, seq_len]
                d_seq: [batch_size, seq_len]
                r_seq: [batch_size, seq_len]
                h_0: [batch_size, dim_v]
                C_0: [batch_size, num_c2, 1]

            Returns:
                alpha_seq: [batch_size, seq_len]
                h_seq: [batch_size, seq_len, dim_v]
                C_seq: [batch_size, seq_len, num_c2, 1]
        '''
        batch_size = c_seq.shape[0]
        seq_len = c_seq.shape[1]

        # gamma_seq: [batch_size, seq_len, 1]
        # v_d_seq, v_r_seq: [batch_size, seq_len, dim_v]
        gamma_seq = self.D(d_seq)
        v_d_seq = gamma_seq * self.v_d
        v_r_seq = r_seq.unsqueeze(-1) * self.v_r

        # h_seq: [batch_size, seq_len, dim_v]
        if h_0 is not None:
            h_seq, _ = self.gru(
                torch.cat([v_d_seq, v_r_seq], dim=-1),
                h_0.unsqueeze(0)
            )
        else:
            h_seq, _ = self.gru(torch.cat([v_d_seq, v_r_seq], dim=-1))

        # alpha_seq: [batch_size, seq_len]
        alpha_seq = self.linear_1(h_seq).squeeze()
        alpha_seq = torch.reshape(alpha_seq, [batch_size, seq_len])

        # C: [batch_size, num_c2, 1]
        if C_0 is not None:
            C = torch.clone(C_0)
        else:
            C = torch.zeros([batch_size, self.num_c2, 1])
        C_seq = []

        # c_one_hot_seq: [batch_size, seq_len, num_c2]
        c_one_hot_seq = one_hot(c_seq, self.num_c2).float()

        for c_one_hot, v_d, v_r in zip(
            c_one_hot_seq.permute(1, 0, 2),
            v_d_seq.permute(1, 0, 2),
            v_r_seq.permute(1, 0, 2)
        ):
            # c_one_hot: [batch_size, num_c2]
            # v_d, v_r: [batch_size, dim_v]

            # beta_tilde: [batch_size, 1, 1]
            beta_tilde = torch.bmm(c_one_hot.unsqueeze(1), C)

            # v_c: [batch_size, dim_v]
            v_c = (beta_tilde * self.v_c).squeeze()
            v_c = torch.reshape(v_c, [batch_size, self.dim_v])

            # new_c: [batch_size, 1]
            new_c = self.linear_2(torch.cat([v_c, v_d, v_r], dim=-1))

            C = C * (1 - c_one_hot.unsqueeze(-1)) + \
                new_c.unsqueeze(1) * c_one_hot.unsqueeze(-1)

            C_seq.append(C)

        # C_seq: [batch_size, seq_len, num_c2, 1]
        C_seq = torch.stack(C_seq, dim=1)

        return alpha_seq, h_seq, C_seq

    def train_model(
        self, train_loader, test_loader, num_epochs, opt, ckpt_path
    ):
        train_loss_means = []
        test_loss_means = []
        aucs = []

        # min_test_loss_mean = np.inf
        max_auc = 0

        for i in range(1, num_epochs + 1):
            train_loss_mean = []

            for data in train_loader:
                c1_seq, c2_seq, d_seq, r_seq, \
                    c1shft_seq, c2shft_seq, dshft_seq, rshft_seq, m_seq = data

                batch_size = c2_seq.shape[0]
                seq_len = c2_seq.shape[1]

                # rshft_seq: [batch_size, seq_len]
                # m_seq: [batch_size, seq_len]

                self.train()

                alpha_seq, h_seq, C_seq = \
                    self(c2_seq, d_seq, r_seq)

                # alpha_seq: [batch_size, seq_len]

                # c2shft_one_hot_seq: [batch_size, seq_len, 1, num_c2]
                c2shft_one_hot_seq = one_hot(c2shft_seq, self.num_c2).float()
                c2shft_one_hot_seq = torch.reshape(
                    c2shft_one_hot_seq,
                    shape=[
                        -1,
                        c2shft_one_hot_seq.shape[1],
                        c2shft_one_hot_seq.shape[2]
                    ]
                ).unsqueeze(-2)

                # beta_shft_seq: [batch_size, seq_len]
                beta_shft_seq = torch.bmm(
                    torch.reshape(
                        c2shft_one_hot_seq,
                        shape=[
                            -1,
                            c2shft_one_hot_seq.shape[2],
                            c2shft_one_hot_seq.shape[3]
                        ]
                    ),
                    torch.reshape(
                        C_seq, shape=[-1, C_seq.shape[2], C_seq.shape[3]]
                    )
                )
                beta_shft_seq = torch.reshape(
                    beta_shft_seq, shape=[batch_size, seq_len]
                )

                # gamma_shft_seq: [batch_size, seq_len]
                gamma_shft_seq = self.D(dshft_seq).squeeze()
                gamma_shft_seq = torch.reshape(
                    gamma_shft_seq, [batch_size, seq_len]
                )

                opt.zero_grad()
                loss = binary_cross_entropy_with_logits(
                    torch.masked_select(
                        alpha_seq +
                        beta_shft_seq -
                        gamma_shft_seq, m_seq
                    ),
                    torch.masked_select(rshft_seq, m_seq)
                )
                loss.backward()
                opt.step()

                train_loss_mean.append(loss.detach().cpu().numpy())

            with torch.no_grad():
                for data in test_loader:
                    c1_seq, c2_seq, d_seq, r_seq, \
                        c1shft_seq, c2shft_seq, dshft_seq, rshft_seq, m_seq = \
                        data

                    batch_size = c2_seq.shape[0]
                    seq_len = c2_seq.shape[1]

                    self.eval()

                    alpha_seq, h_seq, C_seq = \
                        self(c2_seq, d_seq, r_seq)

                    # alpha_seq: [batch_size, seq_len]

                    # c2shft_one_hot_seq: [batch_size, seq_len, 1, num_c2]
                    c2shft_one_hot_seq = one_hot(
                        c2shft_seq, self.num_c2
                    ).float()
                    c2shft_one_hot_seq = torch.reshape(
                        c2shft_one_hot_seq,
                        shape=[
                            -1,
                            c2shft_one_hot_seq.shape[1],
                            c2shft_one_hot_seq.shape[2]
                        ]
                    ).unsqueeze(-2)

                    # beta_shft_seq: [batch_size, seq_len]
                    beta_shft_seq = torch.bmm(
                        torch.reshape(
                            c2shft_one_hot_seq,
                            shape=[
                                -1,
                                c2shft_one_hot_seq.shape[2],
                                c2shft_one_hot_seq.shape[3]
                            ]
                        ),
                        torch.reshape(
                            C_seq, shape=[-1, C_seq.shape[2], C_seq.shape[3]]
                        )
                    )
                    beta_shft_seq = torch.reshape(
                        beta_shft_seq, shape=[batch_size, seq_len]
                    )

                    # gamma_shft_seq: [batch_size, seq_len]
                    gamma_shft_seq = self.D(dshft_seq).squeeze()
                    gamma_shft_seq = torch.reshape(
                        gamma_shft_seq, [batch_size, seq_len]
                    )

                    # rshft_hat_seq: [batch_size, seq_len]
                    rshft_hat_seq = torch.sigmoid(
                        alpha_seq +
                        beta_shft_seq -
                        gamma_shft_seq
                    )

                    train_loss_mean = np.mean(train_loss_mean)
                    test_loss_mean = binary_cross_entropy_with_logits(
                        torch.masked_select(
                            alpha_seq +
                            beta_shft_seq -
                            gamma_shft_seq, m_seq
                        ),
                        torch.masked_select(rshft_seq, m_seq)
                    ).detach().cpu().numpy()
                    auc = metrics.roc_auc_score(
                        y_true=rshft_seq.detach().cpu().numpy(),
                        y_score=rshft_hat_seq.detach().cpu().numpy(),
                    )

                    print(
                        "Epochs: {},  Train Loss Mean: {},  AUC: {}"
                        .format(i, train_loss_mean, auc)
                    )

                    if auc > max_auc:
                        torch.save(
                            self.state_dict(),
                            os.path.join(
                                ckpt_path, "model_max.ckpt"
                            )
                        )
                        max_auc = auc

                    train_loss_means.append(train_loss_mean)
                    test_loss_means.append(test_loss_mean)
                    aucs.append(auc)

        torch.save(
            self.state_dict(),
            os.path.join(
                ckpt_path, "model_fin.ckpt"
            )
        )

        return train_loss_means, test_loss_means, aucs
