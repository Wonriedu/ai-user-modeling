import os

import numpy as np
import torch

from torch.nn import Module, Embedding, Parameter, GRU, Sequential, Linear, \
    ReLU, Dropout
from torch.nn.functional import one_hot


class UserModel(Module):
    def __init__(self, num_c, num_d, dim_v, window_size=10):
        super().__init__()

        self.num_c = num_c
        self.num_d = num_d

        self.dim_v = dim_v

        self.window_size = window_size

        self.D1 = Embedding(self.num_d, 1)
        self.D2 = Embedding(self.num_d, self.dim_v)

        self.C = Embedding(self.num_c, self.dim_v)

        self.v_r = Parameter(torch.Tensor(self.dim_v))

        self.gru = GRU(self.dim_v * 3, self.dim_v, batch_first=True)
        self.linear = Sequential(
            Linear(self.dim_v, self.dim_v),
            ReLU(),
            Dropout(),
            Linear(self.dim_v, self.num_c),
            Dropout(),
        )

    def forward(
        self, c_seq, d_seq, r_seq, h_0=None,
    ):
        '''
            Args:
                c_seq: [batch_size, seq_len]
                d_seq: [batch_size, seq_len]
                r_seq: [batch_size, seq_len]
                h_0: [batch_size, dim_v]

            Returns:
                alpha_seq: [batch_size, seq_len]
                h_seq: [batch_size, seq_len, dim_v]
        '''
        # v_c_seq, v_d_seq, v_r_seq: [batch_size, seq_len, dim_v]
        v_c_seq = self.C(c_seq)
        v_d_seq = self.D2(d_seq)
        v_r_seq = r_seq.unsqueeze(-1) * self.v_r

        # h_seq: [batch_size, seq_len, dim_v]
        if h_0:
            h_seq, _ = self.gru(
                torch.cat([v_c_seq, v_d_seq, v_r_seq], dim=-1), h_0
            )
        else:
            h_seq, _ = self.gru(
                torch.cat([v_c_seq, v_d_seq, v_r_seq], dim=-1)
            )

        # alpha_seq: [batch_size, seq_len, num_c]
        alpha_seq = self.linear(h_seq).squeeze()

        return alpha_seq, h_seq

    def train_model(
        self, train_loader, test_loader, num_epochs, opt, ckpt_path
    ):
        train_loss_means = []
        test_loss_means = []

        min_test_loss_mean = np.inf

        for i in range(1, num_epochs + 1):
            train_loss_mean = []

            for data in train_loader:
                c_seq, d_seq, r_seq, \
                    cshft_seq, dshft_seq, rshft_seq, m_seq = data

                # rshft_seq: [batch_size, seq_len]
                # rshft_seq_window: [batch_size * num_windows, window_size]
                rshft_seq_window = torch.cat(
                    [
                        rshft_seq[:, j:j + self.window_size]
                        for j in range(
                            rshft_seq.shape[1] - self.window_size + 1
                        )
                    ],
                    dim=0
                )
                # m_seq: [batch_size, seq_len]
                # m_seq_window: [batch_size * num_windows, window_size]
                m_seq_window = torch.cat(
                    [
                        m_seq[:, j:j + self.window_size]
                        for j in range(
                            m_seq.shape[1] - self.window_size + 1
                        )
                    ],
                    dim=0
                )

                self.train()

                alpha_seq, h_seq = \
                    self(c_seq, d_seq, r_seq)

                # cshft_one_hot_seq: [batch_size, seq_len, num_c]
                # cshft_one_hot_seq_window:
                # [batch_size * num_windows, window_size, num_c]
                cshft_one_hot_seq = one_hot(cshft_seq, self.num_c).float()
                cshft_one_hot_seq_window = torch.cat(
                    [
                        cshft_one_hot_seq[:, j:j + self.window_size]
                        for j in range(
                            cshft_one_hot_seq.shape[1] - self.window_size + 1
                        )
                    ],
                    dim=0
                )

                # alpha_seq: [batch_size, seq_len, num_c]
                # alpha_seq_window: [batch_size * num_windows, 1, num_c]
                # alpha_shft_seq_window:
                # [batch_size * num_windows, window_size]
                alpha_seq_window = torch.cat(
                    [
                        alpha_seq[:, j:j + 1]
                        for j in range(
                            alpha_seq.shape[1] - self.window_size + 1
                        )
                    ],
                    dim=0
                )
                alpha_shft_seq_window = \
                    (alpha_seq_window * cshft_one_hot_seq_window).sum(-1)

                # gamma_shft_seq: [batch_size, seq_len]
                # gamma_shft_seq_window:
                # [batch_size * num_windows, window_size]
                gamma_shft_seq = self.D1(dshft_seq).squeeze()
                gamma_shft_seq_window = torch.cat(
                    [
                        gamma_shft_seq[:, j:j + self.window_size]
                        for j in range(
                            gamma_shft_seq.shape[1] - self.window_size + 1
                        )
                    ],
                    dim=0
                )

                # rshft_hat_seq_window: [batch_size * num_windows, window_size]
                rshft_hat_seq_window = torch.sigmoid(
                    alpha_shft_seq_window +
                    gamma_shft_seq_window
                )

                opt.zero_grad()
                loss = (rshft_hat_seq_window - rshft_seq_window) ** 2
                loss = torch.masked_select(loss, m_seq_window).mean()
                loss.backward()
                opt.step()

                train_loss_mean.append(loss.detach().cpu().numpy())

            with torch.no_grad():
                for data in test_loader:
                    c_seq, d_seq, r_seq, \
                        cshft_seq, dshft_seq, rshft_seq, m_seq = data

                    # rshft_seq: [batch_size, seq_len]
                    # rshft_seq_window: [batch_size * num_windows, window_size]
                    rshft_seq_window = torch.cat(
                        [
                            rshft_seq[:, j:j + self.window_size]
                            for j in range(
                                rshft_seq.shape[1] - self.window_size + 1
                            )
                        ],
                        dim=0
                    )
                    # m_seq: [batch_size, seq_len]
                    # m_seq_window: [batch_size * num_windows, window_size]
                    m_seq_window = torch.cat(
                        [
                            m_seq[:, j:j + self.window_size]
                            for j in range(
                                m_seq.shape[1] - self.window_size + 1
                            )
                        ],
                        dim=0
                    )

                    self.eval()

                    alpha_seq, h_seq = \
                        self(c_seq, d_seq, r_seq)

                    # cshft_one_hot_seq: [batch_size, seq_len, num_c]
                    # cshft_one_hot_seq_window:
                    # [batch_size * num_windows, window_size, num_c]
                    cshft_one_hot_seq = one_hot(cshft_seq, self.num_c).float()
                    cshft_one_hot_seq_window = torch.cat(
                        [
                            cshft_one_hot_seq[:, j:j + self.window_size]
                            for j in range(
                                cshft_one_hot_seq.shape[1] -
                                self.window_size +
                                1
                            )
                        ],
                        dim=0
                    )

                    # alpha_seq: [batch_size, seq_len, num_c]
                    # alpha_seq_window: [batch_size * num_windows, 1, num_c]
                    # alpha_shft_seq_window:
                    # [batch_size * num_windows, window_size]
                    alpha_seq_window = torch.cat(
                        [
                            alpha_seq[:, j:j + 1]
                            for j in range(
                                alpha_seq.shape[1] - self.window_size + 1
                            )
                        ],
                        dim=0
                    )
                    alpha_shft_seq_window = \
                        (alpha_seq_window * cshft_one_hot_seq_window).sum(-1)

                    # gamma_shft_seq: [batch_size, seq_len]
                    # gamma_shft_seq_window:
                    # [batch_size * num_windows, window_size]
                    gamma_shft_seq = self.D1(dshft_seq).squeeze()
                    gamma_shft_seq_window = torch.cat(
                        [
                            gamma_shft_seq[:, j:j + self.window_size]
                            for j in range(
                                gamma_shft_seq.shape[1] - self.window_size + 1
                            )
                        ],
                        dim=0
                    )

                    # rshft_hat_seq_window:
                    # [batch_size * num_windows, window_size]
                    rshft_hat_seq_window = torch.sigmoid(
                        alpha_shft_seq_window +
                        gamma_shft_seq_window
                    )

                    train_loss_mean = np.mean(train_loss_mean)
                    test_loss_mean = \
                        (rshft_hat_seq_window - rshft_seq_window) ** 2
                    test_loss_mean = torch.masked_select(loss, m_seq_window)\
                        .mean().detach().cpu().numpy()

                    print(
                        "Epochs: {},  Train Loss Mean: {},  Test Loss Mean: {}"
                        .format(i, train_loss_mean, test_loss_mean)
                    )

                    if test_loss_mean < min_test_loss_mean:
                        torch.save(
                            self.state_dict(),
                            os.path.join(
                                ckpt_path, "model.ckpt"
                            )
                        )
                        min_test_loss_mean = test_loss_mean

                    train_loss_means.append(train_loss_mean)
                    test_loss_means.append(test_loss_mean)

        return train_loss_means, test_loss_means
