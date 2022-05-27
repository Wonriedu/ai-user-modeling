import os

import numpy as np
import torch

from torch.nn import Module, Embedding, Parameter, GRU, Sequential, Linear, \
    ReLU, Dropout
from torch.nn.functional import one_hot


class UserModel(Module):
    def __init__(self, num_c, num_d, dim_v):
        super().__init__()

        self.num_c = num_c
        self.num_d = num_d

        self.dim_v = dim_v

        self.X = Embedding(self.num_c * self.num_d, self.dim_v)

        self.v_r = Parameter(torch.Tensor(self.dim_v))
        self.v_beta = Parameter(torch.Tensor(self.dim_v))

        self.gru = GRU(self.dim_v * 2, self.dim_v, batch_first=True)
        self.linear_1 = Sequential(
            Linear(self.dim_v, self.dim_v),
            ReLU(),
            Dropout(),
            Linear(self.dim_v, self.num_d),
            Dropout(),
        )

        self.linear_2 = Sequential(
            Linear(self.dim_v * 3, self.dim_v),
            ReLU(),
            Dropout(),
            Linear(self.dim_v, self.num_d),
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
                C_0: [batch_size, num_c, 1]

            Returns:
                alpha_seq: [batch_size, seq_len]
                h_seq: [batch_size, seq_len, dim_v]
                C_seq: [batch_size, seq_len, num_c, 1]
        '''
        batch_size = c_seq.shape[0]

        # v_x_seq, v_r_seq: [batch_size, seq_len, dim_v]
        x_seq = c_seq + self.num_c * d_seq
        v_x_seq = self.X(x_seq)
        v_r_seq = r_seq.unsqueeze(-1) * self.v_r

        # h_seq: [batch_size, seq_len, dim_v]
        if h_0:
            h_seq, _ = self.gru(torch.cat([v_x_seq, v_r_seq], dim=-1), h_0)
        else:
            h_seq, _ = self.gru(torch.cat([v_x_seq, v_r_seq], dim=-1))

        # alpha_seq: [batch_size, seq_len, num_d]
        alpha_seq = self.linear_1(h_seq)

        # C: [batch_size, num_c, num_d]
        if C_0:
            C = torch.clone(C_0)
        else:
            C = torch.zeros([batch_size, self.num_c, self.num_d])
        C_seq = []

        # c_one_hot_seq: [batch_size, seq_len, num_c, 1]
        # d_one_hot_seq: [batch_size, seq_len, 1, num_d]
        c_one_hot_seq = one_hot(c_seq, self.num_c).unsqueeze(-1).float()
        d_one_hot_seq = one_hot(d_seq, self.num_d).unsqueeze(-2).float()

        # x_one_hot_seq: [batch_size, seq_len, num_c, num_d]
        x_one_hot_seq = c_one_hot_seq * d_one_hot_seq

        for c_one_hot, x_one_hot, v_x, v_r in zip(
            c_one_hot_seq.permute(1, 0, 2, 3),
            x_one_hot_seq.permute(1, 0, 2, 3),
            v_x_seq.permute(1, 0, 2),
            v_r_seq.permute(1, 0, 2)
        ):
            # c_one_hot: [batch_size, num_c, 1]
            # x_one_hot: [batch_size, num_c, num_d]
            # v_x, v_r: [batch_size, dim_v]

            # beta_tilde: [batch_size]
            beta_tilde = (x_one_hot * C).sum(-1).sum(-1)

            # v_beta_tilde: [batch_size, dim_v]
            v_beta_tilde = beta_tilde.unsqueeze(-1) * self.v_beta
            # if batch_size == 1:
            #     v_beta_tilde = v_beta_tilde.unsqueeze(0)

            # new_c: [batch_size, num_d]
            new_c = self.linear_2(torch.cat([v_beta_tilde, v_x, v_r], dim=-1))

            C = C * (1 - c_one_hot) + \
                new_c.unsqueeze(1) * c_one_hot

            C_seq.append(C)

        # C_seq: [batch_size, seq_len, num_c, num_d]
        C_seq = torch.stack(C_seq, dim=1)

        return alpha_seq, h_seq, C_seq

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
                # m_seq: [batch_size, seq_len]

                self.train()

                alpha_seq, h_seq, C_seq = \
                    self(c_seq, d_seq, r_seq)

                # cshft_one_hot_seq: [batch_size, seq_len, num_c, 1]
                # dshft_one_hot_seq: [batch_size, seq_len, 1, num_d]
                cshft_one_hot_seq = one_hot(cshft_seq, self.num_c)\
                    .unsqueeze(-1).float()
                dshft_one_hot_seq = one_hot(dshft_seq, self.num_d)\
                    .unsqueeze(-2).float()

                # xshft_one_hot_seq: [batch_size, seq_len, num_c, num_d]
                xshft_one_hot_seq = cshft_one_hot_seq * dshft_one_hot_seq

                # alpha_seq: [batch_size, seq_len]
                alpha_seq = (alpha_seq * dshft_one_hot_seq.squeeze())\
                    .sum(-1)

                # beta_shft_seq: [batch_size, seq_len]
                beta_shft_seq = (C_seq * xshft_one_hot_seq)\
                    .sum(-1).sum(-1)

                # rshft_hat_seq: [batch_size, seq_len]
                rshft_hat_seq = torch.sigmoid(
                    alpha_seq +
                    beta_shft_seq
                )

                opt.zero_grad()
                loss = (rshft_hat_seq - rshft_seq) ** 2
                loss = torch.masked_select(loss, m_seq).mean()
                loss.backward()
                opt.step()

                train_loss_mean.append(loss.detach().cpu().numpy())

            with torch.no_grad():
                for data in test_loader:
                    c_seq, d_seq, r_seq, \
                        cshft_seq, dshft_seq, rshft_seq, m_seq = data

                    # rshft_seq: [batch_size, seq_len]
                    # m_seq: [batch_size, seq_len]

                    self.eval()

                    alpha_seq, h_seq, C_seq = \
                        self(c_seq, d_seq, r_seq)

                    # cshft_one_hot_seq: [batch_size, seq_len, num_c, 1]
                    # dshft_one_hot_seq: [batch_size, seq_len, 1, num_d]
                    cshft_one_hot_seq = one_hot(cshft_seq, self.num_c)\
                        .unsqueeze(-1).float()
                    dshft_one_hot_seq = one_hot(dshft_seq, self.num_d)\
                        .unsqueeze(-2).float()

                    # xshft_one_hot_seq: [batch_size, seq_len, num_c, num_d]
                    xshft_one_hot_seq = cshft_one_hot_seq * dshft_one_hot_seq

                    # alpha_seq: [batch_size, seq_len]
                    alpha_seq = (alpha_seq * dshft_one_hot_seq.squeeze())\
                        .sum(-1)

                    # beta_shft_seq: [batch_size, seq_len]
                    beta_shft_seq = (C_seq * xshft_one_hot_seq)\
                        .sum(-1).sum(-1)

                    # rshft_hat_seq: [batch_size, seq_len]
                    rshft_hat_seq = torch.sigmoid(
                        alpha_seq +
                        beta_shft_seq
                    )

                    train_loss_mean = np.mean(train_loss_mean)
                    test_loss_mean = \
                        (rshft_hat_seq - rshft_seq) ** 2
                    test_loss_mean = torch.masked_select(loss, m_seq)\
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
