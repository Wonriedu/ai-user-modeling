import os

import numpy as np
import torch

from torch.nn import Module, Embedding, Parameter, GRU, Sequential, Linear, \
    ReLU, Dropout
from torch.nn.functional import one_hot, binary_cross_entropy_with_logits
from sklearn import metrics


class UserModel(Module):
    def __init__(self, num_c1, num_c2, num_c3, num_d, dim_v):
        super().__init__()

        self.num_c1 = num_c1
        self.num_c2 = num_c2
        self.num_c3 = num_c3
        self.num_d = num_d

        self.dim_v = dim_v

        self.v_c3 = Parameter(torch.Tensor(self.dim_v))

        self.D = Embedding(self.num_d, 1)

        self.v_d = Parameter(torch.Tensor(self.dim_v))

        self.R = Embedding(2, self.dim_v)

        self.init_alpha = torch.tensor([0])

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
        self, c3_seq, d_seq, r_seq,
        h_0=None, C3_0=None
    ):
        '''
            Args:
                c3_seq: [batch_size, seq_len]
                d_seq: [batch_size, seq_len]
                r_seq: [batch_size, seq_len]
                h_0: [batch_size, dim_v]
                C3_0: [batch_size, num_c3]

            Returns:
                alpha_seq: [batch_size, seq_len]
                h_seq: [batch_size, seq_len, dim_v]
                C3_seq: [batch_size, seq_len, num_c3]
        '''
        batch_size = c3_seq.shape[0]

        # gamma_seq: [batch_size, seq_len, 1]
        # v_d_seq, v_r_seq: [batch_size, seq_len, dim_v]
        gamma_seq = self.D(d_seq)
        v_d_seq = gamma_seq * self.v_d
        v_r_seq = self.R(r_seq)

        # h_seq: [batch_size, seq_len, dim_v]
        if h_0 is not None:
            h_seq, _ = self.gru(
                torch.cat([v_d_seq, v_r_seq], dim=-1),
                h_0.unsqueeze(0)
            )
        else:
            h_seq, _ = self.gru(torch.cat([v_d_seq, v_r_seq], dim=-1))

        # alpha: [batch_size]
        alpha = self.init_alpha.repeat([batch_size])
        alpha_seq = []

        # C3: [batch_size, num_c4]
        if C3_0 is not None:
            C3 = torch.clone(C3_0)
        else:
            C3 = torch.zeros([batch_size, self.num_c3])
        C3_seq = []

        for h, gamma, r, c3, v_d, v_r in zip(
            h_seq.permute(1, 0, 2),
            gamma_seq.permute(1, 0, 2),
            r_seq.permute(1, 0,),
            c3_seq.permute(1, 0),
            v_d_seq.permute(1, 0, 2),
            v_r_seq.permute(1, 0, 2)
        ):
            # h: [batch_size, dim_v]
            # gamma: [batch_size, 1]
            # r: [batch_size]
            # c3: [batch_size]
            # v_d, v_r: [batch_size, dim_v]

            # alpha_new: [batch_size]
            gamma = gamma.reshape([batch_size])
            alpha_new = self.linear_1(h).reshape([batch_size])
            alpha = \
                (r == 1) * (
                    (alpha - gamma >= 0) * alpha_new +
                    (alpha - gamma < 0) * alpha
                ) + \
                (r == 0) * (
                    (alpha - gamma >= 0) * alpha +
                    (alpha - gamma < 0) * alpha_new
                )

            alpha_seq.append(alpha)

            # beta3_tilde: [batch_size, 1]
            beta3_tilde = torch.gather(C3, dim=-1, index=c3.unsqueeze(-1))

            # v_c3: [batch_size, dim_v]
            v_c3 = (beta3_tilde * self.v_c3)

            v_c3 = torch.reshape(v_c3, [batch_size, self.dim_v])

            # new_c: [batch_size]
            new_c3 = self.linear_2(
                torch.cat([v_c3, v_d, v_r], dim=-1)
            ).reshape([batch_size])

            # c3_one_hot: [batch_size, num_c3]
            c3_one_hot = one_hot(c3, self.num_c3)

            C3 = C3 * (1 - c3_one_hot) + \
                new_c3.unsqueeze(-1) * c3_one_hot

            C3_seq.append(C3)

        # C3_seq: [batch_size, seq_len, num_c3]
        C3_seq = torch.stack(C3_seq, dim=1)

        # alpha_seq: [batch_size, seq_len]
        alpha_seq = torch.stack(alpha_seq, dim=1)

        return alpha_seq, h_seq, C3_seq

    def get_logits(
        self,
        c3_seq, d_seq, r_seq,
        c3shft_seq, dshft_seq, rshft_seq,
        m_seq
    ):
        batch_size = c3_seq.shape[0]
        seq_len = c3_seq.shape[1]

        # rshft_seq: [batch_size, seq_len]
        # m_seq: [batch_size, seq_len]

        alpha_seq, h_seq, C3_seq = \
            self(c3_seq, d_seq, r_seq)

        # alpha_seq: [batch_size, seq_len]

        # beta3_shft_seq: [batch_size, seq_len]
        beta3_shft_seq = torch.gather(
            C3_seq, dim=-1, index=c3shft_seq.unsqueeze(-1)
        ).reshape([batch_size, seq_len])

        # gamma_shft_seq: [batch_size, seq_len]
        gamma_shft_seq = self.D(dshft_seq).squeeze()

        c3_logits = \
            alpha_seq + \
            beta3_shft_seq - \
            gamma_shft_seq

        return c3_logits

    def train_model(
        self, train_loader, test_loader, num_epochs, opt, ckpt_path
    ):
        train_loss_means = []
        test_loss_means = []
        aucs = []

        max_auc = 0

        for i in range(1, num_epochs + 1):
            train_loss_mean = []

            for data in train_loader:
                c3_seq, d_seq, r_seq, \
                    c3shft_seq, dshft_seq, rshft_seq, \
                    m_seq = data

                self.train()
                c3_logits = self.get_logits(*data)

                opt.zero_grad()
                loss = \
                    binary_cross_entropy_with_logits(
                        torch.masked_select(
                            c3_logits, m_seq
                        ),
                        torch.masked_select(rshft_seq.float(), m_seq)
                    )
                loss.backward()
                opt.step()

                train_loss_mean.append(loss.detach().cpu().numpy())

            with torch.no_grad():
                for data in test_loader:
                    c3_seq, d_seq, r_seq, \
                        c3shft_seq, dshft_seq, rshft_seq, \
                        m_seq = data

                    self.eval()
                    c3_logits = self.get_logits(*data)

                    # rshft_hat_seq: [batch_size, seq_len]
                    rshft_hat_seq = torch.sigmoid(c3_logits)

                    train_loss_mean = np.mean(train_loss_mean)
                    test_loss_mean = \
                        binary_cross_entropy_with_logits(
                            torch.masked_select(
                                c3_logits, m_seq
                            ),
                            torch.masked_select(rshft_seq.float(), m_seq)
                        ).detach().cpu().numpy()
                    auc = metrics.roc_auc_score(
                        y_true=torch.masked_select(rshft_seq, m_seq)
                        .detach().cpu().numpy(),
                        y_score=torch.masked_select(rshft_hat_seq, m_seq)
                        .detach().cpu().numpy(),
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
