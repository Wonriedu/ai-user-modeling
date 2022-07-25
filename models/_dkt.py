import os

import numpy as np
import torch

from torch.nn import Module, Embedding, Parameter, GRU, Linear, Dropout
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

        self.X = Embedding(self.num_c3 * self.num_d, self.dim_v)

        self.v_r = Parameter(torch.Tensor(self.dim_v))

        self.gru = GRU(self.dim_v * 2, self.dim_v, batch_first=True)
        self.out_layer = Linear(self.dim_v, self.num_c3 * self.num_d)
        self.dropout_layer = Dropout()

    def forward(
        self, c3_seq, d_seq, r_seq, h_0=None
    ):
        '''
            Args:
                c3_seq: [batch_size, seq_len]
                d_seq: [batch_size, seq_len]
                r_seq: [batch_size, seq_len]
                h_0: [batch_size, dim_v]

            Returns:
                alpha_seq: [batch_size, seq_len]
                h_seq: [batch_size, seq_len, dim_v]
        '''
        batch_size = c3_seq.shape[0]
        seq_len = c3_seq.shape[1]

        # v_x_seq, v_r_seq: [batch_size, seq_len, dim_v]
        x_seq = c3_seq + self.num_c3 * d_seq
        v_x_seq = self.X(x_seq)
        v_r_seq = r_seq.unsqueeze(-1) * self.v_r

        # h_seq: [batch_size, seq_len, dim_v]
        if h_0:
            h_seq, _ = self.gru(
                torch.cat([v_x_seq, v_r_seq], dim=-1),
                h_0.unsqueeze(0)
            )
        else:
            h_seq, _ = self.gru(
                torch.cat([v_x_seq, v_r_seq], dim=-1)
            )

        # alpha_seq: [batch_size, seq_len, num_c3, num_d]
        alpha_seq = self.out_layer(h_seq).squeeze()
        alpha_seq = torch.reshape(
            alpha_seq, shape=[batch_size, seq_len, self.num_c2, self.num_d]
        )

        return alpha_seq, h_seq

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
                c3_seq, d_seq, r_seq, \
                    c3shft_seq, dshft_seq, rshft_seq, \
                    m_seq = data

                # rshft_seq: [batch_size, seq_len]
                # m_seq: [batch_size, seq_len]

                self.train()

                alpha_seq, h_seq = \
                    self(c3_seq, d_seq, r_seq)

                # alpha_seq: [batch_size, seq_len, num_c3, num_d]

                # cshft_one_hot_seq: [batch_size, seq_len, num_c3, 1]
                cshft_one_hot_seq = one_hot(c3shft_seq, self.num_c3)\
                    .unsqueeze(-1).float()

                # dshft_one_hot_seq: [batch_size, seq_len, 1, num_d]
                dshft_one_hot_seq = one_hot(dshft_seq, self.num_d)\
                    .unsqueeze(-2).float()

                # xshft_one_hot_seq:
                # [batch_size, seq_len, num_c3, num_d]
                xshft_one_hot_seq = \
                    cshft_one_hot_seq * dshft_one_hot_seq

                # alpha_shft_seq: [batch_size, seq_len]
                alpha_shft_seq = \
                    (alpha_seq * xshft_one_hot_seq)\
                    .sum(-1).sum(-1)

                opt.zero_grad()
                loss = binary_cross_entropy_with_logits(
                    torch.masked_select(alpha_shft_seq, m_seq),
                    torch.masked_select(rshft_seq, m_seq)
                )
                loss.backward()
                opt.step()

                train_loss_mean.append(loss.detach().cpu().numpy())

            with torch.no_grad():
                for data in test_loader:
                    c3_seq, d_seq, r_seq, \
                        c3shft_seq, dshft_seq, rshft_seq, \
                        m_seq = data

                    # rshft_seq: [batch_size, seq_len]
                    # m_seq: [batch_size, seq_len]

                    self.eval()

                    alpha_seq, h_seq = \
                        self(c3_seq, d_seq, r_seq)

                    # alpha_seq: [batch_size, seq_len, num_c3, num_d]

                    # cshft_one_hot_seq: [batch_size, seq_len, num_c3, 1]
                    cshft_one_hot_seq = one_hot(c3shft_seq, self.num_c3)\
                        .unsqueeze(-1).float()

                    # dshft_one_hot_seq: [batch_size, seq_len, 1, num_d]
                    dshft_one_hot_seq = one_hot(dshft_seq, self.num_d)\
                        .unsqueeze(-2).float()

                    # xshft_one_hot_seq:
                    # [batch_size, seq_len, num_c3, num_d]
                    xshft_one_hot_seq = \
                        cshft_one_hot_seq * dshft_one_hot_seq

                    # alpha_shft_seq: [batch_size, seq_len]
                    alpha_shft_seq = \
                        (alpha_seq * xshft_one_hot_seq)\
                        .sum(-1).sum(-1)

                    # rshft_hat_seq: [batch_size, seq_len]
                    rshft_hat_seq = torch.sigmoid(
                        alpha_shft_seq
                    )

                    train_loss_mean = np.mean(train_loss_mean)
                    test_loss_mean = binary_cross_entropy_with_logits(
                        torch.masked_select(alpha_shft_seq, m_seq),
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
