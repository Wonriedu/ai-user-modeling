import os

import numpy as np
import torch

from torch.nn import Module, Embedding, Parameter, GRU, Sequential, Linear, \
    ReLU, Dropout
from torch.nn.functional import one_hot, binary_cross_entropy_with_logits
from sklearn import metrics


class UserModel(Module):
    def __init__(self, num_c1, num_c2, num_c4, num_d, dim_v):
        super().__init__()

        self.num_c1 = num_c1
        self.num_c2 = num_c2
        self.num_c4 = num_c4
        self.num_d = num_d

        self.dim_v = dim_v

        self.v_c2 = Parameter(torch.Tensor(self.dim_v))

        self.D = Embedding(self.num_d, 1)

        self.v_d = Parameter(torch.Tensor(self.dim_v))

        self.R = Embedding(2, self.dim_v)

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
        self, c1_seq, c2_seq, c4_seq, d_seq, r_seq,
        h_0=None, C1_0=None, C2_0=None, C4_0=None
    ):
        '''
            Args:
                c1_seq: [batch_size, seq_len]
                c2_seq: [batch_size, seq_len]
                c4_seq: [batch_size, seq_len]
                d_seq: [batch_size, seq_len]
                r_seq: [batch_size, seq_len]
                h_0: [batch_size, dim_v]
                C1_0: [batch_size, num_c1]
                C2_0: [batch_size, num_c2]
                C4_0: [batch_size, num_c4]

            Returns:
                alpha_seq: [batch_size, seq_len]
                h_seq: [batch_size, seq_len, dim_v]
                C1_seq: [batch_size, seq_len, num_c1]
                C2_seq: [batch_size, seq_len, num_c2]
                C4_seq: [batch_size, seq_len, num_c4]
        '''
        batch_size = c1_seq.shape[0]
        seq_len = c1_seq.shape[1]

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

        # alpha_seq: [batch_size, seq_len]
        alpha_seq = self.linear_1(h_seq).squeeze()
        alpha_seq = torch.reshape(alpha_seq, [batch_size, seq_len])

        # C2: [batch_size, num_c2]
        if C2_0 is not None:
            C2 = torch.clone(C2_0)
        else:
            C2 = torch.zeros([batch_size, self.num_c2])
        C2_seq = []

        for c1, c2, c4, v_d, v_r in zip(
            c1_seq.permute(1, 0),
            c2_seq.permute(1, 0),
            c4_seq.permute(1, 0),
            v_d_seq.permute(1, 0, 2),
            v_r_seq.permute(1, 0, 2)
        ):
            # c1, c2, c4: [batch_size]
            # v_d, v_r: [batch_size, dim_v]

            # beta1_tilde: [batch_size, 1]
            # beta2_tilde: [batch_size, 1]
            # beta4_tilde: [batch_size, 1]
            beta2_tilde = torch.gather(C2, dim=-1, index=c2.unsqueeze(-1))

            # v_c1: [batch_size, dim_v]
            # v_c2: [batch_size, dim_v]
            # v_c4: [batch_size, dim_v]
            v_c2 = (beta2_tilde * self.v_c2)

            v_c2 = torch.reshape(v_c2, [batch_size, self.dim_v])

            # new_c: [batch_size, 3]
            # new_c1, new_c2, new_c4: [batch_size]
            new_c = self.linear_2(
                torch.cat([v_c2, v_d, v_r], dim=-1)
            )
            new_c2 = new_c[:, 0]

            # c1_one_hot: [batch_size, num_c1]
            # c2_one_hot: [batch_size, num_c2]
            # c4_one_hot: [batch_size, num_c4]
            c2_one_hot = one_hot(c2, self.num_c2)

            C2 = C2 * (1 - c2_one_hot) + \
                new_c2.unsqueeze(-1) * c2_one_hot

            C2_seq.append(C2)

        # C1_seq: [batch_size, seq_len, num_c1]
        # C2_seq: [batch_size, seq_len, num_c2]
        # C4_seq: [batch_size, seq_len, num_c4]
        C2_seq = torch.stack(C2_seq, dim=1)

        return alpha_seq, h_seq, None, C2_seq, None

    def get_logits(
        self,
        c1_seq, c2_seq, c4_seq, d_seq, r_seq,
        c1shft_seq, c2shft_seq, c4shft_seq,
        dshft_seq, rshft_seq, m_seq
    ):
        batch_size = c2_seq.shape[0]
        seq_len = c2_seq.shape[1]

        # rshft_seq: [batch_size, seq_len]
        # m_seq: [batch_size, seq_len]

        alpha_seq, h_seq, C1_seq, C2_seq, C4_seq = \
            self(c1_seq, c2_seq, c4_seq, d_seq, r_seq)

        # alpha_seq: [batch_size, seq_len]

        # beta1_shft_seq: [batch_size, seq_len]
        # beta2_shft_seq: [batch_size, seq_len]
        # beta4_shft_seq: [batch_size, seq_len]
        beta2_shft_seq = torch.gather(
            C2_seq, dim=-1, index=c2shft_seq.unsqueeze(-1)
        ).reshape([batch_size, seq_len])

        # gamma_shft_seq: [batch_size, seq_len]
        gamma_shft_seq = self.D(dshft_seq).squeeze()

        logits = \
            alpha_seq + \
            beta2_shft_seq - \
            gamma_shft_seq

        return logits

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
                c1_seq, c2_seq, c4_seq, d_seq, r_seq, \
                    c1shft_seq, c2shft_seq, c4shft_seq, \
                    dshft_seq, rshft_seq, m_seq = data

                self.train()
                logits = self.get_logits(*data)

                opt.zero_grad()
                loss = binary_cross_entropy_with_logits(
                    torch.masked_select(
                        logits, m_seq
                    ),
                    torch.masked_select(rshft_seq.float(), m_seq)
                )
                loss.backward()
                opt.step()

                train_loss_mean.append(loss.detach().cpu().numpy())

            with torch.no_grad():
                for data in test_loader:
                    c1_seq, c2_seq, c4_seq, d_seq, r_seq, \
                        c1shft_seq, c2shft_seq, c4shft_seq, \
                        dshft_seq, rshft_seq, m_seq = data

                    self.eval()
                    logits = self.get_logits(*data)

                    # rshft_hat_seq: [batch_size, seq_len]
                    rshft_hat_seq = torch.sigmoid(logits)

                    train_loss_mean = np.mean(train_loss_mean)
                    test_loss_mean = binary_cross_entropy_with_logits(
                        torch.masked_select(
                            logits, m_seq
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
