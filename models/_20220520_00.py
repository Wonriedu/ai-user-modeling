import torch

from torch.nn import Module, Embedding, Parameter, GRU, Sequential, Linear, ReLU
from torch.nn.functional import one_hot


class UserModel(Module):
    def __init__(self, num_c, num_d, dim_v):
        super().__init__()

        self.num_c = num_c
        self.num_d = num_d

        self.dim_v = dim_v

        self.D1 = Embedding(self.num_d, 1)
        self.D2 = Embedding(self.num_d, self.dim_v)

        self.v_r = Parameter(torch.Tensor(self.dim_v))
        self.v_beta = Parameter(torch.Tensor(self.dim_v))

        self.gru = GRU(self.dim_v * 2, self.dim_v, batch_first=True)
        self.linear_1 = Sequential(
            Linear(self.dim_v, self.dim_v),
            ReLU,
            Linear(self.dim_v, self.dim_v),
            ReLU,
            Linear(self.dim_v, 1),
        )

        self.linear_2 = Sequential(
            Linear(self.dim_v * 3, self.dim_v),
            ReLU,
            Linear(self.dim_v, self.dim_v),
            ReLU,
            Linear(self.dim_v, 1),
        )

    def forward(self, c_seq, d_seq, r_seq, h_0=None, C_0=None):
        '''
            Args:
                c_seq: [batch_size, seq_len]
                d_seq: [batch_size, seq_len]
                r_seq: [batch_size, seq_len]
                h_0: [batch_size, dim_v]
                C_0: [batch_size, num_c, 1]
        '''
        batch_size = c_seq.shape[0]

        # v_d_seq, v_r_seq: [batch_size, seq_len, dim_v]
        v_d_seq = self.D2(d_seq)
        v_r_seq = r_seq.unsqueeze(-1) * self.v_r

        # h_seq: [batch_size, seq_len, dim_v]
        if h_0:
            h_seq = self.gru(torch.cat([v_d_seq, v_r_seq], dim=-1), h_0)
        else:
            h_seq = self.gru(torch.cat([v_d_seq, v_r_seq], dim=-1))

        # alpha_seq: [batch_size, seq_len]
        alpha_seq = self.linear_1(h_seq).squeeze()

        # C: [batch_size, num_c, 1]
        if C_0:
            C = C_0
        else:
            C = torch.zeros([batch_size, self.num_c, 1])
        C_seq = []

        # c_one_hot_seq: [batch_size, seq_len, num_c]
        c_one_hot_seq = one_hot(c_seq, self.num_c)

        beta_seq = []

        for c_one_hot, v_d, v_r in zip(
            c_one_hot_seq.permute(1, 0, 2),
            v_d_seq.permute(1, 0, 2),
            v_r_seq.permute(1, 0, 2)
        ):
            # c_one_hot: [batch_size, num_c]
            # v_d, v_r: [batch_size, dim_v]

            # beta_prev: [batch_size, 1, 1]
            beta_prev = torch.bmm(c_one_hot.unsqueeze(1), C)

            # v_beta_prev: [batch_size, dim_v]
            v_beta_prev = (beta_prev * self.v_beta).squeeze()

            # new_c: [batch_size, 1]
            new_c = self.linear_2(torch.cat([v_beta_prev, v_d, v_r], dim=-1))

            C = C * (1 - c_one_hot.unsqueeze(-1)) + \
                new_c * c_one_hot.unsqueeze(-1)

            C_seq.append(C)

            # beta: [batch_size]
            beta = torch.bmm(c_one_hot.unsqueeze(1), C).squeeze()
            beta_seq.append(beta)

        # C_seq: [batch_size, seq_len, num_c, 1]
        C_seq = torch.stack(C_seq, dim=1)

        # beta_seq: [batch_size, seq_len]
        beta_seq = torch.stack(beta_seq, dim=1)

        # gamma_seq: [batch_size, seq_len]
        gamma_seq = self.D1(d_seq).squeeze()

        return alpha_seq, beta_seq, gamma_seq, h_seq, C_seq

    def train(self):
        pass
