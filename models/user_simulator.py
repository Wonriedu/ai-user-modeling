import numpy as np
import torch

from torch.nn.functional import one_hot


class UserSimulator:
    def __init__(self, model) -> None:
        self.model = model

    def simulate(
        self, c1_seq, c2_seq, d_seq, h_0=None, C1_0=None, C2_0=None,
    ):
        '''
            Args:
                c1_seq: [batch_size, seq_len]
                c2_seq: [batch_size, seq_len]
                d_seq: [batch_size, seq_len]
                h_0: [batch_size, dim_v]
                C1_0: [batch_size, num_c1, 1]
                C2_0: [batch_size, num_c2, 1]

            Returns:
                alpha_seq: [batch_size, seq_len]
                h_seq: [batch_size, seq_len, dim_v]
                C1_seq: [batch_size, seq_len, num_c1, 1]
                C2_seq: [batch_size, seq_len, num_c2, 1]
        '''
        batch_size = c1_seq.shape[0]
        seq_len = c1_seq.shape[1]

        self.model.eval()

        # Initial response generation

        # h_seq: [batch_size, dim_v]
        if h_0:
            h = torch.clone(h_0)
        else:
            h = torch.zeros([batch_size, self.model.dim_v])

        # alpha: [batch_size]
        alpha = self.model.linear_1(h).squeeze()
        alpha = torch.reshape(alpha, [batch_size])

        # C1: [batch_size, num_c1, 1]
        if C1_0:
            C1 = torch.clone(C1_0)
        else:
            C1 = torch.zeros([batch_size, self.model.num_c1, 1])

        # C2: [batch_size, num_c2, 1]
        if C2_0:
            C2 = torch.clone(C2_0)
        else:
            C2 = torch.zeros([batch_size, self.model.num_c2, 1])

        # c1_one_hot: [batch_size, 1, num_c1]
        # c2_one_hot: [batch_size, 1, num_c2]
        c1_one_hot = one_hot(c1_seq[:, :1], self.model.num_c1).float()
        c2_one_hot = one_hot(c2_seq[:, :1], self.model.num_c2).float()

        # beta1, beta2: [batch_size]
        beta1 = torch.bmm(c1_one_hot, C1).squeeze()
        beta1 = torch.reshape(beta1, [batch_size])
        beta2 = torch.bmm(c2_one_hot, C2).squeeze()
        beta2 = torch.reshape(beta2, [batch_size])

        # gamma: [batch_size]
        gamma = self.model.D(d_seq[:, :1]).squeeze()
        gamma = torch.reshape(gamma, [batch_size])

        # p_0: [batch_size]
        p_0 = torch.sigmoid(alpha + beta1 + beta2 - gamma)\
            .detach().cpu().numpy()

        # r_seq: [batch_size, 1]
        r_seq = torch.tensor(
            np.random.binomial(1, p_0, [batch_size, 1])
        )

        for i in range(seq_len - 1):
            alpha_seq, h_seq, C1_seq, C2_seq = self.model(
                c1_seq[:, :i + 1],
                c2_seq[:, :i + 1],
                d_seq[:, :i + 1],
                r_seq,
                h_0,
                C1_0,
                C2_0
            )

            # Response generation

            # alpha: [batch_size]
            alpha = alpha_seq[:, -1]

            # C1: [batch_size, num_c1, 1]
            # C2: [batch_size, num_c2, 1]
            C1 = C1_seq[:, -1]
            C2 = C2_seq[:, -1]

            # c1_one_hot: [batch_size, 1, num_c1]
            # c2_one_hot: [batch_size, 1, num_c2]
            c1_one_hot = one_hot(c1_seq[:, i + 1:i + 2], self.model.num_c1)\
                .float()
            c2_one_hot = one_hot(c2_seq[:, i + 1:i + 2], self.model.num_c2)\
                .float()

            # beta1, beta2: [batch_size]
            beta1 = torch.bmm(c1_one_hot, C1).squeeze()
            beta1 = torch.reshape(beta1, [batch_size])
            beta2 = torch.bmm(c2_one_hot, C2).squeeze()
            beta2 = torch.reshape(beta2, [batch_size])

            # gamma: [batch_size]
            gamma = self.model.D(d_seq[:, i + 1:i + 2]).squeeze()
            gamma = torch.reshape(gamma, [batch_size])

            # p: [batch_size]
            p = torch.sigmoid(alpha + beta1 + beta2 - gamma)\
                .detach().cpu().numpy()

            # r: [batch_size, 1]
            r = torch.tensor(
                np.random.binomial(1, p, [batch_size, 1])
            )

            r_seq = torch.cat([r_seq, r], dim=1)

        alpha_seq, h_seq, C1_seq, C2_seq = self.model(
            c1_seq, c2_seq, d_seq, r_seq, h_0, C1_0, C2_0
        )

        return alpha_seq, h_seq, C1_seq, C2_seq
