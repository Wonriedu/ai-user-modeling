import numpy as np
import torch

from torch.nn.functional import one_hot


class UserSimulator:
    def __init__(self, model) -> None:
        self.model = model

        self.state_dim = 3
        self.action_dim = self.model.num_d

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
        if h_0 is not None:
            h = torch.clone(h_0)
        else:
            h = torch.zeros([batch_size, self.model.dim_v])

        # alpha: [batch_size]
        alpha = self.model.linear_1(h).squeeze()
        alpha = torch.reshape(alpha, [batch_size])

        # C1: [batch_size, num_c1, 1]
        if C1_0 is not None:
            C1 = torch.clone(C1_0)
        else:
            C1 = torch.zeros([batch_size, self.model.num_c1, 1])

        # C2: [batch_size, num_c2, 1]
        if C2_0 is not None:
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

        return alpha_seq, r_seq, h_seq, C1_seq, C2_seq

    def expectimax(
        self, c1_seq, c2_seq, h_0=None, C1_0=None, C2_0=None,
    ):
        '''
            Args:
                c1_seq: [batch_size, seq_len]
                c2_seq: [batch_size, seq_len]
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
        num_d = self.model.num_d

        self.model.eval()

        ######################################################
        # torch.Tensor.expand need to be contiguous!
        # c1_seq_repeat: [batch_size * num_d, seq_len]
        # c2_seq_repeat: [batch_size * num_d, seq_len]
        c1_seq_repeat = c1_seq.expand(num_d, -1).contiguous()
        c2_seq_repeat = c2_seq.expand(num_d, -1).contiguous()

        # Initial response generation

        # h_seq: [batch_size, dim_v]
        if h_0 is not None:
            h = torch.clone(h_0)
        else:
            h = torch.zeros([batch_size, self.model.dim_v])

        # h_0_repeat: [batch_size * num_d, dim_v]
        h_0_repeat = h.expand(num_d, -1).contiguous()

        # alpha: [batch_size, 1]
        alpha = self.model.linear_1(h).squeeze()
        alpha = torch.reshape(alpha, [batch_size, 1])

        # C1: [batch_size, num_c1, 1]
        if C1_0 is not None:
            C1 = torch.clone(C1_0)
        else:
            C1 = torch.zeros([batch_size, self.model.num_c1, 1])

        # C2: [batch_size, num_c2, 1]
        if C2_0 is not None:
            C2 = torch.clone(C2_0)
        else:
            C2 = torch.zeros([batch_size, self.model.num_c2, 1])

        # C1_0_repeat: [batch_size * num_d, num_c1, 1]
        # C2_0_repeat: [batch_size * num_d, num_c2, 1]
        C1_0_repeat = C1.expand(num_d, -1, -1).contiguous()
        C2_0_repeat = C2.expand(num_d, -1, -1).contiguous()

        # c1_one_hot: [batch_size, 1, num_c1]
        # c2_one_hot: [batch_size, 1, num_c2]
        c1_one_hot = one_hot(c1_seq[:, :1], self.model.num_c1).float()
        c2_one_hot = one_hot(c2_seq[:, :1], self.model.num_c2).float()

        # beta1, beta2: [batch_size, 1]
        beta1 = torch.bmm(c1_one_hot, C1).squeeze()
        beta1 = torch.reshape(beta1, [batch_size, 1])
        beta2 = torch.bmm(c2_one_hot, C2).squeeze()
        beta2 = torch.reshape(beta2, [batch_size, 1])

        # gamma: [1, num_d]
        gamma = self.model.D(
            torch.tensor(np.arange(num_d)).long()
        ).squeeze()
        gamma = torch.reshape(gamma, [1, -1])

        # p: [batch_size, num_d]
        p = torch.sigmoid(alpha + beta1 + beta2 - gamma)\
            .detach().cpu().numpy()

        # d_seq: [batch_size * num_d, 1]
        d_seq = torch.tensor(np.arange(num_d)).unsqueeze(-1).long()
        if batch_size > 1:
            d_seq = d_seq.expand(batch_size, -1).contiguous()

        # r_seq_one: [batch_size * num_d, 1]
        # r_seq_zero: [batch_size * num_d, 1]
        r_seq_one = torch.tensor(np.ones([batch_size * num_d, 1])).float()
        r_seq_zero = torch.tensor(np.zeros([batch_size * num_d, 1])).float()

        alpha_seq_one, _, C1_seq, C2_seq = self.model(
            c1_seq_repeat[:, :1],
            c2_seq_repeat[:, :1],
            d_seq,
            r_seq_one,
            h_0_repeat,
            C1_0_repeat,
            C2_0_repeat
        )
        alpha_seq_zero, _, C1_seq, C2_seq = self.model(
            c1_seq_repeat[:, :1],
            c2_seq_repeat[:, :1],
            d_seq,
            r_seq_zero,
            h_0_repeat,
            C1_0_repeat,
            C2_0_repeat
        )

        # alpha_one: [batch_size, num_d]
        # alpha_zero: [batch_size, num_d]
        alpha_one = torch.reshape(alpha_seq_one, [batch_size, num_d])\
            .detach().cpu().numpy()
        alpha_zero = torch.reshape(alpha_seq_zero, [batch_size, num_d])\
            .detach().cpu().numpy()

        # reward: [batch_size, num_d]
        reward = alpha_one * p + alpha_zero * (1 - p)

        # d_seq: [batch_size, 1]
        d_seq = np.argmax(reward, axis=-1)
        d_seq = torch.tensor(
            np.expand_dims(d_seq, axis=-1)
        ).contiguous().long()

        # gamma: [batch_size, 1]
        gamma = self.model.D(d_seq)
        gamma = torch.reshape(gamma, [batch_size])

        # p_0: [batch_size]
        p_0 = torch.sigmoid(alpha + beta1 + beta2 - gamma)\
            .squeeze().detach().cpu().numpy()

        # r_seq: [batch_size, 1]
        r_seq = torch.tensor(
            np.random.binomial(1, p_0, [batch_size, 1])
        )

        for i in range(seq_len - 1):
            alpha_seq, h_seq, C1_seq, C2_seq = self.model(
                c1_seq[:, :i + 1],
                c2_seq[:, :i + 1],
                d_seq,
                r_seq,
                h_0,
                C1_0,
                C2_0
            )

            # Response generation

            # alpha: [batch_size, 1]
            alpha = alpha_seq[:, -1]
            alpha = torch.reshape(alpha, [batch_size, 1])

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

            # beta1, beta2: [batch_size, 1]
            beta1 = torch.bmm(c1_one_hot, C1).squeeze()
            beta1 = torch.reshape(beta1, [batch_size, 1])
            beta2 = torch.bmm(c2_one_hot, C2).squeeze()
            beta2 = torch.reshape(beta2, [batch_size, 1])

            # gamma: [1, num_d]
            gamma = self.model.D(
                torch.tensor(np.arange(num_d)).long()
            ).squeeze()
            gamma = torch.reshape(gamma, [1, -1])

            # p: [batch_size, num_d]
            p = torch.sigmoid(alpha + beta1 + beta2 - gamma)\
                .detach().cpu().numpy()

            # d: [batch_size * num_d, 1]
            d = torch.tensor(np.arange(num_d)).unsqueeze(-1).long()
            if batch_size > 1:
                d = d.expand(batch_size, -1).contiguous()

            # d_seq_repeat: [batch_size * num_d, 1]
            d_seq_repeat = d_seq.expand(num_d, -1).contiguous()

            d_seq_repeat = torch.cat([d_seq_repeat, d], dim=-1)

            # r_seq_one: [batch_size * num_d, 1]
            # r_seq_zero: [batch_size * num_d, 1]
            r_seq_one = torch.tensor(np.ones([batch_size * num_d, 1]))\
                .float()
            r_seq_zero = torch.tensor(np.zeros([batch_size * num_d, 1]))\
                .float()

            r_seq_one = torch.cat(
                [r_seq.expand(num_d, -1).contiguous(), r_seq_one], dim=-1
            )
            r_seq_zero = torch.cat(
                [r_seq.expand(num_d, -1).contiguous(), r_seq_zero], dim=-1
            )

            alpha_seq_one, _, C1_seq, C2_seq = self.model(
                c1_seq_repeat[:, :i + 2],
                c2_seq_repeat[:, :i + 2],
                d_seq_repeat,
                r_seq_one,
                h_0_repeat,
                C1_0_repeat,
                C2_0_repeat
            )
            alpha_seq_zero, _, C1_seq, C2_seq = self.model(
                c1_seq_repeat[:, :i + 2],
                c2_seq_repeat[:, :i + 2],
                d_seq_repeat,
                r_seq_zero,
                h_0_repeat,
                C1_0_repeat,
                C2_0_repeat
            )

            # alpha_one: [batch_size, num_d]
            # alpha_zero: [batch_size, num_d]
            alpha_one = torch.reshape(
                alpha_seq_one[:, -1], [batch_size, num_d]
            ).detach().cpu().numpy()
            alpha_zero = torch.reshape(
                alpha_seq_zero[:, -1], [batch_size, num_d]
            ).detach().cpu().numpy()

            # reward: [batch_size, num_d]
            reward = alpha_one * p + alpha_zero * (1 - p)

            # d: [batch_size, 1]
            d = np.argmax(reward, axis=-1)
            d = torch.tensor(np.expand_dims(d, axis=-1)).long()

            d_seq = torch.cat([d_seq, d], dim=-1)

            # gamma: [batch_size, 1]
            gamma = self.model.D(d)
            gamma = torch.reshape(gamma, [batch_size])

            # p: [batch_size]
            p = torch.sigmoid(alpha + beta1 + beta2 - gamma)\
                .squeeze().detach().cpu().numpy()

            # r: [batch_size, 1]
            r = torch.tensor(
                np.random.binomial(1, p, [batch_size, 1])
            )

            r_seq = torch.cat([r_seq, r], dim=1)

        alpha_seq, h_seq, C1_seq, C2_seq = self.model(
            c1_seq, c2_seq, d_seq, r_seq, h_0, C1_0, C2_0
        )

        return alpha_seq, d_seq, r_seq, h_seq, C1_seq, C2_seq

    def render(self):
        pass

    def reset(self):
        # h: [1, dim_v]
        self.h = torch.zeros([1, self.model.dim_v])

        # alpha: [batch_size]
        alpha = self.model.linear_1(self.h).squeeze()
        alpha = torch.reshape(alpha, [1])

        # C1: [1, num_c1, 1]
        self.C1 = torch.zeros([1, self.model.num_c1, 1])

        # C2: [1, num_c2, 1]
        self.C2 = torch.zeros([1, self.model.num_c2, 1])

        ob = torch.tensor(
            [alpha, self.C1.squeeze()[0], self.C2.squeeze()[0]]
        ).detach().cpu().numpy()

        return ob

    def step(self, d, c1=0, c2=0):
        '''
            Args:
                d: []
                c1: []
                c2: []
            Returns:
        '''

        # d_seq, c1_seq, c2_seq: [1, 1]
        d_seq = torch.tensor(np.array([[d]])).long()
        c1_seq = torch.tensor(np.array([[c1]])).long()
        c2_seq = torch.tensor(np.array([[c2]])).long()

        self.model.eval()

        # Initial response generation

        # alpha: [batch_size]
        alpha = self.model.linear_1(self.h).squeeze()
        alpha = torch.reshape(alpha, [1])

        # c1_one_hot: [1, 1, num_c1]
        # c2_one_hot: [1, 1, num_c2]
        c1_one_hot = one_hot(c1_seq[:, :1], self.model.num_c1).float()
        c2_one_hot = one_hot(c2_seq[:, :1], self.model.num_c2).float()

        # beta1, beta2: [1]
        beta1 = torch.bmm(c1_one_hot, self.C1).squeeze()
        beta1 = torch.reshape(beta1, [1])
        beta2 = torch.bmm(c2_one_hot, self.C2).squeeze()
        beta2 = torch.reshape(beta2, [1])

        # gamma: [1]
        gamma = self.model.D(d_seq[:, :1]).squeeze()
        gamma = torch.reshape(gamma, [1])

        # p_0: [1]
        p_0 = torch.sigmoid(alpha + beta1 + beta2 - gamma)\
            .detach().cpu().numpy()

        # r_seq: [1, 1]
        r_seq = torch.tensor(
            np.random.binomial(1, p_0, [1, 1])
        )

        alpha_seq, h_seq, C1_seq, C2_seq = self.model(
            c1_seq, c2_seq, d_seq, r_seq, self.h, self.C1, self.C2
        )

        self.h = h_seq[:, -1, :]
        self.C1 = C1_seq[:, -1, :, :]
        self.C2 = C2_seq[:, -1, :, :]

        # alpha: []
        alpha = alpha_seq[:, -1].squeeze()

        ob = torch.tensor(
            [alpha, self.C1.squeeze()[c1], self.C2.squeeze()[c2]]
        ).detach().cpu().numpy()
        rwd = alpha.detach().cpu().numpy()
        done = False
        info = None

        return ob, rwd, done, info
