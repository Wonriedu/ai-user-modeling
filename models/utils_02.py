import numpy as np
import torch

from torch.nn.utils.rnn import pad_sequence

if torch.cuda.is_available():
    from torch.cuda import LongTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import LongTensor


def match_seq_len(
    c3_seqs, d_seqs, r_seqs, seq_len, pad_val=-1
):
    '''
        Args:
            c3_seqs: [batch_size, some_sequence_length]
            d_seqs: [batch_size, some_sequence_length]
            r_seqs: [batch_size, some_sequence_length]
            seq_len: 균일하게 맞추고자 하는 시퀀스 길이
            pad_val: 시퀀스 길이를 맞추며 발생하는 패딩 값
        Returns:
            proc_c3_seqs: [batch_size, seq_len + 1]
            proc_d_seqs: [batch_size, seq_len + 1]
            proc_r_seqs: [batch_size, seq_len + 1]
    '''
    proc_c3_seqs = []
    proc_d_seqs = []
    proc_r_seqs = []

    for c3_seq, d_seq, r_seq in zip(
        c3_seqs, d_seqs, r_seqs
    ):
        i = 0
        while i + seq_len + 1 < len(d_seq):
            proc_c3_seqs.append(c3_seq[i:i + seq_len + 1])
            proc_d_seqs.append(d_seq[i:i + seq_len + 1])
            proc_r_seqs.append(r_seq[i:i + seq_len + 1])

            i += seq_len + 1

        proc_c3_seqs.append(
            np.concatenate(
                [
                    c3_seq[i:],
                    np.array([pad_val] * (i + seq_len + 1 - len(c3_seq)))
                ]
            )
        )
        proc_d_seqs.append(
            np.concatenate(
                [
                    d_seq[i:],
                    np.array([pad_val] * (i + seq_len + 1 - len(d_seq)))
                ]
            )
        )
        proc_r_seqs.append(
            np.concatenate(
                [
                    r_seq[i:],
                    np.array([pad_val] * (i + seq_len + 1 - len(d_seq)))
                ]
            )
        )

    return proc_c3_seqs, proc_d_seqs, proc_r_seqs


def collate_fn(batch, pad_val=-1):
    '''
        The collate function for torch.utils.data.DataLoader
        Returns:
            c3_seqs:
            d_seqs:
            r_seqs:
            c3shft_seqs: [batch_size, maximum_sequence_length_in_the_batch]
            dshft_seqs: [batch_size, maximum_sequence_length_in_the_batch]
            rshft_seqs: [batch_size, maximum_sequence_length_in_the_batch]
            mask_seqs: the mask sequences indicating where \
                the padded entry is with the size of \
                [batch_size, maximum_sequence_length_in_the_batch]
    '''
    c3_seqs = []
    d_seqs = []
    r_seqs = []
    c3shft_seqs = []
    dshft_seqs = []
    rshft_seqs = []

    for c3_seq, d_seq, r_seq in batch:
        c3_seqs.append(LongTensor(c3_seq[:-1]))
        d_seqs.append(LongTensor(d_seq[:-1]))
        r_seqs.append(LongTensor(r_seq[:-1]))
        c3shft_seqs.append(LongTensor(c3_seq[1:]))
        dshft_seqs.append(LongTensor(d_seq[1:]))
        rshft_seqs.append(LongTensor(r_seq[1:]))

    c3_seqs = pad_sequence(
        c3_seqs, batch_first=True, padding_value=pad_val
    )

    d_seqs = pad_sequence(
        d_seqs, batch_first=True, padding_value=pad_val
    )

    r_seqs = pad_sequence(
        r_seqs, batch_first=True, padding_value=pad_val
    )

    c3shft_seqs = pad_sequence(
        c3shft_seqs, batch_first=True, padding_value=pad_val
    )

    dshft_seqs = pad_sequence(
        dshft_seqs, batch_first=True, padding_value=pad_val
    )

    rshft_seqs = pad_sequence(
        rshft_seqs, batch_first=True, padding_value=pad_val
    )

    mask_seqs = (d_seqs != pad_val) * (dshft_seqs != pad_val)

    c3_seqs = c3_seqs * mask_seqs
    d_seqs = d_seqs * mask_seqs
    r_seqs = r_seqs * mask_seqs
    c3shft_seqs = c3shft_seqs * mask_seqs
    dshft_seqs = dshft_seqs * mask_seqs
    rshft_seqs = rshft_seqs * mask_seqs

    return c3_seqs, d_seqs, r_seqs, \
        c3shft_seqs, dshft_seqs, rshft_seqs, \
        mask_seqs
