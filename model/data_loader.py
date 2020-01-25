"""
Define the data to feed the deep learning model.

If batch_size = 1, each sequence is padded to reach length multiple of
"padded_seq_len"; each sequence is tehn trimmed in subsequences of
length "padded_seq_len".
"""

from torch.utils.data import Dataset
from utils import len_padded
import random
import torch
import os
import csv

"""
EHR data class
"""


class EHRdata(Dataset):

    def __init__(self, datadir, ehr_file, sampling):
        self.ehr = {}
        with open(os.path.join(datadir, ehr_file)) as f:
            rd = csv.reader(f)
            next(rd)
            for r in rd:
                seq = list(map(int, r[1::]))
                if len(seq) < len_padded:
                    self.ehr[r[0]] = seq + [0] * (len_padded - len(seq))

                elif len(seq) % len_padded != 0:
                    nseq, nleft = divmod(len(seq), len_padded)
                    self.ehr[r[0]] = seq + [0] * \
                                     (len_padded - nleft)
                else:
                    self.ehr[r[0]] = seq
        # sub-sample dataset
        if sampling is not None:
            mrns = list(self.ehr.keys())
            random.shuffle(mrns)
            ehr = {}
            for k in mrns[:sampling]:
                ehr[k] = self.ehr[k]
            self.ehr = ehr

        self.ehr_list = [[mrn, term] for mrn, term in self.ehr.items()]

    def __getitem__(self, index):
        seq = self.ehr_list[index][1]
        pat = self.ehr_list[index][0]
        return pat, seq

    def __len__(self):
        return len(self.ehr)


def ehr_collate(batch):
    data = []
    mrn = []
    for pat, seq in batch:
        mrn.append(pat)
        if len(seq) == len_padded:
            data.append(torch.tensor(
                [seq], dtype=torch.long).view(-1, len_padded))

        elif len(seq) > len_padded:
            ps = []
            for i in range(0, len(seq) - len_padded + 1,
                           len_padded + 1):
                ps.append(seq[i:i + len_padded])
            data.append(torch.tensor(
                ps, dtype=torch.long).view(-1, len_padded))

        else:
            raise Warning(
                'Not all sequences have length multiple than %d' % len_padded)

    return mrn, data
