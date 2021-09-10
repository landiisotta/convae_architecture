import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class ehrEncoding(nn.Module):

    def __init__(self, vocab_size, max_seq_len,
                 emb_size, kernel_size, pre_embs=None, vocab=None):
        super(ehrEncoding, self).__init__()

        # initiate parameters
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.emb_size = emb_size
        self.kernel_size = kernel_size
        self.ch_l1 = int(emb_size / 2)
        # self.ch_l2 = int(self.ch_l1 / 2)

        self.padding = int((kernel_size - 1) / 2)
        self.features = math.floor(
            max_seq_len + 2 * self.padding - kernel_size + 1) \
                        + 2 * self.padding - kernel_size + 1

        self.embedding = nn.Embedding(
            vocab_size, self.emb_size, padding_idx=0)

        # load pre-computed embeddings
        cnd_emb = pre_embs is not None
        cnd_vocab = vocab is not None
        if cnd_emb and cnd_vocab:
            weight_mtx = np.zeros((vocab_size, emb_size))
            wfound = 0
            for i in range(vocab_size):
                if i != 0:
                    try:
                        weight_mtx[i] = pre_embs[vocab[i]]
                        wfound += 1
                    except KeyError:
                        weight_mtx[i] = np.random.normal(
                            scale=0.6, size=(emb_size,))
            print('Found pre-computed embeddings for {0} concepts'.format(
                wfound))
            self.embedding = self.embedding.from_pretrained(torch.FloatTensor(weight_mtx), freeze=False)

        self.cnn_l1 = nn.Conv1d(self.emb_size, self.ch_l1,
                                kernel_size=kernel_size, padding=self.padding)
        self.bn1 = nn.BatchNorm1d(self.ch_l1)

        # self.cnn_l2 = nn.Conv1d(self.ch_l1, self.ch_l2,
        #                        kernel_size=kernel_size, padding=self.padding)

        self.linear1 = nn.Linear(self.ch_l1 * self.features, 200)
        self.linear2 = nn.Linear(200, 100)
        self.linear3 = nn.Linear(100, 200)
        self.linear4 = nn.Linear(200, vocab_size * max_seq_len)

        self.sigm = nn.Sigmoid()
        self.softplus = nn.Softplus()

        # self.bn1 = nn.BatchNorm1d(self.ch_l1)
        # self.bn2 = nn.BatchNorm1d(self.ch_l2)

    def forward(self, x):
        # embedding
        embeds = self.embedding(x)
        embeds = embeds.permute(0, 2, 1)

        # first CNN layer
        out = F.relu(self.cnn_l1(embeds))
        out = F.max_pool1d(out, kernel_size=self.kernel_size,
                           stride=1, padding=self.padding)

        # second CNN layer
        # out = F.relu(self.cnn_l2(out))
        # out = F.max_pool1d(out, kernel_size=self.kernel_size,
        #                   stride=1, padding=self.padding)

        out = out.view(-1, out.shape[2] * out.shape[1])

        # two layers of encoding
        out = self.linear1(out)
        out = F.dropout(out)
        out = F.relu(out)

        out = self.linear2(out)

        # encoded representation
        encoded_vect = out.view(-1, out.shape[1])

        # two layers of decoding
        out = self.linear3(out)
        out = F.softplus(out)

        out = self.linear4(out)

        out = out.view(-1, self.vocab_size, x.shape[1])

        return out, encoded_vect


def accuracy(out, target):
    logsoft = F.log_softmax(out, dim=1)
    pred = torch.argmax(logsoft, dim=1)
    return torch.sum((pred == target).float()) / (out.shape[2] * out.shape[0])


criterion = nn.CrossEntropyLoss()

metrics = {'accuracy': accuracy}
