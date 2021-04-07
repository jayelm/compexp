"""
Custom models
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


class SentimentClassifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()

        self.encoder = encoder
        self.encoder_dim = encoder.output_dim
        self.mlp_input_dim = self.encoder_dim
        self.dropout = nn.Dropout(0.5)

        self.mlp = nn.Sequential(
            nn.Linear(self.mlp_input_dim, 50),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(50, 1),
        )

        self.output_dim = 1

    def forward(self, text, length):
        enc = self.encoder(text, length)
        enc = self.dropout(enc)

        logits = self.mlp(enc)
        logits = logits.squeeze(1)
        return logits

    def get_final_reprs(self, text, length):
        """
        Get features right up to final decision
        """
        enc = self.encoder(text, length)
        rep = self.mlp[:-1](enc)
        return rep


class EntailmentClassifier(nn.Module):
    """
    An NLI entailment classifier where the hidden rep features are much
    "closer" to the actual feature decision
    """

    def __init__(self, encoder):
        super().__init__()

        self.encoder = encoder
        self.encoder_dim = encoder.output_dim
        self.mlp_input_dim = self.encoder_dim
        self.dropout = nn.Dropout(0.1)
        self.bn = nn.BatchNorm1d(self.mlp_input_dim)

        self.mlp = nn.Linear(self.mlp_input_dim, 3)

        self.output_dim = 3

    def forward(self, s1, s1len, s2, s2len):
        s1enc = self.encoder(s1, s1len)
        s2enc = self.encoder(s2, s2len)

        mlp_input = s1enc * s2enc

        mlp_input = self.bn(mlp_input)
        mlp_input = self.dropout(mlp_input)

        preds = self.mlp(mlp_input)

        return preds

    def get_final_reprs(self, s1, s1len, s2, s2len):
        """
        Get features right up to final decision
        """
        s1enc = self.encoder(s1, s1len)
        s2enc = self.encoder(s2, s2len)
        mlp_input = s1enc * s2enc

        return mlp_input


class BowmanEntailmentClassifier(nn.Module):
    """
    The RNN-based entailment model of Bowman et al 2017
    """

    def __init__(self, encoder):
        super().__init__()

        self.encoder = encoder
        self.encoder_dim = encoder.output_dim
        self.mlp_input_dim = self.encoder_dim * 4
        self.dropout = nn.Dropout(0.1)
        self.bn = nn.BatchNorm1d(self.mlp_input_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.mlp_input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),  # Mimic classifier MLP keep rate of 94%
            nn.Linear(1024, 3),
        )

        self.output_dim = 3

    def forward(self, s1, s1len, s2, s2len):
        s1enc = self.encoder(s1, s1len)
        s2enc = self.encoder(s2, s2len)

        diffs = s1enc - s2enc
        prods = s1enc * s2enc

        mlp_input = torch.cat([s1enc, s2enc, diffs, prods], 1)

        mlp_input = self.bn(mlp_input)
        mlp_input = self.dropout(mlp_input)

        preds = self.mlp(mlp_input)

        return preds

    def get_final_reprs(self, s1, s1len, s2, s2len):
        s1enc = self.encoder(s1, s1len)
        s2enc = self.encoder(s2, s2len)

        diffs = s1enc - s2enc
        prods = s1enc * s2enc

        mlp_input = torch.cat([s1enc, s2enc, diffs, prods], 1)

        mlp_input = self.bn(mlp_input)
        mlp_input = self.dropout(mlp_input)

        rep = self.mlp[:-1](mlp_input)
        return rep

    def forward_from_final(self, rep):
        preds = self.mlp[-1:](rep)
        return preds


class DropoutLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.W_i = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.U_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))

        self.W_f = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.U_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))

        self.W_c = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.U_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_c = nn.Parameter(torch.Tensor(hidden_size))

        self.W_o = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.U_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        self._input_dropout_mask = self._h_dropout_mask = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.W_i)
        nn.init.orthogonal_(self.U_i)
        nn.init.orthogonal_(self.W_f)
        nn.init.orthogonal_(self.U_f)
        nn.init.orthogonal_(self.W_o)
        nn.init.orthogonal_(self.U_o)
        nn.init.orthogonal_(self.W_c)
        nn.init.orthogonal_(self.U_c)
        self.b_f.data.fill_(1.0)
        self.b_i.data.fill_(1.0)
        self.b_o.data.fill_(1.0)

    def set_dropout_masks(self, batch_size):
        if self.dropout:
            if self.training:
                self._input_dropout_mask = torch.bernoulli(
                    torch.Tensor(4, batch_size, self.input_size).fill_(1 - self.dropout)
                )
                self._input_dropout_mask.requires_grad = False
                self._h_dropout_mask = torch.bernoulli(
                    torch.Tensor(4, batch_size, self.hidden_size).fill_(
                        1 - self.dropout
                    )
                )
                self._h_dropout_mask.requires_grad = False

                if torch.cuda.is_available():
                    self._input_dropout_mask = self._input_dropout_mask.cuda()
                    self._h_dropout_mask = self._h_dropout_mask.cuda()
            else:
                self._input_dropout_mask = self._h_dropout_mask = [
                    1.0 - self.dropout
                ] * 4
        else:
            self._input_dropout_mask = self._h_dropout_mask = [1.0] * 4

    def forward(self, input, hidden_state):
        h_tm1, c_tm1 = hidden_state

        if self._input_dropout_mask is None:
            self.set_dropout_masks(input.size(0))

        xi_t = F.linear(input * self._input_dropout_mask[0], self.W_i, self.b_i)
        xf_t = F.linear(input * self._input_dropout_mask[1], self.W_f, self.b_f)
        xc_t = F.linear(input * self._input_dropout_mask[2], self.W_c, self.b_c)
        xo_t = F.linear(input * self._input_dropout_mask[3], self.W_o, self.b_o)

        i_t = F.sigmoid(xi_t + F.linear(h_tm1 * self._h_dropout_mask[0], self.U_i))
        f_t = F.sigmoid(xf_t + F.linear(h_tm1 * self._h_dropout_mask[1], self.U_f))
        c_t = f_t * c_tm1 + i_t * F.tanh(
            xc_t + F.linear(h_tm1 * self._h_dropout_mask[2], self.U_c)
        )
        o_t = F.sigmoid(xo_t + F.linear(h_tm1 * self._h_dropout_mask[3], self.U_o))
        h_t = o_t * F.tanh(c_t)

        return h_t, c_t


class TextEncoder(nn.Module):
    def __init__(
        self, vocab_size, embedding_dim=300, hidden_dim=512, bidirectional=False
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.bidirectional = bidirectional
        self.emb = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=1)
        self.rnn = nn.LSTM(
            self.embedding_dim, self.hidden_dim, bidirectional=bidirectional
        )
        self.output_dim = self.hidden_dim

    def forward(self, s, slen):
        semb = self.emb(s)
        spk = pack_padded_sequence(semb, slen.cpu(), enforce_sorted=False)
        _, (states, _) = self.rnn(spk)

        return states[-1]

    def get_states(self, s, slen):
        semb = self.emb(s)
        spk = pack_padded_sequence(semb, slen.cpu(), enforce_sorted=False)
        outputs, _ = self.rnn(spk)

        outputs_pad = pad_packed_sequence(outputs)[0]
        return outputs_pad


class DropoutTextEncoder(nn.Module):
    def __init__(
        self, vocab_size, embedding_dim=300, hidden_dim=512, bidirectional=False
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.bidirectional = bidirectional
        self.emb = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=1)
        self.rnn_cell = DropoutLSTMCell(
            self.embedding_dim, self.hidden_dim, dropout=0.5
        )
        self.output_dim = self.hidden_dim

    def forward(self, s, slen):
        semb = self.emb(s)

        hx = torch.zeros(semb.shape[1], self.hidden_dim).to(semb.device)
        cx = torch.zeros(semb.shape[1], self.hidden_dim).to(semb.device)
        for i in range(semb.shape[0]):
            hx, cx = self.rnn_cell(semb[i], (hx, cx))
        return hx

    def get_states(self, s, slen):
        raise NotImplementedError
