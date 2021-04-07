"""
SNLI dataset
"""


import os

import numpy as np
import spacy
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence


import models
from . import analysis


def load_for_analysis(
    ckpt_path,
    analysis_path,
    model_type="bowman",
    cuda=False,
    **kwargs,
):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    enc = models.TextEncoder(len(ckpt["stoi"]), **kwargs)
    if model_type == "minimal":
        clf = models.EntailmentClassifier
    elif model_type == "bowman":
        clf = models.BowmanEntailmentClassifier
    else:
        raise NotImplementedError

    model = clf(enc)
    model.load_state_dict(ckpt["state_dict"])

    vocab = {"itos": ckpt["itos"], "stoi": ckpt["stoi"]}
    with open(analysis_path, "r") as f:
        lines = f.readlines()

    dataset = analysis.AnalysisDataset(lines, vocab)

    if cuda:
        model = model.cuda()

    return model, dataset


def pad_collate(batch):
    """
    We don't sort here to take advantage of enforce_sorted=False since we'd
    have to sort separately for both s1 and s2
    """
    s1, s1len, s2, s2len, label = zip(*batch)
    label = torch.tensor(label)

    s1_pad = pad_sequence(s1, padding_value=1)
    s1len = torch.tensor(s1len)

    s2_pad = pad_sequence(s2, padding_value=1)
    s2len = torch.tensor(s2len)

    return s1_pad, s1len, s2_pad, s2len, label


LABEL_STOI = {"entailment": 0, "neutral": 1, "contradiction": 2}
LABEL_ITOS = {v: k for k, v in LABEL_STOI.items()}


class SNLI:
    def __init__(self, path, split, vocab=None, max_data=None, unknowns=False):
        self.path = path
        self.unknowns = unknowns

        # Counterfactual SNLI
        self.c = "counterfactual" in self.path

        self.split = split
        if not self.c:
            assert self.split in {"train", "dev", "test"}
            self.text_path = os.path.join(self.path, f"snli_1.0_{self.split}.txt")
        else:
            self.text_path = os.path.join(self.path, "csnli.tsv")
        self.max_data = max_data
        self.label_stoi = LABEL_STOI
        self.label_itos = LABEL_ITOS
        self.spacy = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner"])

        if vocab is None:
            self.stoi = {
                "UNK": 0,
                "PAD": 1,
            }
        else:
            self.stoi, self.itos = vocab

        self.labels = []
        self.raw_s1s = []
        self.raw_s2s = []
        self.s1s = []
        self.s2s = []
        self.s1lens = []
        self.s2lens = []
        n_skipped = 0
        with open(self.text_path, "r") as f:
            for i, line in enumerate(tqdm(f, desc=self.split)):

                if i == 0:  # Header
                    continue

                if self.c:
                    s1, s2, label = line.strip().split("\t")
                else:
                    label, _, _, _, _, s1, s2, *_ = line.strip().split("\t")

                if label not in self.label_stoi:  # Hard example
                    assert label == "-"
                    if self.unknowns:
                        label_i = -1
                    else:
                        n_skipped += 1
                        continue
                else:
                    label_i = self.label_stoi[label]
                self.labels.append(label_i)

                if self.max_data is not None and i >= self.max_data:
                    break

                s1_doc = self.spacy(s1)
                s1_tok = [t.lower_ for t in s1_doc]
                s2_doc = self.spacy(s2)
                s2_tok = [t.lower_ for t in s2_doc]

                self.raw_s1s.append(s1_tok)
                self.raw_s2s.append(s2_tok)

                # Add to vocab
                s1_ns = []
                for tok in s1_tok:
                    if vocab is None and tok not in self.stoi:
                        # Build the vocab
                        self.stoi[tok] = len(self.stoi)
                    s1_ns.append(self.stoi.get(tok, 0))

                # Add to vocab
                s2_ns = []
                for tok in s2_tok:
                    if vocab is None and tok not in self.stoi:
                        # Build the vocab
                        self.stoi[tok] = len(self.stoi)
                    s2_ns.append(self.stoi.get(tok, 0))

                self.s1s.append(np.array(s1_ns))
                self.s1lens.append(len(s1_ns))
                self.s2s.append(np.array(s2_ns))
                self.s2lens.append(len(s2_ns))

        if vocab is None:
            self.itos = {v: k for k, v in self.stoi.items()}

    def __getitem__(self, i):
        s1 = torch.as_tensor(self.s1s[i])
        s1len = self.s1lens[i]

        s2 = torch.as_tensor(self.s2s[i])
        s2len = self.s2lens[i]

        label = self.labels[i]

        return s1, s1len, s2, s2len, label

    def __len__(self):
        return len(self.s1s)
