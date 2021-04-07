"""
SNLI dataset
"""


import os

import numpy as np
import spacy
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import pandas as pd


import models
from . import analysis


def clean(tok):
    if tok == "(":
        return "LPAREN"
    elif tok == ")":
        return "RPAREN"
    else:
        return tok


def load_for_analysis(
    ckpt_path,
    analysis_path,
    analysis_level="sentence",
    model_type="bowman",
    cuda=False,
    **kwargs,
):
    ckpt = torch.load(ckpt_path)
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
    """"""
    sentence, length, label = zip(*batch)
    sentence = pad_sequence(sentence, padding_value=1)
    length = torch.tensor(length)
    label = torch.tensor(label)
    return sentence, length, label


LABEL_STOI = {"negative": 0, "positive": 1}
LABEL_ITOS = {v: k for k, v in LABEL_STOI.items()}


class IMDB:
    def __init__(self, path, split, vocab=None):
        self.path = path

        # Counterfactual IMDB
        self.c = "counterfactual" in self.path

        self.split = split
        if not self.c:
            assert self.split in {"train", "dev", "test"}
            self.text_path = os.path.join(self.path, f"{self.split}.tsv")
        else:
            self.text_path = os.path.join(self.path, "cimdb.tsv")
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
        self.raw_sentences = []
        self.sentences = []
        self.lengths = []

        counts = Counter()
        df = pd.read_csv(self.text_path, header=0, sep="\t")
        # First parse - get tokens and counts if vocab exists
        for (i, row) in df.iterrows():
            sent = row["Sentiment"].lower()
            self.labels.append(LABEL_STOI[sent])
            text = row["Text"].lower()
            text_doc = self.spacy(text)
            # Max 300
            text_tok = [clean(t.lower_) for t in text_doc]
            text_tok = text_tok[:300]
            self.raw_sentences.append(text_tok)
            counts.update(text_tok)

        if vocab is None:
            # Build vocab
            top20k = [x[0] for x in counts.most_common(20000)]
            for t in top20k:
                self.stoi[t] = len(self.stoi)
            self.itos = {v: k for k, v in self.stoi.items()}

        for rs in self.raw_sentences:
            ris = [self.stoi.get(t, self.stoi["UNK"]) for t in rs]
            self.sentences.append(np.array(ris))
            self.lengths.append(len(ris))

    def __getitem__(self, i):
        sentence = torch.as_tensor(self.sentences[i])
        length = self.lengths[i]
        label = self.labels[i]
        return sentence, length, label

    def __len__(self):
        return len(self.sentences)
