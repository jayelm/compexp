"""
Dataset for analysis of features
"""


import torch
import numpy as np


PAD_IDX = 1
UNK_TOKEN = "UNK"


class AnalysisDataset:
    def __init__(self, text_raw, fields):
        """
        Initialize an analysis dataset from the given text file and vocab
        (`fields`) file which gives word-level annotations.

        Closed categories are categories whose # feats is constant w.r.t. dataset size.

        Open categories are those whose # feats grows roughly linear in the dataset
        size (e.g. lemmas). We keep these as separate categories and do not
        automatically search for these.
        """
        self.words = []
        self.fs = []
        self.mfs = []
        self.lengths = []
        self.fields = fields

        if "src" in fields:  # onmt-style fields
            self.ignore_tokens = {
                self.fields["src"].base_field.pad_token,
                self.fields["src"].base_field.eos_token,
            }
            self.stoi = self.fields["src"].base_field.vocab.stoi
            self.itos = self.fields["src"].base_field.vocab.itos
        else:
            self.ignore_tokens = {
                "PAD",
            }
            self.stoi = self.fields["stoi"]
            self.itos = self.fields["itos"]

        self.pos_stoi = {}

        first_line = text_raw[0]
        self.byte_fmt = isinstance(first_line, bytes)
        if self.byte_fmt:
            first_line = first_line.decode("utf-8")
        # FIXME: closed feats should be part of the shared vocab (pos:)...etc.
        (
            self.cs,
            self.cnames,
            self.ctypes,
            self.ocnames,
            self.ccnames,
            self.mcnames,
        ) = self.compute_cat_names(first_line)

        self.text = text_raw[1:]

        # Mapping legend:
        # feats (short: f) are the individual values
        # cats (short: c) are the categories (e.g. POS, Lemma, Synset)
        # Mappings - feat names to individual features
        self.cnames2fis = {cn: set() for cn in self.cnames}
        self.parse_docs()
        self.fis2cnames = {}
        for cname, fis in self.cnames2fis.items():
            for fi in fis:
                self.fis2cnames[fi] = cname

        self.citos = dict(zip(self.cs, self.cnames))
        self.cstoi = dict(zip(self.cnames, self.cs))

        self.cis2fis = {
            self.cstoi[cname]: fis for cname, fis in self.cnames2fis.items()
        }
        self.fis2cis = {}
        for ci, fis in self.cis2fis.items():
            for fi in fis:
                self.fis2cis[fi] = ci

        self.n_cs = len(self.cnames)
        self.n_fs = len(self.fstoi)
        self.cfis = []
        self.ofis = []
        self.mfis = []
        for cname, fis in self.cnames2fis.items():
            ctype = self.ctypes[cname]
            if ctype == "open":
                self.ofis.extend(fis)
            elif ctype == "multi":
                self.mfis.extend(fis)
            else:
                self.cfis.extend(fis)

        for ws, fs, mfs in self.docs:
            self.words.append(np.array(ws))
            self.fs.append(np.array(fs))
            self.mfs.append(mfs)
            self.lengths.append(len(ws))

    def parse_docs(self):
        """
        Generator to parse documents, tokens, and feats
        """
        self.docs = []
        self.fstoi = {UNK_TOKEN: 0}
        self.fitos = {0: UNK_TOKEN}
        self.idx2multi = {}
        self.multi2idx = {}
        for line in self.text:
            line = line.strip()
            if self.byte_fmt:
                line = line.decode("utf-8")
            doc_words = []
            doc_feats = []
            doc_multifeats = []
            for tok in line.split(" "):
                word, *feats = tok.split("|")
                word_n = self.stoi.get(word.lower(), self.stoi["UNK"])
                feats = dict(zip(self.cnames, feats))
                feats_p = []
                multifeats_p = []
                for fn, f in feats.items():
                    if self.is_multi(fn):
                        fs = f.split(";")
                        fs_n = []
                        for f in fs:
                            # First assign global feature id
                            f = f"{fn}:{f}"
                            if f not in self.fstoi:
                                new_n = len(self.fstoi)
                                self.fstoi[f] = new_n
                                self.fitos[new_n] = f
                            f_n = self.fstoi[f]

                            # Next map it to a one hot index
                            if f_n not in self.multi2idx:
                                new_n = len(self.multi2idx)
                                self.multi2idx[f_n] = new_n
                                self.idx2multi[new_n] = f

                            fs_n.append(f_n)
                            self.cnames2fis[fn].add(f_n)
                        multifeats_p.append(fs_n)
                    else:
                        if fn == "lemma":
                            # Lowercase lemmas
                            f = f.lower()
                        if not f:
                            f = UNK_TOKEN
                        else:
                            f = f"{fn}:{f}"
                        if f not in self.fstoi:
                            new_n = len(self.fstoi)
                            self.fstoi[f] = new_n
                            self.fitos[new_n] = f
                        f_n = self.fstoi[f]
                        feats_p.append(f_n)
                        # Update feature name
                        self.cnames2fis[fn].add(f_n)
                doc_words.append(word_n)
                doc_feats.append(feats_p)
                doc_multifeats.append(multifeats_p)
            self.docs.append((doc_words, doc_feats, doc_multifeats))

    def is_open(self, feat):
        return self.ctypes[feat] == "open"

    def is_closed(self, feat):
        return self.ctypes[feat] == "closed"

    def is_multi(self, feat):
        return self.ctypes[feat] == "multi"

    def __getitem__(self, i):
        ws = torch.as_tensor(self.words[i]).unsqueeze(1)
        fs = torch.as_tensor(self.fs[i])
        # These are multi-hot vectors with width equal to the number of
        # multi-hot features.
        mfs = torch.zeros((len(self.words[i]), len(self.mfis)), dtype=torch.bool)
        for w_i, word_mfeatures in enumerate(self.mfs[i]):
            for cat_f in word_mfeatures:
                for mfi in cat_f:
                    # Map to index
                    mfi_index = self.multi2idx[mfi]
                    mfs[w_i, mfi_index] = 1
        ls = torch.tensor(self.lengths[i])
        return ws, fs, mfs, ls, i

    def compute_cat_names(self, feat_str):
        cat_names = feat_str.strip().split("|")
        cat_names, ocs = zip(*[fn.split(";") for fn in cat_names])
        for oc in ocs:
            if oc not in {"open", "closed", "multi"}:
                raise RuntimeError(f"Invalid tag {oc}")
        ocn = [cn for cn, oc in zip(cat_names, ocs) if oc == "open"]
        ccn = [cn for cn, oc in zip(cat_names, ocs) if oc == "closed"]
        mcn = [cn for cn, oc in zip(cat_names, ocs) if oc == "multi"]
        cat_types = dict(zip(cat_names, ocs))
        cat_is = list(range(len(cat_names)))
        return cat_is, cat_names, cat_types, ocn, ccn, mcn

    def __len__(self):
        return len(self.words)

    def to_text(self, sentence):
        words = []
        for tok in sentence:
            word = self.itos[tok]
            if word == "<unk>":
                word = "UNK"
            if word not in self.ignore_tokens:
                words.append(word)
        return words

    def to_text_batch(self, src):
        batch_size = src.shape[1]
        batch = []
        for i in range(batch_size):
            sentence = src[:, i, :].squeeze(1)
            words = []
            for tok in sentence.cpu().numpy():
                word = self.itos[tok]
                if word == "<unk>":
                    word = UNK_TOKEN
                if word not in self.ignore_tokens:
                    words.append(word)
            batch.append(words)
        return batch

    def name_feat(self, i):
        return self.fitos.get(i, UNK_TOKEN)
