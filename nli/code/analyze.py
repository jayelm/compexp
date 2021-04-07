#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import multiprocessing as mp
import os
from collections import Counter, defaultdict

import numpy as np
import onmt.opts as opts
import pandas as pd
import torch
from onmt.utils.parse import ArgumentParser
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import precision_score, recall_score

import formula as FM
import settings
import util
from vis import report, pred_report
import data
import data.snli
import data.analysis


GLOBALS = {}


def save_with_acts(preds, acts, fname):
    preds_to_save = preds.copy()
    for i in range(acts.shape[1]):
        preds_to_save[str(i)] = acts[:, i] * 1
    preds_to_save.to_csv(fname, index=False)


def load_vecs(path):
    vecs = []
    vecs_stoi = {}
    vecs_itos = {}
    with open(path, "r") as f:
        for line in f:
            tok, *nums = line.split(" ")
            nums = np.array(list(map(float, nums)))

            assert tok not in vecs_stoi
            new_n = len(vecs_stoi)
            vecs_stoi[tok] = new_n
            vecs_itos[new_n] = tok
            vecs.append(nums)
    vecs = np.array(vecs)
    return vecs, vecs_stoi, vecs_itos


# Load vectors
VECS, VECS_STOI, VECS_ITOS = load_vecs(settings.VECPATH)


NEIGHBORS_CACHE = {}


def get_neighbors(lemma):
    """
    Get neighbors of lemma given glove vectors.
    """
    if lemma not in VECS_STOI:
        # No neighbors
        return []
    if lemma in NEIGHBORS_CACHE:
        return NEIGHBORS_CACHE[lemma]
    lemma_i = VECS_STOI[lemma]
    lvec = VECS[lemma_i][np.newaxis]
    dists = cdist(lvec, VECS, metric="cosine")[0]
    # first dist will always be the vector itself
    nearest_i = np.argsort(dists)[1 : settings.EMBEDDING_NEIGHBORHOOD_SIZE + 1]
    nearest = [VECS_ITOS[i] for i in nearest_i]
    NEIGHBORS_CACHE[lemma] = nearest
    return nearest


def get_mask(feats, f, dataset, feat_type):
    """
    Serializable/global version of get_mask for multiprocessing
    """
    # Mask has been cached
    if f.mask is not None:
        return f.mask
    if isinstance(f, FM.And):
        masks_l = get_mask(feats, f.left, dataset, feat_type)
        masks_r = get_mask(feats, f.right, dataset, feat_type)
        return masks_l & masks_r
    elif isinstance(f, FM.Or):
        masks_l = get_mask(feats, f.left, dataset, feat_type)
        masks_r = get_mask(feats, f.right, dataset, feat_type)
        return masks_l | masks_r
    elif isinstance(f, FM.Not):
        masks_val = get_mask(feats, f.val, dataset, feat_type)
        return 1 - masks_val
    elif isinstance(f, FM.Neighbors):
        if feat_type == "word":
            # Neighbors can only be called on Lemma Leafs. Can they be called on
            # ORs of Lemmas? (NEIGHBORS(A or B))? Is this equivalent to
            # NEIGHBORS(A) or NEIGHBORS(B)?
            # (When doing search, you should do unary nodes that apply first,
            # before looping through binary nodes)
            # Can this only be done on an atomic leaf? What are NEIGHBORS(
            # (1) GET LEMMAS belonging to the lemma mentioned by f;
            # then search for other LEMMAS; return a mask that is 1 for all of
            # those lemmas.
            # We can even do NEIGHBORS(NEIGHBORS) by actually looking at where the
            # masks are 1s...but I wouldskip that for now
            # FOR NOW - just do N nearest neighbors?
            # TODO: Just pass in the entire dataset.
            # The feature category should be lemma
            # Must call neighbors on a leaf
            assert isinstance(f.val, FM.Leaf)
            ci = dataset.fis2cis[f.val.val]
            assert dataset.citos[ci] == "lemma"

            # The feature itself should be a lemma
            full_fname = dataset.fitos[f.val.val]
            assert full_fname.startswith("lemma:")
            # Get the actual lemma
            fname = full_fname[6:]

            # Get neighbors in vector space
            neighbors = get_neighbors(fname)
            # Turn neighbors into candidate feature names
            neighbor_fnames = set([f"lemma:{word}" for word in neighbors])
            # Add the original feature name
            neighbor_fnames.add(full_fname)
            # Convert to indices if they exist
            neighbors = [
                dataset.fstoi[fname]
                for fname in neighbor_fnames
                if fname in dataset.fstoi
            ]
            return np.isin(feats["onehot"][:, ci], neighbors)
        else:
            assert isinstance(f.val, FM.Leaf)
            fval = f.val.val
            fname = dataset["itos"][fval]
            part, fword = fname.split(":", maxsplit=1)

            neighbors = get_neighbors(fword)
            part_neighbors = [f"{part}:{word}" for word in neighbors]
            neighbor_idx = [
                dataset["stoi"][word]
                for word in part_neighbors
                if word in dataset["stoi"]
            ]
            neighbor_idx.append(fval)
            neighbor_idx = np.array(list(set(neighbor_idx)))

            neighbors_mask = np.logical_or.reduce(feats[:, neighbor_idx], 1)
            return neighbors_mask
    elif isinstance(f, FM.Leaf):
        if feat_type == "word":
            # Get category
            ci = dataset.fis2cis[f.val]
            cname = dataset.fis2cnames[f.val]
            if dataset.ctypes[cname] == "multi":
                # multi is in n-hot tensor shape, so we just return the column
                # corresponding to the correct feature
                midx = dataset.multi2idx[f.val]
                return feats["multi"][:, midx]
            else:
                return feats["onehot"][:, ci] == f.val
        else:
            return feats[:, f.val]
    else:
        raise ValueError("Most be passed formula")


def iou(a, b):
    intersection = (a & b).sum()
    union = (a | b).sum()
    return intersection / (union + np.finfo(np.float32).tiny)


def get_max_ofis(states, feats, dataset):
    """
    Get maximally activated open feats
    """
    max_order = np.argsort(states)[::-1]
    sel_ofeats = []
    for ocname in dataset.ocnames:
        ci = dataset.cstoi[ocname]
        ofeats = feats["onehot"][:, ci]
        max_ofeats = ofeats[max_order]
        max_ofeats = max_ofeats[max_ofeats != 0]
        # pd preserves order
        unique_ofeats = pd.unique(max_ofeats)
        sel_ofeats.extend(unique_ofeats[: settings.MAX_OPEN_FEATS])
    return sel_ofeats


# Category-specific composition operators
# are tuples of the shape (op, do_negate)
OPS = defaultdict(
    list,
    {
        "all": [(FM.Or, False), (FM.And, False), (FM.And, True)],
        "lemma": [(FM.Neighbors, False)],
        # WordNet synsets. For now just do hypernyms? Note: for hypernyms - how far
        # up to go? Go too far = activates for all synsets. Too low = ?
        #  'synset': [(FM.Hypernym, False)],
        # NOTE: Will beam search even work? Can I even do "compounds"? I.e. if I
        # have synset OR synset, will I ever explore synset OR hyponyms(synset)?
        # ALSO: don't forget glove vectors
    },
)


def compute_iou(formula, acts, feats, dataset, feat_type="word"):
    masks = get_mask(feats, formula, dataset, feat_type)
    # Cache mask
    formula.mask = masks

    if settings.METRIC == "iou":
        comp_iou = iou(masks, acts)
    elif settings.METRIC == "precision":
        comp_iou = precision_score(masks, acts)
    elif settings.METRIC == "recall":
        comp_iou = recall_score(masks, acts)
    else:
        raise NotImplementedError(f"metric: {settings.METRIC}")
    comp_iou = (settings.COMPLEXITY_PENALTY ** (len(formula) - 1)) * comp_iou

    return comp_iou


def compute_best_word_iou(args):
    (unit,) = args

    acts = GLOBALS["acts"][:, unit]
    feats = GLOBALS["feats"]
    states = GLOBALS["states"][:, unit]
    dataset = GLOBALS["dataset"]

    # Start search with closed feats + maximally activated open feats
    search_ofis = get_max_ofis(states, feats, dataset)
    # Add closed + multi feats
    feats_to_search = dataset.cfis + dataset.mfis + search_ofis
    formulas = {}
    for fval in feats_to_search:
        formula = FM.Leaf(fval)
        formulas[formula] = compute_iou(formula, acts, feats, dataset, feat_type="word")

        # Try unary ops
        fcat = dataset.fis2cnames[fval]
        for op, negate in OPS[fcat]:
            # FIXME: Don't evaluate on neighbors if they don't exist
            new_formula = formula
            if negate:
                new_formula = FM.Not(new_formula)
            new_formula = op(new_formula)
            new_iou = compute_iou(new_formula, acts, feats, dataset, feat_type="word")
            formulas[new_formula] = new_iou

    formulas = dict(Counter(formulas).most_common(settings.BEAM_SIZE))
    best_noncomp = Counter(formulas).most_common(1)[0]

    for i in range(settings.MAX_FORMULA_LENGTH - 1):
        new_formulas = {}
        for formula in formulas:
            # Unary ops if the current formula is a leaf
            # NOTE: This is now redundant since leaf formulas will have been
            # accessed already.
            # Here you shoudl make decisions about e.g. "neighbors of
            # neighbors" or something like that. SPECIFICALLY, maybe neighbors
            # should be treated the same as negates?
            #  if formula.is_leaf():
            #  fcat = dataset.fis2cnames[formula.val]
            #  for op, negate in OPS[fcat]:
            #  new_formula = formula
            #  if negate:
            #  new_formula = FM.Not(new_formula)
            #  new_formula = op(new_formula)
            #  new_iou = compute_iou(new_formula, acts, feats, dataset, feat_type='word')
            #  new_formulas[new_formula] = new_iou

            # Generic binary ops
            for feat in feats_to_search:
                for op, negate in OPS["all"]:
                    new_formula = FM.Leaf(feat)
                    if negate:
                        new_formula = FM.Not(new_formula)
                    new_formula = op(formula, new_formula)
                    new_iou = compute_iou(
                        new_formula, acts, feats, dataset, feat_type="word"
                    )
                    new_formulas[new_formula] = new_iou

        formulas.update(new_formulas)
        # Trim the beam
        formulas = dict(Counter(formulas).most_common(settings.BEAM_SIZE))

    best = Counter(formulas).most_common(1)[0]

    return {
        "unit": unit,
        "best": best,
        "best_noncomp": best_noncomp,
    }


def compute_best_sentence_iou(args):
    (unit,) = args

    acts = GLOBALS["acts"][:, unit]
    feats = GLOBALS["feats"]
    dataset = GLOBALS["dataset"]

    if acts.sum() < settings.MIN_ACTS:
        null_f = (FM.Leaf(0), 0)
        return {"unit": unit, "best": null_f, "best_noncomp": null_f}

    feats_to_search = list(range(feats.shape[1]))
    formulas = {}
    for fval in feats_to_search:
        formula = FM.Leaf(fval)
        formulas[formula] = compute_iou(
            formula, acts, feats, dataset, feat_type="sentence"
        )

        for op, negate in OPS["lemma"]:
            # FIXME: Don't evaluate on neighbors if they don't exist
            new_formula = formula
            if negate:
                new_formula = FM.Not(new_formula)
            new_formula = op(new_formula)
            new_iou = compute_iou(
                new_formula, acts, feats, dataset, feat_type="sentence"
            )
            formulas[new_formula] = new_iou

    nonzero_iou = [k.val for k, v in formulas.items() if v > 0]
    formulas = dict(Counter(formulas).most_common(settings.BEAM_SIZE))
    best_noncomp = Counter(formulas).most_common(1)[0]

    for i in range(settings.MAX_FORMULA_LENGTH - 1):
        new_formulas = {}
        for formula in formulas:
            # Generic binary ops
            for feat in nonzero_iou:
                for op, negate in OPS["all"]:
                    if not isinstance(feat, FM.F):
                        new_formula = FM.Leaf(feat)
                    else:
                        new_formula = feat
                    if negate:
                        new_formula = FM.Not(new_formula)
                    new_formula = op(formula, new_formula)
                    new_iou = compute_iou(
                        new_formula, acts, feats, dataset, feat_type="sentence"
                    )
                    new_formulas[new_formula] = new_iou

        formulas.update(new_formulas)
        # Trim the beam
        formulas = dict(Counter(formulas).most_common(settings.BEAM_SIZE))

    best = Counter(formulas).most_common(1)[0]

    return {
        "unit": unit,
        "best": best,
        "best_noncomp": best_noncomp,
    }


def pad_collate(batch, sort=True):
    src, src_feats, src_multifeats, src_len, idx = zip(*batch)
    idx = torch.tensor(idx)
    src_len = torch.tensor(src_len)
    src_pad = pad_sequence(src, padding_value=data.analysis.PAD_IDX)
    # NOTE: part of speeches are padded with 0 - we don't actually care here
    src_feats_pad = pad_sequence(src_feats, padding_value=-1)
    src_multifeats_pad = pad_sequence(src_multifeats, padding_value=-1)

    if sort:
        src_len_srt, srt_idx = torch.sort(src_len, descending=True)
        src_pad_srt = src_pad[:, srt_idx]
        src_feats_pad_srt = src_feats_pad[:, srt_idx]
        src_multifeats_pad_srt = src_multifeats_pad[:, srt_idx]
        idx_srt = idx[srt_idx]
        return (
            src_pad_srt,
            src_feats_pad_srt,
            src_multifeats_pad_srt,
            src_len_srt,
            idx_srt,
        )
    return src_pad, src_feats_pad, src_multifeats_pad, src_len, idx


def pairs(x):
    """
    (max_len, batch_size, *feats)
    -> (max_len, batch_size / 2, 2, *feats)
    """
    if x.ndim == 1:
        return x.unsqueeze(1).view(-1, 2)
    else:
        return x.unsqueeze(2).view(x.shape[0], -1, 2, *x.shape[2:])


def extract_features(
    model,
    dataset,
):
    loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=32,
        collate_fn=lambda batch: pad_collate(batch, sort=False),
    )

    all_srcs = []
    all_states = []
    all_feats = []
    all_multifeats = []
    all_idxs = []
    for src, src_feats, src_multifeats, src_lengths, idx in tqdm(loader):
        #  words = dataset.to_text(src)
        if settings.CUDA:
            src = src.cuda()
            src_lengths = src_lengths.cuda()
        # Memory bank - hidden states for each step
        with torch.no_grad():
            # Combine q/h pairs
            src_one = src.squeeze(2)
            src_one_comb = pairs(src_one)
            src_lengths_comb = pairs(src_lengths)

            s1 = src_one_comb[:, :, 0]
            s1len = src_lengths_comb[:, 0]

            s2 = src_one_comb[:, :, 1]
            s2len = src_lengths_comb[:, 1]

            final_reprs = model.get_final_reprs(s1, s1len, s2, s2len)

        # Pack the sequence
        all_srcs.extend(list(np.transpose(src_one_comb.cpu().numpy(), (1, 2, 0))))
        all_feats.extend(
            list(np.transpose(pairs(src_feats).cpu().numpy(), (1, 2, 0, 3)))
        )
        all_multifeats.extend(
            list(np.transpose(pairs(src_multifeats).cpu().numpy(), (1, 2, 0, 3)))
        )
        all_states.extend(list(final_reprs.cpu().numpy()))
        all_idxs.extend(list(pairs(idx).cpu().numpy()))

    all_feats = {"onehot": all_feats, "multi": all_multifeats}

    return all_srcs, all_states, all_feats, all_idxs


def get_quantiles(feats, alpha):
    quantiles = np.apply_along_axis(lambda a: np.quantile(a, 1 - alpha), 0, feats)
    return quantiles


def quantile_features(feats):
    if settings.ALPHA is None:
        return np.stack(feats) > 0

    quantiles = get_quantiles(feats, settings.ALPHA)
    return feats > quantiles[np.newaxis]


def search_feats(acts, states, feats, weights, dataset):
    rfile = os.path.join(settings.RESULT, "result.csv")
    if os.path.exists(rfile):
        print(f"Loading cached {rfile}")
        return pd.read_csv(rfile).to_dict("records")

    # Set global vars
    GLOBALS["acts"] = acts
    GLOBALS["states"] = states

    GLOBALS["feats"] = feats[0]
    GLOBALS["dataset"] = feats[1]
    feats_vocab = feats[1]

    def namer(i):
        return feats_vocab["itos"][i]

    def cat_namer(i):
        return feats_vocab["itos"][i].split(":")[0]

    def cat_namer_fine(i):
        return ":".join(feats_vocab["itos"][i].split(":")[:2])

    ioufunc = compute_best_sentence_iou

    records = []
    if settings.NEURONS is None:
        units = range(acts.shape[1])
    else:
        units = settings.NEURONS
    mp_args = [(u,) for u in units]

    if settings.PARALLEL < 1:
        pool_cls = util.FakePool
    else:
        pool_cls = mp.Pool

    n_done = 0
    with pool_cls(settings.PARALLEL) as pool, tqdm(
        total=len(units), desc="Units"
    ) as pbar:
        for res in pool.imap_unordered(ioufunc, mp_args):
            unit = res["unit"]
            best_lab, best_iou = res["best"]
            best_name = best_lab.to_str(namer, sort=True)
            best_cat = best_lab.to_str(cat_namer, sort=True)
            best_cat_fine = best_lab.to_str(cat_namer_fine, sort=True)

            entail_weight = weights[unit, 0]
            neutral_weight = weights[unit, 1]
            contra_weight = weights[unit, 2]

            if best_iou > 0:
                tqdm.write(f"{unit:02d}\t{best_name}\t{best_iou:.3f}")
            r = {
                "neuron": unit,
                "feature": best_name,
                "category": best_cat,
                "category_fine": best_cat_fine,
                "iou": best_iou,
                "feature_length": len(best_lab),
                "w_entail": entail_weight,
                "w_neutral": neutral_weight,
                "w_contra": contra_weight,
            }
            records.append(r)
            pbar.update()
            n_done += 1
            if n_done % settings.SAVE_EVERY == 0:
                pd.DataFrame(records).to_csv(rfile, index=False)

        # Save progress
        if len(records) % 32 == 0:
            pd.DataFrame(records).to_csv(rfile, index=False)

    pd.DataFrame(records).to_csv(rfile, index=False)
    return records


def to_sentence(toks, feats, dataset, tok_feats_vocab=None):
    """
    Convert token-level feats to sentence feats
    """
    tokens = np.zeros(len(dataset.stoi), dtype=np.int64)
    encoder_uniques = []
    decoder_uniques = []
    #  both_uniques = []

    encoder_tag_uniques = []
    decoder_tag_uniques = []
    #  both_tag_uniques = []

    tag_i = dataset.cstoi["tag"]

    other_features = []
    oth_names = [
        ("overlap25", "overlap"),
        ("overlap50", "overlap"),
        ("overlap75", "overlap"),
    ]

    for pair, featpair in zip(toks, feats["onehot"]):
        pair_counts = np.bincount(pair.ravel())
        tokens[: len(pair_counts)] += pair_counts

        enct = np.unique(pair[0])
        dect = np.unique(pair[1])

        encu = np.setdiff1d(enct, dect)
        decu = np.setdiff1d(dect, enct)
        both = np.intersect1d(enct, dect)
        encoder_uniques.append(enct)
        decoder_uniques.append(dect)
        #  both_uniques.append(both)

        # PoS
        enctag = np.unique(featpair[0, :, tag_i])
        dectag = np.unique(featpair[1, :, tag_i])

        enctag = enctag[enctag != -1]
        dectag = dectag[dectag != -1]

        #  enctagu = np.setdiff1d(enctag, dectag)
        #  dectagu = np.setdiff1d(dectag, enctag)
        #  bothtagu = np.intersect1d(enctag, dectag)

        encoder_tag_uniques.append(enctag)
        decoder_tag_uniques.append(dectag)
        #  both_tag_uniques.append(bothtagu)

        # Compute degree of overlap in tokens (gt 50%)
        overlap = len(both) / (len(encu) + len(decu) + 1e-5)
        # TODO: Do overlap at various degrees
        other_features.append(
            (
                overlap > 0.25,
                overlap > 0.5,
                overlap > 0.75,
            )
        )

    SKIP = {"a", "an", "the", "of", ".", ",", "UNK", "PAD"}
    if tok_feats_vocab is None:
        for s in SKIP:
            if s in dataset.stoi:
                tokens[dataset.stoi[s]] = 0

        # Keep top tokens, use as features
        tokens_by_count = np.argsort(tokens)[::-1]
        tokens_by_count = tokens_by_count[: settings.N_SENTENCE_FEATS]

        # Create feature dict
        # Token features
        tokens_stoi = {}
        for prefix in ["pre", "hyp"]:
            for t in tokens_by_count:
                ts = dataset.itos[t]
                t_prefixed = f"{prefix}:tok:{ts}"
                tokens_stoi[t_prefixed] = len(tokens_stoi)

            # PoS
            for pos_i in dataset.cnames2fis["tag"]:
                pos = dataset.fitos[pos_i].lower()
                assert pos.startswith("tag:")
                pos_prefixed = f"{prefix}:{pos}"
                tokens_stoi[pos_prefixed] = len(tokens_stoi)

        # Other features
        for oth, oth_type in oth_names:
            oth_prefixed = f"oth:{oth_type}:{oth}"
            tokens_stoi[oth_prefixed] = len(tokens_stoi)

        tokens_itos = {v: k for k, v in tokens_stoi.items()}

        tok_feats_vocab = {
            "itos": tokens_itos,
            "stoi": tokens_stoi,
        }

    # Binary mask - encoder/decoder
    token_masks = np.zeros((len(toks), len(tok_feats_vocab["stoi"])), dtype=np.bool)
    for i, (encu, decu, enctagu, dectagu, oth) in enumerate(
        zip(
            encoder_uniques,
            decoder_uniques,
            encoder_tag_uniques,
            decoder_tag_uniques,
            other_features,
        )
    ):
        # Tokens
        for prefix, toks in [("pre", encu), ("hyp", decu)]:
            for t in toks:
                ts = dataset.itos[t]
                t_prefixed = f"{prefix}:tok:{ts}"
                if t_prefixed in tok_feats_vocab["stoi"]:
                    ti = tok_feats_vocab["stoi"][t_prefixed]
                    token_masks[i, ti] = 1

        # PoS
        for prefix, tags in [("pre", enctagu), ("hyp", dectagu)]:
            for t in tags:
                ts = dataset.fitos[t].lower()
                t_prefixed = f"{prefix}:{ts}"
                assert t_prefixed in tok_feats_vocab["stoi"]
                ti = tok_feats_vocab["stoi"][t_prefixed]
                token_masks[i, ti] = 1

        # Other features
        assert len(oth) == len(oth_names)
        for (oth_name, oth_type), oth_u in zip(oth_names, oth):
            oth_prefixed = f"oth:{oth_type}:{oth_name}"
            oi = tok_feats_vocab["stoi"][oth_prefixed]
            token_masks[i, oi] = oth_u

    return token_masks, tok_feats_vocab


def main():
    os.makedirs(settings.RESULT, exist_ok=True)

    print("Loading model/vocab")
    model, dataset = data.snli.load_for_analysis(
        settings.MODEL,
        settings.DATA,
        model_type=settings.MODEL_TYPE,
        cuda=settings.CUDA,
    )

    # Last model weight
    if settings.MODEL_TYPE == "minimal":
        weights = model.mlp.weight.t().detach().cpu().numpy()
    else:
        weights = model.mlp[-1].weight.t().detach().cpu().numpy()

    print("Extracting features")
    toks, states, feats, idxs = extract_features(
        model,
        dataset,
    )

    print("Computing quantiles")
    acts = quantile_features(states)

    print("Extracting sentence token features")
    tok_feats, tok_feats_vocab = to_sentence(toks, feats, dataset)
    print("Mask search")
    records = search_feats(acts, states, (tok_feats, tok_feats_vocab), weights, dataset)

    print("Mask search")
    records = search_feats(acts, states, feats, weights, dataset)

    print("Load predictions")
    mbase = os.path.splitext(os.path.basename(settings.MODEL))[0]
    dbase = os.path.splitext(os.path.basename(settings.DATA))[0]
    predf = f"data/analysis/preds/{mbase}_{dbase}.csv"
    # Add the feature activations so we can do correlation
    preds = pd.read_csv(predf)

    save_with_acts(preds, acts, os.path.join(settings.RESULT, "preds_acts.csv"))

    print("Visualizing features")
    from vis import sentence_report

    sentence_report.make_html(
        records,
        # Features
        toks,
        states,
        (tok_feats, tok_feats_vocab),
        idxs,
        preds,
        # General stuff
        weights,
        dataset,
        settings.RESULT,
    )


if __name__ == "__main__":
    main()
