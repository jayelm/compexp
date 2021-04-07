"""
Visualize predictions
"""

from collections import Counter, defaultdict
import pandas as pd

from . import common as c
from .report import make_spans, orig_val, to_ul
import os
import numpy as np


PRED_CARD_HTML = """
<div class="card unit {maybe_correct}" data-i="{idx}" data-correct="{correct}">
  <div class="card-body">
    {toolbar}
    {items}
    <p class="gt"><strong>True</strong>: {gt}</p>
    <p class="pred"><strong>Pred</strong>: {pred}</p>
  </div>
</div>
"""


def get_feats_single_sentence(words, feat, mfeat, dataset):
    feats = []
    for i, w in enumerate(words):
        these_feats = feat[i]
        these_mfeats = mfeat[i]
        feats.append(
            {
                **{
                    cname: orig_val(dataset.name_feat(these_feats[ci]))
                    for ci, cname in enumerate(dataset.cnames)
                    if not dataset.is_multi(cname)
                },
                **{
                    mcname: ";".join(
                        [
                            orig_val(dataset.name_feat(f))
                            for f in dataset.cnames2fis[mcname]
                            if these_mfeats[dataset.multi2idx[f]]
                        ]
                    )
                    for mcname in dataset.mcnames
                },
            }
        )
    return feats


def keep_highest(acts, n_mean=5, n_max=5):
    """
    Keep neurons with highest activations
    """
    highest_means = Counter()
    highest_maxes = Counter()
    # 5 means, 5 maxes
    for i in range(acts[0].shape[0]):
        act = [a[i] for a in acts]
        highest_means[i] = np.mean(act)
        highest_maxes[i] = np.max(act)
    best = highest_means.most_common(n_mean) + highest_maxes.most_common(n_max)
    best = np.array([b[0] for b in best])
    acts_best = [a[best] for a in acts]
    acts_best = [[(b, a) for a, b in zip(ac, best)] for ac in acts_best]
    return acts_best, best


def make_highest(x, records):
    record_spans = []
    for neuron in x:
        try:
            label = records[neuron]["feature"]
            iou = records[neuron]["iou"]
            entail = records[neuron]["w_entail"]
            neutral = records[neuron]["w_neutral"]
            contra = records[neuron]["w_contra"]
        except KeyError:
            label = "UNK"
            iou = -1
            entail = -1
            neutral = -1
            contra = -1
        record_spans.append(
            f"<span class='neuron' data-neuron='{neuron}'>{neuron} ({label}: {iou:.3f} entail: {entail:.3f} neutral: {neutral:.3f} contra: {contra:.3f})</span>"
        )
    return "".join(record_spans)


def make_card(records, i, tok, state, feat, mfeat, pred, gt, correct, dataset):
    pre_words = dataset.to_text(tok[0])
    pre_acts = [a for a, _ in zip(state[0], pre_words)]
    pre_feats = get_feats_single_sentence(pre_words, feat[0], mfeat[0], dataset)
    pre_acts, pre_highest = keep_highest(pre_acts)

    hyp_words = dataset.to_text(tok[1])
    hyp_acts = [a for a, _ in zip(state[1], hyp_words)]
    hyp_acts, hyp_highest = keep_highest(hyp_acts)
    hyp_feats = get_feats_single_sentence(hyp_words, feat[1], mfeat[1], dataset)

    np.sort(pre_highest)
    np.sort(hyp_highest)
    pre_highest_html = make_highest(pre_highest, records)
    hyp_highest_html = make_highest(hyp_highest, records)
    pre_highest_html = f"<div class='pre-highest-toolbar'>{pre_highest_html}</div>"
    hyp_highest_html = f"<div class='hyp-highest-toolbar'>{hyp_highest_html}</div>"
    pre = make_spans(pre_words, pre_acts, pre_feats, multiact=True)
    pre = f"<div class='premise'>{pre}</div>"
    hyp = make_spans(hyp_words, hyp_acts, hyp_feats, multiact=True)
    hyp = f"<div class='premise'>{hyp}</div>"
    items = to_ul([pre, hyp])
    card_fmt = PRED_CARD_HTML.format(
        idx=i,
        items=items,
        gt=gt,
        pred=pred,
        correct=correct * 1,
        toolbar=f"<div class='toolbar'>Pre: {pre_highest_html} Hyp: {hyp_highest_html}</div>",
        maybe_correct="correct" if correct else "incorrect",
    )
    return card_fmt, np.concatenate([pre_highest, hyp_highest])


def make_html(
    records,
    toks,
    states,
    feats,
    idxs,
    preds,
    weights,
    dataset,
    result_dir,
    filename="pred.html",
):
    # Convert records to dict by unit
    records = {r["neuron"]: r for r in records}
    html = [c.HTML_PREFIX]
    html_dir = os.path.join(result_dir, "html")
    os.makedirs(html_dir, exist_ok=True)

    # Here, we loop through data points
    wrong_v_right = defaultdict(Counter)
    for i, pred_df in preds.iterrows():
        gt = pred_df["gt"]
        pred = pred_df["pred"]
        correct = pred_df["correct"]
        orig_idx = i * 2
        tok = toks[orig_idx : orig_idx + 2]
        state = states[orig_idx : orig_idx + 2]
        feat = feats["onehot"][orig_idx : orig_idx + 2]
        mfeat = feats["multi"][orig_idx : orig_idx + 2]
        card_html, highest = make_card(
            records, i, tok, state, feat, mfeat, pred, gt, correct, dataset
        )
        for val in highest:
            wrong_v_right[val][correct] += 1
        if i < 100:
            html.append(card_html)

    html.append(c.HTML_SUFFIX)
    html_final = "\n".join(html)
    with open(os.path.join(html_dir, filename), "w") as f:
        f.write(html_final)

    # Bonus - wrong vs right - normalize
    units = list(wrong_v_right.keys())
    correct_p = [
        wrong_v_right[u][True] / (sum(wrong_v_right[u].values()) + 0.01) for u in units
    ]
    correct_df = pd.DataFrame({"neuron": units, "p_correct": correct_p})
    correct_df.sort_values("p_correct", ascending=False, inplace=True)
    correct_df.to_csv(os.path.join(result_dir, "p_correct.csv"), index=False)
