"""
Visualize predictions for sentences
"""

from collections import Counter, defaultdict
import pandas as pd

from . import common as c
from .report import to_ul
import formula as FM
import settings
import os
import numpy as np
from analyze import get_mask
import pyparsing as pp
from scipy.stats import entropy
from sklearn.metrics import confusion_matrix
from data.snli import LABEL_STOI


def make_cm_html(preds_where_true, snli_entropy):
    if preds_where_true.shape[0] == 0:
        cm = np.zeros((3, 3), dtype=np.int64)
        snli_acc = np.nan
    else:
        cm = confusion_matrix(
            preds_where_true["gt"],
            preds_where_true["pred"],
            labels=list(LABEL_STOI.keys()),
        )
        snli_acc = np.diagonal(cm).sum() / cm.sum()

    cm_html = cm_to_table(cm, f"SNLI (Acc: {snli_acc:.2f} Entropy: {snli_entropy:.2f})")
    return f"""
    <div class="cm-section">
    {cm_html}
    </div>
    """


def cm_to_table(cm, title):
    return c.CM_TABLE.format(title, *cm.ravel())


def pred_entropy(preds_df):
    gt_counts = Counter(preds_df["gt"])
    # Normalize
    vals = np.array(
        [
            gt_counts.get("entailment", 0),
            gt_counts.get("neutral", 0),
            gt_counts.get("contradiction", 0),
        ]
    )
    vals = vals / (sum(gt_counts.values()) + 0.01)
    return entropy(vals)


def make_spans(text, tok_feats_vocab):
    spans = []
    for word in text:
        span = f"<span class='word' data-act='0'>{c.unquote(word)}</span>"
        spans.append(span)
    return "".join(spans)


def make_spans_pair(act, pair, pred_df, tok_feats_vocab, dataset):
    pre = dataset.to_text(pair[0])
    hyp = dataset.to_text(pair[1])

    # Better scaling for highlighting
    if act == 0:
        actlabel = -1
    else:
        actlabel = act / 10

    pre_spans = make_spans(pre, tok_feats_vocab)
    hyp_spans = make_spans(hyp, tok_feats_vocab)
    pre_spans = f"<div class='pre'><strong>Premise:</strong> {pre_spans}</div>"
    hyp_spans = f"<div class='hyp'><strong>Hypothesis:</strong> {hyp_spans}</div>"
    info = f"<div class='pairinfo'><span class='act'><strong>ACT</strong> <span class='word' data-act='{actlabel:.2f}'>{act:.2f}</span></span> <strong>GT</strong> <span class='word {pred_df['gt']}'>{pred_df['gt']}</span> <strong>PRED</strong> <span class='word {pred_df['pred']}'>{pred_df['pred']}</span></div>"
    return f"<div class='pair'>{pre_spans}{hyp_spans}{info}</div>"


def make_examples(idx, toks, states, preds, tok_feats_vocab, dataset):
    pairs = [(states[t], toks[t], preds.iloc[t]) for t in idx]
    pairs = [
        make_spans_pair(*pairstuff, tok_feats_vocab, dataset) for pairstuff in pairs
    ]
    return to_ul(pairs)


def combine_examples(example_cats, i):
    """
    Combine into an accordion
    """
    accordion_id = f"{i}-accordion"
    html = []
    for j, (title, h) in enumerate(example_cats):
        this_id = f"{i}-{j}"
        accordion_card = c.ACCORDION_MEMBER.format(
            title=title, id=this_id, accordion_id=accordion_id, body=h
        )
        html.append(accordion_card)
    return c.ACCORDION.format(accordion_id=accordion_id, body="".join(html))


def make_card(
    record,
    toks,
    tok_feats,
    tok_feats_vocab,
    states,
    preds,
    dataset,
):
    # Get highest activation
    neuron = record["neuron"]

    # Get states for this neuron only
    states = states[:, neuron]

    # Count how often neuron activates. If it's for less than 5 examples across 10k, discard it
    if (states > 0).sum() < settings.MIN_ACTS:
        return ""

    # Highest activations pairs
    top_act = np.argsort(states)[::-1][: settings.TOPN]
    top_act_html = make_examples(top_act, toks, states, preds, tok_feats_vocab, dataset)

    # Combine example categories
    examples = [
        ("Highest Act (SNLI)", top_act_html),
    ]

    def reverse_namer(fstr):
        return tok_feats_vocab["stoi"][fstr]

    try:
        feature_f = FM.parse(record["feature"], reverse_namer)
    except pp.ParseException as p:
        # Some I can't parse
        # FIXME: if you want this to be cacahble, you need to be able to get
        # back from text formula to real formula....or actually cache the masks
        print(p)
        print(f"Couldn't parse {record['feature']}")
        feature_f = None

    if feature_f is not None:
        feature_mask = get_mask(tok_feats, feature_f, tok_feats_vocab, "sentence")
        sel_mask = np.argwhere(feature_mask).squeeze(1)[: settings.TOPN]
        mask_html = make_examples(
            sel_mask, toks, states, preds, tok_feats_vocab, dataset
        )

        examples.append(
            ("Other examples of mask (SNLI)", mask_html),
        )

        # Get statistics for how well this formula "partitions" the dataset
        # i.e. what are the distribution of data points for formulas where this is true?
        preds_where_true = preds.iloc[np.argwhere(feature_mask).squeeze(1)]

        snli_entropy = pred_entropy(preds_where_true)
        cm_html = make_cm_html(preds_where_true, snli_entropy)
    else:
        snli_entropy = 0.0
        cm_html = ""

    all_examples_html = combine_examples(examples, record["neuron"])
    all_examples_html = f"{all_examples_html}{cm_html}"

    fmt = c.SCARD_HTML.format(
        unit=record["neuron"],
        iou=f"{record['iou']:.3f}",
        category=record["category"],
        label=record["feature"],
        entail=record["w_entail"],
        neutral=record["w_neutral"],
        contra=record["w_contra"],
        snli_entropy=snli_entropy,
        title=f"Unit {record['neuron']} {record['feature']}",
        subtitle=f"<span class='category text-muted'>{record['category']}</span> IoU: {record['iou']:.3f} Entail: <span class='word' data-act='{record['w_entail'] * 10:.3f}'>{record['w_entail']:.3f}</span> Neutral: <span class='word' data-act='{record['w_neutral'] * 10:.3f}'>{record['w_neutral']:.3f}</span> Contra: <span class='word' data-act='{record['w_contra'] * 10:.3f}'>{record['w_contra']:.3f}</span>",
        items=all_examples_html,
    )
    return fmt


def make_html(
    records,
    # Standard feats
    toks,
    states,
    tok_feats,
    idxs,
    preds,
    # General stuff
    weights,
    dataset,
    result_dir,
    filename="pred.html",
):
    states = np.stack(states)
    assert preds.shape[0] == len(toks)
    tok_feats, tok_feats_vocab = tok_feats
    html = [c.HTML_PREFIX]
    html_dir = os.path.join(result_dir, "html")
    os.makedirs(html_dir, exist_ok=True)

    # Loop through units
    for record in records:
        card_html = make_card(
            record, toks, tok_feats, tok_feats_vocab, states, preds, dataset
        )
        html.append(card_html)

    html.append(c.HTML_SUFFIX)
    html_final = "\n".join(html)
    with open(os.path.join(html_dir, filename), "w") as f:
        f.write(html_final)
