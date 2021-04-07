"""
Visualize active learning decisions.
"""


import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import settings

from . import common as c


def to_ul(items):
    items_html = [f"<li class='sentence list-group-item'>{it}</li>" for it in items]
    items_html = "".join(items_html)
    return f"<ul class='list-group list-group-flush'>{items_html}</ul>"


def make_dtags(fs):
    dtags = [
        f"data-{cname}='{val}'"
        for cname, val in fs.items()
        # Skip lemma as it contains apostrophes
        if cname != "lemma"
    ]
    return " ".join(dtags)


def make_tooltip(fs):
    """
    Make a tooltip from the given feat info
    """
    fs_clean = {cname: c.unquote(val) for cname, val in fs.items()}
    fstrs = [
        f'<li class="list-group-item"><span class="fname">{cname}</span>: <span class="fval">{val}</span></li>'
        for cname, val in fs_clean.items()
    ]
    fstrs = "".join(fstrs)
    return f'<ul class="list-group list-group-flush">{fstrs}</ul>'


def make_spans(words, acts, feats, multiact=False):
    data_tags = [make_dtags(fs) for fs in feats]
    tooltips = [make_tooltip(fs) for fs in feats]
    if multiact:
        data_acts = [
            " ".join([f"data-act-{ai}='{a:f}'" for ai, a in ac]) for ac in acts
        ]
    else:
        data_acts = [f"data-act='{a:f}'" for a in acts]
    spans = [
        f"<span class='word' {dact} {dtag} data-toggle='tooltip' data-html='true' data-placement='top' title='{t}'>{c.unquote(w)}</span>"
        for w, dact, dtag, t in zip(words, data_acts, data_tags, tooltips)
    ]
    return " ".join(spans)


def orig_val(val):
    """
    Remove the :
    """
    if val == "UNK":
        return val
    return val.split(":", maxsplit=1)[1]


def get_feat_dicts(df, dataset):
    feats_i = {
        cname: [dataset.name_feat(f) for f in df[f"f_{cname}"]]
        for cname in dataset.ocnames + dataset.ccnames
    }
    mfeats_i = {
        cname: [
            ";".join(orig_val(dataset.name_feat(f)) for f in mfs)
            for mfs in df[f"f_{cname}"]
        ]
        for cname in dataset.mcnames
    }
    # To individual dicts
    feats_indiv = []
    for i in range(len(df)):
        fs = {
            cname: orig_val(feats_i[cname][i])
            for cname in dataset.ocnames + dataset.ccnames
        }
        for mcname in dataset.mcnames:
            fs[mcname] = mfeats_i[mcname][i]
        feats_indiv.append(fs)
    return feats_indiv


def get_mfeats(feats, dataset):
    mfeats_split = {}
    for mcname in dataset.mcnames:
        orig_fis = dataset.cnames2fis[mcname]
        orig_idxs = [dataset.multi2idx[ofi] for ofi in orig_fis]
        mc_feats = []
        for i in range(feats["multi"].shape[0]):
            these_feats = []
            for fi, idx in zip(orig_fis, orig_idxs):
                if feats["multi"][i, idx]:
                    these_feats.append(fi)
            mc_feats.append(these_feats)
        mfeats_split[f"f_{mcname}"] = mc_feats
    return mfeats_split


# FIXME: stop passing so many arguments around
def get_examples(toks, states, feats, idxs, dataset):
    # Split up feats
    feats_split = {
        f"f_{cname}": feats["onehot"][:, ci]
        for ci, cname in enumerate(dataset.cnames)
        if not dataset.is_multi(cname)
    }
    mfeats_split = get_mfeats(feats, dataset)
    # Turn into dataframe
    df = pd.DataFrame(
        {"words": toks, "act": states, "idx": idxs, **feats_split, **mfeats_split}
    )
    agg = (
        df.groupby("idx")
        .agg(
            max_act=pd.NamedAgg(column="act", aggfunc=np.max),
            mean_act=pd.NamedAgg(column="act", aggfunc=np.mean),
        )
        .sort_values("max_act", ascending=False)
    )

    topn = agg.head(settings.TOPN)
    items = []
    for i in topn.index:
        df_i = df[df["idx"] == i]
        words = dataset.to_text(df_i["words"])
        # Get rid of first part
        fdicts = get_feat_dicts(df_i, dataset)
        acts = df_i["act"]
        spans = make_spans(words, acts, fdicts)
        items.append(spans)
    return to_ul(items)


def make_card(record, toks, states, feats, idxs, dataset):
    fmt = c.CARD_HTML.format(
        unit=record["neuron"],
        iou=f"{record['iou']:.3f}",
        label=record["feature"],
        entail=record["w_entail"],
        neutral=record["w_neutral"],
        contra=record["w_contra"],
        title=f"Unit {record['neuron']} {record['feature']}",
        subtitle=f"IoU: {record['iou']:.3f} Entail: {record['w_entail']:.3f} Neutral: {record['w_neutral']:.3f} Contra: {record['w_contra']:.3f}",
        items=get_examples(toks, states, feats, idxs, dataset),
    )
    return fmt


def make_html(records, toks, states, feats, idxs, weights, dataset, result_dir):
    html = [c.HTML_PREFIX]
    html_dir = os.path.join(result_dir, "html")
    os.makedirs(html_dir, exist_ok=True)

    for record in tqdm(records):
        i = record["neuron"]
        card_html = make_card(record, toks, states[:, i], feats, idxs, dataset)
        html.append(card_html)

    html.append(c.HTML_SUFFIX)
    html_final = "\n".join(html)
    with open(os.path.join(html_dir, "index.html"), "w") as f:
        f.write(html_final)
