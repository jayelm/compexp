"""
viewprobe creates visualizations for a certain eval.
"""

import numpy
import visualize.expdir as expdir
import visualize.bargraph as bargraph
from visualize.report import html_common, cards
from visualize.report.image import score_histogram
import settings
import numpy as np
import warnings
from tqdm import tqdm
import os


# unit,category,label,score


def order_by(records, key):
    for i, record in enumerate(sorted(records, key=lambda record: -float(record[key]))):
        record[f"{key}-order"] = i


def generate_html_summary(
    ds,
    layer,
    preds,
    mc,
    maxfeature=None,
    features=None,
    thresholds=None,
    imsize=None,
    imscale=72,
    tally_result=None,
    contributors=None,
    prev_layername=None,
    prev_tally=None,
    prev_features=None,
    prev_thresholds=None,
    gridwidth=None,
    gap=3,
    limit=None,
    force=False,
    verbose=False,
    skip=False,
):
    if skip:
        return
    ed = expdir.ExperimentDirectory(settings.OUTPUT_FOLDER)
    print(
        "Generating html summary %s"
        % ed.filename("html/%s.html" % expdir.fn_safe(layer))
    )
    if verbose:
        print("Sorting units by score.")
    if imsize is None:
        imsize = settings.IMG_SIZE
    # 512 units x top_n.
    # For each unit, images that get the highest activation (anywhere?), in
    # descending order.
    top = np.argsort(maxfeature, 0)[: -1 - settings.TOPN : -1, :].transpose()
    ed.ensure_dir("html", "image")
    html = [html_common.HTML_PREFIX]
    html.append(f"<h3>{settings.OUTPUT_FOLDER}</h3>")
    rendered_order = []
    barfn = "image/%s-bargraph.svg" % (expdir.fn_safe(layer))
    try:
        bargraph.bar_graph_svg(
            ed,
            layer,
            tally_result=tally_result,
            rendered_order=rendered_order,
            save=ed.filename("html/" + barfn),
        )
    except ValueError as e:
        # Probably empty
        warnings.warn(f"could not make svg bargraph: {e}")
        pass
    html.extend(
        [
            '<div class="histogram">',
            '<img class="img-fluid" src="%s" title="Summary of %s %s">'
            % (barfn, ed.basename(), layer),
            "</div>",
        ]
    )
    # histogram 2 ==== iou
    ioufn = f"image/{expdir.fn_safe(layer)}-iou.svg"
    ious = [float(r["score"]) for r in rendered_order]
    iou_mean = np.mean(ious)
    iou_std = np.std(ious)
    iou_title = f"IoUs ({iou_mean:.3f} +/- {iou_std:.3f})"
    score_histogram(
        rendered_order, os.path.join(ed.directory, "html", ioufn), title=iou_title
    )
    html.extend(
        [
            '<div class="histogram">',
            '<img class="img-fluid" src="%s" title="Summary of %s %s">'
            % (ioufn, ed.basename(), layer),
            "</div>",
        ]
    )
    html.append(html_common.FILTERBOX)
    html.append('<div class="gridheader">')
    html.append('<div class="layerinfo">')
    html.append(
        "%d/%d units covering %d concepts with IoU &ge; %.2f"
        % (
            len(
                [
                    record
                    for record in rendered_order
                    if float(record["score"]) >= settings.SCORE_THRESHOLD
                ]
            ),
            len(rendered_order),
            len(
                set(
                    record["label"]
                    for record in rendered_order
                    if float(record["score"]) >= settings.SCORE_THRESHOLD
                )
            ),
            settings.SCORE_THRESHOLD,
        )
    )
    html.append("</div>")
    sort_by = ["score", "unit"]

    if settings.SEMANTIC_CONSISTENCY:
        sort_by.append("consistency")
    if settings.EMBEDDING_SUMMARY:
        sort_by.append("emb_summary")
    if settings.WN_SUMMARY:
        sort_by.append("wn_summary")
    if settings.WN_SIMILARITY:
        sort_by.append("wn_mean_sim")
        sort_by.append("wn_min_sim")

    html.append(html_common.get_sortheader(sort_by))
    html.append("</div>")

    if gridwidth is None:
        gridname = ""
        gridwidth = settings.TOPN
    else:
        gridname = "-%d" % gridwidth

    html.append('<div class="unitgrid">')
    if limit is not None:
        rendered_order = rendered_order[:limit]

    # Assign ordering based on score, consistency, and/or
    order_by(rendered_order, "score")
    order_by(rendered_order, "consistency")
    order_by(rendered_order, "emb_summary_sim")
    order_by(rendered_order, "wn_summary_sim")
    order_by(rendered_order, "wn_mean_sim")
    order_by(rendered_order, "wn_min_sim")

    # TODO: Make embedding summary searchable too.

    # Visualize neurons
    card_htmls = {}
    for label_order, record in enumerate(
        tqdm(rendered_order, desc="Visualizing neurons")
    ):
        card_html = cards.make_card_html(
            ed,
            label_order,
            record,
            ds,
            mc,
            layer,
            gridname,
            top,
            features,
            thresholds,
            preds,
            contributors,
            prev_layername,
            prev_features,
            prev_thresholds,
            prev_tally,
            force=force,
        )
        html.append(card_html)
        card_htmls[record["unit"]] = card_html

    html.append("</div>")
    html.extend([html_common.HTML_SUFFIX])
    with open(ed.filename("html/%s.html" % expdir.fn_safe(layer)), "w") as f:
        f.write("\n".join(html))

    return card_htmls
