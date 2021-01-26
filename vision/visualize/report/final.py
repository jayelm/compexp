"""
viewprobe creates visualizations for a certain eval.
"""

import re
import numpy
from imageio import imread, imwrite
import visualize.expdir as expdir
import visualize.bargraph as bargraph
from visualize.report import html_common
from visualize.report.image import score_histogram
from loader.data_loader import ade20k
import settings
import numpy as np
from PIL import Image
import warnings
from tqdm import tqdm, trange
import loader.data_loader.formula as F
import os
import shutil


def generate_final_layer_summary(
    ds,
    weight,
    last_features,
    last_thresholds,
    last_preds,
    last_logits,
    prev_layername=None,
    prev_tally=None,
    contributors=None,
    skip=False,
):
    if skip:
        return
    ed = expdir.ExperimentDirectory(settings.OUTPUT_FOLDER)
    ed.ensure_dir("html", "image")

    html_fname = ed.filename(f"html/{expdir.fn_safe('final')}.html")

    print(f"Generating html summary {html_fname}")
    html = [html_common.HTML_PREFIX]
    html.append(f"<h3>{settings.OUTPUT_FOLDER} - Final Layer</h3>")

    # Loop through classes
    card_htmls = {}
    for cl in trange(weight.shape[0], desc="Final classes"):
        card_html = []
        # TODO: Make this compatible with non-ade20k
        cl_name = ade20k.I2S[cl]
        card_html.append(
            f'<div class="card contr contr_final">'
            f'<div class="card-header">'
            f'<h5 class="mb-0">{cl_name}</h5>'
            f"</div>"
            f'<div class="card-body">'
        )
        all_contrs = []
        for contr_i, contr_name in enumerate(sorted(list(contributors.keys()))):
            contr_dict = contributors[contr_name]
            if contr_dict["contr"][0] is None:
                continue
            contr, inhib = contr_dict["contr"]
            if contr.shape[0] != weight.shape[0]:
                raise RuntimeError(
                    f"Probably mismatched contrs: weight shape {weight.shape} contr shape {contr.shape}"
                )
            weight = contr_dict["weight"]
            contr_url_str, contr_label_str, contr = html_common.to_labels(
                cl, contr, weight, prev_tally, uname=cl_name
            )
            inhib_url_str, inhib_label_str, inhib = html_common.to_labels(
                cl, inhib, weight, prev_tally, uname=cl_name, label_class="inhib-label"
            )

            all_contrs.extend(contr)

            card_html.append(
                f'<p class="contributors"><a href="{prev_layername}.html?u={contr_url_str}">Contributors ({contr_name}): {contr_label_str}</a></p>'
                f'<p class="inhibitors"><a href="{prev_layername}.html?u={inhib_url_str}">Inhibitors ({contr_name}): {inhib_label_str}</a></p>'
            )

        all_contrs = list(set(all_contrs))
        # Save images with highest logits
        cl_images = [
            i for i in range(len(last_features)) if f"{ds.scene(i)}-s" == cl_name
        ]
        cl_images = sorted(cl_images, key=lambda i: last_logits[i, cl], reverse=True)
        if cl_images:
            for i, im_index in enumerate(cl_images[:5]):
                imfn = ds.filename(im_index)
                imfn_base = os.path.basename(imfn)
                html_imfn = ed.filename(f"html/image/{imfn_base}")
                shutil.copy(imfn, html_imfn)
                img_html = f'<img loading="lazy" class="mask-img" id="{cl_name}-{i}" data-masked="false" data-uname="{cl_name}" width="100" height="100" data-imfn="{imfn_base}" src="image/{imfn_base}">'
                card_html.append(html_common.wrap_image(img_html))
                # Save masks
                for cunit in all_contrs:
                    imfn_alpha = imfn_base.replace(".jpg", ".png")
                    mask = html_common.create_mask(
                        im_index,
                        cunit,
                        last_features,
                        last_thresholds,
                        settings.IMG_SIZE,
                    )
                    mask_fname = ed.filename(
                        f"html/image/mask-{cunit}-{imfn_alpha}"
                    )
                    mask.save(mask_fname)

        card_html.append(f"</div></div>")
        card_html_str = "".join(card_html)
        card_htmls[cl] = card_html_str
        html.append(card_html_str)

    html.append(html_common.HTML_SUFFIX)
    with open(html_fname, "w") as f:
        f.write("\n".join(html))
    return card_htmls
