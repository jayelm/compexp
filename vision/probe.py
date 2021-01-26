import settings
from loader.model_loader import loadmodel
from dissection.neuron import hook_feature, NeuronOperator
from dissection import contrib
from visualize.report import (
    neuron as vneuron,
    final as vfinal,
    index as vindex,
)
from util.clean import clean
from util.misc import safe_layername
from tqdm import tqdm
import pickle
import os
import pandas as pd
from loader.data_loader import ade20k
import numpy as np


def noop(*args, **kwargs):
    return None


layernames = list(map(safe_layername, settings.FEATURE_NAMES))


hook_modules = []


def spread_contrs(weights, contrs, layernames):
    """
    [
    (layer1)
    {
        'feat_corr': {
            'weight': ...,
            'contr': ...
        }
    }
    ]
    """
    return [
        {
            name: {"weight": weights[name][i], "contr": contrs[name][i]}
            for name in weights.keys()
        }
        for i in range(len(layernames))
    ]


model = loadmodel(hook_feature, hook_modules=hook_modules)
fo = NeuronOperator()

# ==== STEP 1: Feature extraction ====
# features: list of activations - one 63305 x c x h x w tensor for each feature
# layer (defined by settings.FEATURE_NAMES; default is just layer4)
# maxfeature: the maximum activation across the input map for each channel (e.g. for layer 4, there is a 7x7 input map; what's the max value). one 63305 x c tensor for each feature
features, maxfeature, preds, logits = fo.feature_extraction(model=model)

# ==== STEP 2: Threshold quantization ====
thresholds = [
    fo.quantile_threshold(lf, savepath=f"quantile_{ln}.npy")
    for lf, ln in zip(features, layernames)
]

# ==== Average feature activation quantiling ====
# You can threshold by quantiling top alpha% of MEAN activations...
#  wholeacts = fo.get_total_activation(features[-1])
# ...or simply wherever there's a hit
wholeacts = features[-1] > thresholds[-1][np.newaxis, :, np.newaxis, np.newaxis]
wholeacts = wholeacts.any((2, 3))

# ==== Confusion matrix =====
pred_records = []
for i, ((p, t), acts) in enumerate(zip(preds, wholeacts)):
    acts = acts * 1  # To int
    pred_name = ade20k.I2S[p]
    target_name = f"{fo.data.scene(i)}-s"
    if target_name in ade20k.S2I:
        pred_records.append((pred_name, target_name, *acts))

pred_df = pd.DataFrame.from_records(
    pred_records, columns=["pred", "target", *map(str, range(wholeacts.shape[1]))]
)
pred_df.to_csv(os.path.join(settings.OUTPUT_FOLDER, "preds.csv"), index=False)
print(f"Accuracy: {(pred_df.pred == pred_df.target).mean() * 100:.2f}%")

# ==== Multilayer case - neuron contributions ====
if settings.CONTRIBUTIONS:
    contr_fname = os.path.join(settings.OUTPUT_FOLDER, "contrib.pkl")
    if os.path.exists(contr_fname):
        print(f"Loading cached contributions {contr_fname}")
        with open(contr_fname, "rb") as f:
            contrs_spread = pickle.load(f)
    else:
        print("Computing contributions")
        weights = {
            "weight": contrib.get_weights(hook_modules),
            "feat_corr": contrib.get_feat_corr(features),
            "act_iou": contrib.get_act_iou(features, thresholds),
        }

        # Final layer
        final_weight_np = model.fc.weight.detach().cpu().numpy()
        weights["weight"].append(final_weight_np)
        weights["feat_corr"].append(
            contrib.get_feat_corr(
                [features[-1].mean(2).mean(2), logits], flattened=True
            )[1]
        )

        # For consistency w/ correlation measures, do 1% activations
        final_thresholds = fo.quantile_threshold(
            logits[:, :, np.newaxis, np.newaxis], savepath="quantile_logits.npy"
        )
        weights["act_iou"].append(
            contrib.get_act_iou(
                [features[-1], logits[:, :, np.newaxis, np.newaxis]],
                [thresholds[-1], final_thresholds],
            )[1]
        )
        contrs = {
            name: contrib.threshold_contributors(weight, alpha_global=0.01)
            for name, weight in weights.items()
        }

        contrs_spread = spread_contrs(weights, contrs, layernames + ["final"])

        with open(contr_fname, "wb") as f:
            pickle.dump(contrs_spread, f)
else:
    contrs_spread = [{} for _ in layernames + ["final"]]

# Zip it all together
ranger = tqdm(
    zip(
        layernames,
        features,
        maxfeature,
        thresholds,
        [None, *layernames],
        [None, *features],
        [None, *thresholds],
        contrs_spread,
    ),
    total=len(layernames),
)

tallies = []
# Load cached card htmls if they exist - will be overwritten if not skipping
# summarization step
all_card_htmls_fname = os.path.join(settings.OUTPUT_FOLDER, "card_htmls.pkl")
if os.path.exists(all_card_htmls_fname):
    print(f"Loading cached card htmls {all_card_htmls_fname}")
    with open(all_card_htmls_fname, "rb") as f:
        all_card_htmls = pickle.load(f)
else:
    all_card_htmls = {}


for (
    layername,
    layer_features,
    layer_maxfeature,
    layer_thresholds,
    prev_layername,
    prev_features,
    prev_thresholds,
    layer_contrs,
) in ranger:
    ranger.set_description(f"Layer {layername}")

    # ==== STEP 3: calculating IoU scores ====
    # Get tally dfname
    if settings.UNIT_RANGE is None:
        tally_dfname = f"tally_{layername}.csv"
    else:
        # Only use a subset of units
        tally_dfname = f"tally_{layername}_{min(settings.UNIT_RANGE)}_{max(settings.UNIT_RANGE)}.csv"

    tally_result, mc = fo.tally(
        layer_features, layer_thresholds, savepath=tally_dfname
    )

    # ==== STEP 4: generating results ====
    card_htmls = vneuron.generate_html_summary(
        fo.data,
        layername,
        preds,
        mc,
        tally_result=tally_result,
        contributors=layer_contrs,
        maxfeature=layer_maxfeature,
        features=layer_features,
        prev_layername=prev_layername,
        prev_tally=None if not tallies else tallies[-1],
        prev_features=prev_features,
        prev_thresholds=prev_thresholds,
        thresholds=layer_thresholds,
        force=True,
        skip=False,
    )

    tallies.append({record["unit"]: record for record in tally_result})
    if card_htmls is not None:
        all_card_htmls[layername] = card_htmls

# ==== STEP 5: generate last layer contributions visualization ====
if settings.CONTRIBUTIONS:
    final_weight = contrs_spread[-1]["weight"]["weight"]
    final_card_htmls = vfinal.generate_final_layer_summary(
        fo.data,
        final_weight,
        features[-1],
        thresholds[-1],
        preds,
        logits,
        prev_layername=layernames[-1],
        prev_tally=tallies[-1],
        contributors=contrs_spread[-1],
        skip=False,
    )


    # ==== STEP 6: generate index html ====
    # Add final contrs/tallies/layernames/htmls:
    all_card_htmls["final"] = final_card_htmls
    if len(all_card_htmls) <= 1:
        # Don't overwrite the existing one
        print("Warning - no card htmls collected")
    else:
        with open(all_card_htmls_fname, "wb") as f:
            pickle.dump(all_card_htmls, f)

    # Add final tallies and layernames (one indexed)
    tallies.append({k: {"label": v} for k, v in ade20k.I2S.items()})
    layernames.append("final")
    vindex.generate_index(layernames, contrs_spread, tallies, all_card_htmls)

if settings.CLEAN:
    clean()
