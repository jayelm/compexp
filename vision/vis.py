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
from scipy.spatial import distance
import torch
#  import torch.nn.functional as F
import pickle
import os
import pandas as pd
from loader.data_loader import ade20k
from loader.data_loader import formula as F
import numpy as np
import pycocotools.mask as cmask
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from scipy import ndimage
import matplotlib.colors


MG = 1
BLUE_TINT = np.array([-MG * 255, -MG * 255, MG * 255])
PURPLE_TINT = np.array([MG * 255, -MG * 600, MG * 255])
RED_TINT = np.array([MG * 255, -MG * 255, -MG * 255])


def add_heatmap(img, acts, actmin=0.0, actmax=1.0):
    cmap = plt.cm.rainbow
    norm = matplotlib.colors.Normalize(vmin=actmin, vmax=actmax)
    act_colors = cmap(norm(acts))
    act_colors = act_colors[:, :, :3]
    # weighted combination of colors and mask
    weighted = (0.5 * act_colors * 255) + (0.5 * img)
    weighted = np.clip(weighted, 0, 255)
    weighted = np.round(weighted).astype(np.uint8)
    return weighted


def iou(m, n):
    intersection = np.logical_and(m, n).sum()
    union = np.logical_or(m, n).sum()
    return (intersection) / (union + 1e-10)


def iou_mask(img, neuron, mask):
    img = img.astype(np.int64)

    nborder = get_border(neuron)
    mborder = get_border(mask)
    border = np.logical_or(nborder, mborder)

    intersection = np.logical_and(neuron, mask)
    #  union = np.logical_xor(neuron, mask)

    int_mask = intersection[:, :, np.newaxis] * PURPLE_TINT
    int_mask = np.round(int_mask).astype(np.int64)

    neuron_mask = neuron[:, :, np.newaxis] * RED_TINT
    neuron_mask = np.round(neuron_mask).astype(np.int64)

    mask_mask = mask[:, :, np.newaxis] * BLUE_TINT
    mask_mask = np.round(mask_mask).astype(np.int64)

    img += neuron_mask + mask_mask + int_mask

    img[border] = (255, 255, 0)

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def noop(*args, **kwargs):
    return None


layernames = list(map(safe_layername, settings.FEATURE_NAMES))


hook_modules = []


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

# ==== New: multilayer case - neuron contributions ====
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


ds = fo.data


def upsample(actmap, shape):
    actmap_im_rsz = Image.fromarray(actmap).resize(shape, resample=Image.BILINEAR)
    actmap_rsz = np.array(actmap_im_rsz)
    return actmap_rsz


def get_border(x):
    x = x.astype(np.uint8)
    border = x - ndimage.morphology.binary_dilation(x)
    border = border.astype(np.bool)
    return border


def add_mask(img, mask):
    img = img.astype(np.int64)
    mask_border = get_border(mask)

    mask_weights = np.clip(mask.astype(np.float32), 0.25, 1.0)

    img_masked = img * mask_weights[:, :, np.newaxis]
    img_masked[mask_border] = (255, 255, 0)
    img_masked = np.clip(np.round(img_masked), 0, 255).astype(np.uint8)
    return img_masked


def friendly(mstr):
    return mstr.replace('(', '').replace(')', '').replace(' ', '_').replace(',', '')


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
    N = 483
    top = np.argsort(layer_maxfeature, 0)[: -1 -20 : -1, :].transpose()
    visdir = os.path.join(settings.OUTPUT_FOLDER, f'vis_{N}')
    os.makedirs(visdir, exist_ok=True)

    # DO THE FULL UPSAMPLING OF THIS FEATURE MASK (this is very inefficient_)
    feats = layer_features[:, N]
    from tqdm import tqdm
    acts = np.stack([
        upsample(feats_i, (112, 112))
        for feats_i in tqdm(feats, desc='upsampling')
    ])
    neuron_masks = acts > layer_thresholds[N]
    actmin = acts.min()
    actmax = acts.max()

    for record in tally_result:
        if record['unit'] != N:
            continue

        def get_labs(label):
            # Image masks
            labs_enc = mc.get_mask(label)
            # Mask labels
            labs = cmask.decode(labs_enc)
            labs = labs.reshape((layer_features.shape[0], *mc.mask_shape))
            return labs

        labd = {}
        def rec_add(lab_f):
            lab_str = lab_f.to_str(lambda x: ds.name(None, x))
            labd[lab_str] = get_labs(lab_f)
            # Measure IoU again.
            #  print(f"{lab_str} IoU: {iou(labd[lab_str], neuron_masks):f}")

            if isinstance(lab_f, F.Leaf):
                return
            elif isinstance(lab_f, F.Not):
                rec_add(lab_f.val)
            elif isinstance(lab_f, F.And) or isinstance(lab_f, F.Or):
                # binary op
                rec_add(lab_f.left)
                rec_add(lab_f.right)
            else:
                raise ValueError(f"Unknown formula {lab_f}")

        root_f = F.parse(record["label"], reverse_namer=ds.rev_name)
        rec_add(root_f)

        # Neuron mask
        # Upsample
        for i, index in enumerate(tqdm(top[N], desc='topn')):
            # Most popular images
            imfn = ds.filename(index)
            img = np.array(Image.open(imfn))
            # Neuron activations - upsample

            # Here's your neuron mask
            acts_up = acts[index]
            acts_up = upsample(acts_up, img.shape[:2])

            img_hm = add_heatmap(img, acts_up, actmin, actmax)
            new_imfn = os.path.join(visdir, f"{i}_neuron_act.jpg")
            Image.fromarray(img_hm).save(new_imfn)

            # Find borders
            neuron_mask = acts_up > layer_thresholds[N]
            img_masked = add_mask(img, neuron_mask)
            new_imfn = os.path.join(visdir, f"{i}_neuron.jpg")
            Image.fromarray(img_masked).save(new_imfn)

            # Go through primitives and masks
            # Save original
            orig_imfn = os.path.join(visdir, f"{i}_orig.jpg")
            Image.fromarray(img).save(orig_imfn)

            for mstr, m in labd.items():
                m = m[index]
                m = upsample(m, img.shape[:2])
                mstr_friendly = friendly(mstr)
                img_masked = add_mask(img, m)
                new_imfn = os.path.join(visdir, f"{i}_{mstr_friendly}.jpg")
                Image.fromarray(img_masked).save(new_imfn)

                # IoU masks
                iou_imfn = os.path.join(visdir, f"{i}_{mstr_friendly}_iou.jpg")
                iou_img = iou_mask(img, neuron_mask, m)
                Image.fromarray(iou_img).save(iou_imfn)

        # Find examples - water river AND blue
        if N == 483:
            mstr = "((water OR river) AND blue-c)"
            new_m = F.parse(mstr, reverse_namer=ds.rev_name)
            mstr_friendly = friendly(mstr)
            new_labs = get_labs(new_m)
            # Sort by most common
            top = np.argsort(new_labs.sum((1, 2)))[::-1]
            top = top[30:50]
            for i, index in enumerate(tqdm(top, desc='blue')):
                # Most popular images
                imfn = ds.filename(index)
                img = np.array(Image.open(imfn))
                # Neuron activations - upsample
                actmap = acts[index]
                actmap = upsample(actmap, img.shape[:2])

                img_hm = add_heatmap(img, actmap, actmin, actmax)
                new_imfn = os.path.join(visdir, f"BLUE-{i}_neuron_act.jpg")
                Image.fromarray(img_hm).save(new_imfn)

                # Here's your neuron mask
                neuron_mask = actmap > layer_thresholds[N]

                img_masked = add_mask(img, neuron_mask)
                new_imfn = os.path.join(visdir, f"BLUE-{i}_neuron.jpg")
                Image.fromarray(img_masked).save(new_imfn)

                # Go through primitives and masks
                m = new_labs[index]
                m = upsample(m, img.shape[:2])
                img_masked = add_mask(img, m)
                new_imfn = os.path.join(visdir, f"BLUE-{i}_{mstr_friendly}.jpg")
                Image.fromarray(img_masked).save(new_imfn)
                for mstr, m in labd.items():
                    m = m[index]
                    m = upsample(m, img.shape[:2])
                    mstr_friendly = friendly(mstr)
                    img_masked = add_mask(img, m)
                    new_imfn = os.path.join(visdir, f"BLUE-{i}_{mstr_friendly}.jpg")
                    Image.fromarray(img_masked).save(new_imfn)

                    iou_imfn = os.path.join(visdir, f"BLUE-{i}_{mstr_friendly}_iou.jpg")
                    iou_img = iou_mask(img, neuron_mask, m)
                    Image.fromarray(iou_img).save(iou_imfn)

                # Save original
                orig_imfn = os.path.join(visdir, f"BLUE-{i}_orig.jpg")
                Image.fromarray(img).save(orig_imfn)
