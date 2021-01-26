"""
Contributions

TODO: get high firing images when neuron fires past threshold
"""

import numpy as np
from scipy.spatial.distance import cdist
from skimage.measure import block_reduce


def reduce_feats(prev, curr):
    assert prev.shape[2] > curr.shape[2]
    # Do pooling over previous layer
    kernel_size = (prev.shape[2] // curr.shape[2], prev.shape[3] // curr.shape[3])
    return block_reduce(prev, (1, 1, *kernel_size), np.mean)


def flatten_features(feats):
    """
    Assume features is
    (n_images x n_units x rf_height x rf_width)
    this reshapes to
    (n_units x ...)
    so we can measure correlation across all patches.
    """
    return np.transpose(feats, (1, 0, 2, 3)).reshape(feats.shape[1], -1)


def get_feat_corr(features, flattened=False):
    """
    Get correlations between firing patterns across the features

    If flattened is True, assume features have shape NC rather than NCHW
    """
    corrs = [None]
    for prev, curr in zip(features, features[1:]):
        if not flattened:
            if prev.shape[2:] != curr.shape[2:]:
                prev = reduce_feats(prev, curr)

            prev_flat = flatten_features(prev)
            curr_flat = flatten_features(curr)
        else:
            prev_flat = prev.T
            curr_flat = curr.T
        # Transpose so we have vectors of activations for any patch anywhere in
        # the dataset
        corr = np.corrcoef(curr_flat, prev_flat)
        # cor(x, x)  cor(x, y)
        # cor(y, x)  cor(y, y)
        # Get top right (starts after x.shape, x.shape)
        corr = corr[: curr_flat.shape[0], curr_flat.shape[0] :]
        corrs.append(corr)
    return corrs


def get_act_iou(features, threshold, mode="contr"):
    """
    Jaccard similarity between "active" neurons, where activations are defined
    via thresholds
    """
    if mode not in {"contr", "inhib"}:
        raise ValueError(mode)
    # Get jaccard similarity between "active" neurons (where activations are
    # Get correlations between firing patterns across the features
    ious = [None]
    for prev, curr, prev_thresh, curr_thresh in zip(
        features, features[1:], threshold, threshold[1:]
    ):
        if prev.shape[2:] != curr.shape[2:]:
            prev = reduce_feats(prev, curr)
        prev = flatten_features(prev)
        curr = flatten_features(curr)

        curr_acts = curr > curr_thresh[:, np.newaxis]
        prev_acts = prev > prev_thresh[:, np.newaxis]

        if mode == "inhib":
            curr_acts = 1 - curr_acts

        iou = 1 - cdist(curr_acts, prev_acts, metric="jaccard")
        ious.append(iou)
    return ious


def get_act_iou_inhib(features, threshold):
    """
    If a previous neuron fires, does it inhibit a current neuron from firing?
    Jaccard similarity between active neurons of previous layer and inactive
    neurons of current layer, where activations are defined via thresholds
    """
    return get_act_iou(features, threshold, mode="inhib")


def get_weights(modules):
    """
    "Similarity" matrix just defined by strength of connections between
    adjacent modules.
    """
    weights = [m.weight.detach().cpu().numpy() for m in modules[1:]]
    # Take average over receptive field
    weights_mean = [None, *[w.mean(2).mean(2) for w in weights]]
    return weights_mean


def threshold_contributors(weights, n=None, alpha=None, alpha_global=None):
    """
    For the "similarity" matrices defined by "weights",
    for each non-initial layer in modules, get the name of the neurons that
    contribute and inhibit the neuron activation most -  in other words, that
    have the highest and lowest similarity values.

    :param weights: a list of matrices defining "similarity"-ish matrices (no
        similarity requirements, but weight should roughly correspond to
        strength of connection), where (important!):
            the ith row of the similraity matrix indicates weights of the
            current neuron i to the previous neurons (i.e. columns are previous
            neurons)
    :param n: number of contributing neurons to get.
    :param alpha: for each neuron get contributing neurons in this top percentile of weights FOR THE NEURON.
    :param alpha_global: for each neuron get contributing neurons in this top percentile of weights ACROSS ALL NEURONS IN THE LAYER (therefore # of weights will be variable)

    :return contr, inhib: tuple - thresholded binary matrix, one of
    contributors (max values); the other of inhibitors (min values)

    You must specify exactly one of either n, alpha, or alpha global.
    """
    nones = sum([n is None, alpha is None, alpha_global is None])
    if nones != 2:
        raise ValueError("Must specify exactly one of n, alpha, or alpha_global")

    contr = [None]
    inhib = [None]
    for curr in weights[1:]:
        # curr: (in_channels x out_channels x h x w)
        # Take average or max over kernel? Let's do max
        if n is not None:
            raise NotImplementedError
            #  inhib_threshold = n
            #  contr_threshold = max_kernel.shape[0] - n
            #  max_kernel = np.argsort(max_kernel, axis=0)
        else:
            # Compute by threshold
            if alpha_global is not None:
                thresholds = np.quantile(
                    curr, [alpha_global, 1 - alpha_global], keepdims=True
                )
            elif alpha is not None:
                thresholds = np.quantile(
                    curr, [alpha, 1 - alpha], axis=1, keepdims=True
                )
            inhib_threshold = thresholds[0]
            contr_threshold = thresholds[1]

        kernel_inhib = curr < inhib_threshold
        kernel_contr = curr > contr_threshold

        inhib.append(kernel_inhib)
        contr.append(kernel_contr)

    return list(zip(contr, inhib))
