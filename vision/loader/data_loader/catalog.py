"""
Functions for returning masks
"""

import numpy as np
import settings
from tqdm import tqdm
from . import formula as F

import os
import pickle
from pycocotools import mask as cmask


def get_mask_global(masks, f):
    """
    Serializable/global version of get_mask for multiprocessing
    """
    # TODO: Handle here when doing AND and ORs of scenes vs scalars.
    if isinstance(f, F.And):
        masks_l = get_mask_global(masks, f.left)
        masks_r = get_mask_global(masks, f.right)
        return cmask.merge((masks_l, masks_r), intersect=True)
    elif isinstance(f, F.Or):
        masks_l = get_mask_global(masks, f.left)
        masks_r = get_mask_global(masks, f.right)
        return cmask.merge((masks_l, masks_r), intersect=False)
    elif isinstance(f, F.Not):
        masks_val = get_mask_global(masks, f.val)
        return cmask.invert(masks_val)
    elif isinstance(f, F.Leaf):
        return masks[f.val]
    else:
        raise ValueError("Most be passed formula")


class MaskCatalog:
    """
    A map from features to (possibly run-length-encoded, possibly cached)
    masks. Used for efficient mask lookup.
    """

    def __init__(self, prefetcher, cache=True, rle=True):
        """
        Initialize a mask catalog.

        if cache is true, use a cached mask file defined by
        settings.DATA_DIRECTORY and settings.INDEX_SUFFIX (see rle_masks_file)
        if rle is true, use mscoco's run length encoding.
        """
        self.prefetcher = prefetcher

        self.masks = {}
        self.data_size = self.prefetcher.segmentation.size()
        if hasattr(self.prefetcher.segmentation, "classes"):
            self.classes = self.prefetcher.segmentation.classes
            self.n_classes = len(np.unique(self.classes))
        else:
            self.classes = np.zeros(self.data_size, dtype=np.int64)
            self.n_classes = 1

        self.categories = self.prefetcher.segmentation.category_names()
        self.n_labels = len(self.prefetcher.segmentation.primary_categories_per_index())
        self.img2cat = np.zeros((self.data_size, len(self.categories)), dtype=np.bool)
        self.img2label = np.zeros((self.data_size, self.n_labels), dtype=np.bool)
        if settings.PROBE_DATASET == "broden":
            if '227' in settings.DATA_DIRECTORY:
                self.mask_shape = (113, 113)
            else:
                self.mask_shape = (112, 112)
        else:
            self.mask_shape = (224, 224)

        rle_masks_file = os.path.join(
            settings.DATA_DIRECTORY, f"rle_masks{settings.INDEX_SUFFIX}.pkl"
        )
        if cache and os.path.exists(rle_masks_file):
            with open(rle_masks_file, "rb") as f:
                cache = pickle.load(f)
                self.masks = cache["masks"]
                self.img2cat = cache["img2cat"]
                self.img2label = cache["img2label"]
        else:
            n_batches = int(np.ceil(self.data_size / settings.TALLY_BATCH_SIZE))
            for batch in tqdm(
                self.prefetcher.batches(), desc="Loading masks", total=n_batches
            ):
                for concept_map in batch:
                    img_index = concept_map["i"]
                    for cat_i, cat in enumerate(self.categories):
                        label_group = concept_map[cat]
                        shape = np.shape(label_group)
                        if len(shape) % 2 == 0:
                            label_group = np.array([label_group])
                        if len(shape) < 2:
                            # Scalar
                            for feat in label_group:
                                if feat == 0 and settings.PROBE_DATASET == "broden":
                                    # Somehow 0 is a feature?  Just continue -
                                    # it exists in index.csv under scenes but
                                    # not in c_scenes
                                    continue
                                if feat not in self.masks:
                                    # Treat it at pixel-level because a feature
                                    # can be both a pixel-level annotation and
                                    # a scene-level annotation
                                    self.initialize_mask(feat, "scalar")
                                self.masks[feat][img_index] = True
                                # This image displays this category
                                self.img2cat[img_index][cat_i] = True
                                self.img2label[img_index, feat] = True
                        else:
                            # Pixels
                            feats = np.unique(label_group.ravel())
                            for feat in feats:
                                # 0 is not a feature
                                if feat == 0 and settings.PROBE_DATASET == "broden":
                                    continue
                                if feat not in self.masks:
                                    self.initialize_mask(feat, "pixel")
                                # NOTE: sometimes label group is > 1 length (e.g.
                                # for parts) which means there are overlapping
                                # parts belonging to differrent objects. afaict
                                # these are ignored during normal tallying
                                # (np.concatenate followed by np.bincount.ravel())
                                if label_group.shape[0] == 1:
                                    bin_mask = label_group.squeeze(0) == feat
                                else:
                                    bin_mask = np.zeros_like(label_group[0])
                                    for lg in label_group:
                                        bin_mask = np.logical_or(bin_mask, (lg == feat))
                                if self.masks[feat].ndim == 1:
                                    # Sometimes annotation is both pixel and
                                    # scalar level. Retroactively fix this
                                    print(f"Coercing {feat} to pixel level")
                                    self.masks[feat] = np.tile(
                                        self.masks[feat][:, np.newaxis, np.newaxis],
                                        (1, *self.mask_shape),
                                    )
                                self.masks[feat][img_index] = np.logical_or(
                                    self.masks[feat][img_index], bin_mask
                                )
                                self.img2label[img_index, feat] = True
                                self.img2cat[img_index][cat_i] = True

            if rle:
                # Convert to run-length encoding
                for feat, mask in tqdm(
                    self.masks.items(), total=len(self.masks), desc="RLE"
                ):
                    if mask.ndim == 1:
                        mask = mask[:, np.newaxis, np.newaxis]
                        mask = np.broadcast_to(mask, (mask.shape[0], *self.mask_shape))
                    mask_flat = mask.reshape(
                        (mask.shape[0] * mask.shape[1], mask.shape[2])
                    )
                    mask_flat = np.asfortranarray(mask_flat)
                    self.masks[feat] = cmask.encode(mask_flat)
            if cache:
                with open(rle_masks_file, "wb") as f:
                    pickle.dump(
                        {
                            "masks": self.masks,
                            "img2label": self.img2label,
                            "img2cat": self.img2cat,
                        },
                        f,
                    )

        self.labels = sorted(list(self.masks.keys()))

    def get_mask(self, f):
        return get_mask_global(self.masks, f)

    def initialize_mask(self, i, mask_type):
        if i in self.masks:
            raise ValueError(f"Already initialized {i}")
        # NOTE: Some features are both scalars and pixel.
        if mask_type == "scalar":
            self.masks[i] = np.zeros(self.data_size, dtype=np.bool)
        elif mask_type == "pixel":
            self.masks[i] = np.zeros((self.data_size, *self.mask_shape), dtype=np.bool)
        else:
            raise ValueError(f"Unknown mask type {mask_type}")
