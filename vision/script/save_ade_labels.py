"""
Get ground-truth places365 labels by filename
"""

import pandas as pd
import glob
import os
import json
import string

if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )

    args = parser.parse_args()

    ade_path = "dataset/ADE20K_2016_07_26/images/"

    img2scene = {}

    for split in ["training", "validation"]:
        for l in string.ascii_lowercase:
            l_dir = os.path.join(ade_path, split, l)
            if not os.path.exists(l_dir):
                continue

            for scene in os.listdir(l_dir):
                scene_dir = os.path.join(l_dir, scene)
                for img in os.listdir(scene_dir):
                    maybe_new_scene_dir = os.path.join(scene_dir, img)
                    if os.path.isdir(maybe_new_scene_dir):
                        # Possible to go two deep
                        new_scene = f"{scene}-{img}"
                        for subimg in os.listdir(maybe_new_scene_dir):
                            if subimg.endswith(".jpg"):
                                img2scene[subimg] = new_scene
                    elif img.endswith(".jpg"):
                        img2scene[img] = scene

    with open("dataset/broden1_224/ade20k_scenes.json", "w") as f:
        json.dump(img2scene, f)
