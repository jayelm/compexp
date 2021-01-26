"""
Run an image one-off
"""

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import settings
from imageio import imread
import os
import pandas as pd

from loader.data_loader import ade20k
from loader.data_loader.broden import normalize_image
from loader.model_loader import loadmodel
from adversarial_examples.make_size_position import EXAMPLES, SIZE_GRID, XY_GRID

if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )

    args = parser.parse_args()

    model = loadmodel(None)

    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    records = []
    for orig_fname, _, orig_class, new_class, mask_class in EXAMPLES:
        orig_basename = os.path.splitext(orig_fname)[0]
        for x_pos in XY_GRID:
            for y_pos in XY_GRID:
                for size in SIZE_GRID:
                    new_fname = f'{orig_basename}_{x_pos}_{y_pos}_{size}.jpg'
                    full_fname = os.path.join('adversarial_examples', 'size_position', new_fname)

                    img = imread(full_fname)
                    img = normalize_image(img, bgr_mean=[109.5388, 118.6897, 124.6901])[np.newaxis]
                    img = torch.from_numpy(img[:, ::-1, :, :].copy())
                    img.div_(255.0 * 0.224)

                    if settings.GPU:
                        img = img.cuda()

                    preds = model(img)

                    pred = preds[0].argmax().item()
                    score = preds[0].max().item()

                    pred_lbl = ade20k.I2S[pred]

                    success = int(pred_lbl == new_class)
                    failure = int(pred_lbl == orig_class or pred_lbl == mask_class)
                    partial_success = not success and not failure

                    print(f"{new_fname}\t{pred_lbl}: success {success}, partial success {partial_success}, failure {failure}")

                    records.append({
                        'fname': new_fname,
                        'success': success,
                        'partial_success': partial_success,
                        'failure': failure,
                        'predicted': pred_lbl,
                    })

    pd.DataFrame(records).to_csv('adversarial_examples/size_position/results.csv', index=False)
