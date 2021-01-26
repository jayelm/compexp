"""
Vary size and position of masks
"""


from PIL import Image
import numpy as np
import os


EXAMPLES = [
    # orig_fname, mask_fname, orig_class, intended_class, mask_class
    ('corridor_orig.jpg', 'igloo_mask.png', 'corridor-s', 'clean_room-s', 'igloo-s'),
    ('street_orig.jpg', 'nursery_mask.png', 'street-s', 'fire_escape-s', 'nursery-s'),
    ('forest_path_orig.jpg', 'laundromat_mask.png', 'forest_path-s', 'viaduct-s', 'laundromat-s'),
]

XY_GRID = [-150, -100, -50, 0, 50, 100]
SIZE_GRID = [int(x * 224) for x in [0.33, 0.5, 0.75, 1, 1.5, 2]]


if __name__ == '__main__':
    os.makedirs('size_position', exist_ok=True)
    for orig_fname, mask_fname, *_ in EXAMPLES:
        orig_basename = os.path.splitext(orig_fname)[0]

        orig = Image.open(orig_fname)
        mask = Image.open(mask_fname)
        mask_values = Image.fromarray(np.array(mask)[:, :, -1], mode='L')
        mask = mask.convert('RGB')

        for x_pos in XY_GRID:
            for y_pos in XY_GRID:
                for size in SIZE_GRID:
                    mask_transformed = mask.resize((size, size))
                    mask_values_transformed = mask_values.resize((size, size))
                    orig_masked = orig.copy()
                    orig_masked.paste(mask_transformed, (x_pos, y_pos), mask_values_transformed)
                    new_fname = os.path.join('size_position', f'{orig_basename}_{x_pos}_{y_pos}_{size}.jpg')
                    orig_masked.save(new_fname)
