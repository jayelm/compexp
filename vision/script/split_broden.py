"""
Split broden into its consistuent datasets
"""

import pandas as pd
from operator import itemgetter
import numpy as np

DIM = 224

np.random.seed(0)

index = pd.read_csv(f"./dataset/broden1_{DIM}/index.csv")

dsets = index.image.str.split("/").map(itemgetter(0))

SPLITS = sorted(dsets.unique().tolist())
COMBOS = [("dtd", "opensurfaces")]

for split in SPLITS + COMBOS:
    if isinstance(split, tuple):
        split_name = "_".join(sorted(split))
        splitdf = index[dsets.isin(split)].copy()
    else:
        split_name = split
        splitdf = index[dsets == split].copy()
    print(split_name)

    splitdf.to_csv(
        f"./dataset/broden1_{DIM}/index_{split_name}.csv", index=False, float_format="%d"
    )

    # Random version: shuffle all masks
    splitdf["color"] = np.random.permutation(splitdf["color"].values)
    splitdf["material"] = np.random.permutation(splitdf["material"].values)
    splitdf["texture"] = np.random.permutation(splitdf["texture"].values)
    splitdf["object"] = np.random.permutation(splitdf["object"].values)
    splitdf["part"] = np.random.permutation(splitdf["part"].values)
    splitdf["scene"] = np.random.permutation(splitdf["scene"].values)

    splitdf.to_csv(
        f"./dataset/broden1_{DIM}/index_{split_name}_random.csv",
        index=False,
        float_format="%d",
    )
