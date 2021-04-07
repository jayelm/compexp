import formula as FM
import pandas as pd

RESULT = [
    "exp/debug_step_5000-3/result.csv",
    "exp/debug_step_25000-3/result.csv",
    "exp/debug_step_50000-3/result.csv",
    "exp/debug_step_75000-3/result.csv",
    "exp/debug_step_100000-3/result.csv",
    "exp/debug_step_50000-1/result.csv",
    "exp/debug_step_50000-2/result.csv",
    "exp/debug_step_50000-3/result.csv",
    "exp/debug_step_50000-4/result.csv",
    "exp/debug_step_50000-5/result.csv",
    "exp/debug_step_50000-6/result.csv",
]

for res in RESULT:
    x = pd.read_csv(res)
    TYPES = ["tag", "ent", "lemma", "synset", "dep", "const"]

    records = []
    for fd in x.to_dict("records"):
        f = fd["feature"]
        tydict = {"neuron": fd["neuron"]}
        for ty in TYPES:
            n_ty = f.count(f"{ty}:")
            tydict[ty] = n_ty
        records.append(tydict)
    records = pd.DataFrame(records)
    records.to_csv(res.replace("result.csv", "types.csv"))
