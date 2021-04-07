"""
Model logical form distillation
"""

import pandas as pd

import settings
import os
import data
import analyze
from analyze import get_mask
import formula as FM


def main():
    # Should have results already.
    result_file = os.path.join(settings.RESULT, "result.csv")
    if not os.path.exists(result_file):
        raise ValueError(
            f"Need a result file for analysis, couldn't find {result_file}"
        )
    result = pd.read_csv(result_file)
    result.sort_values("neuron", inplace=True)

    model, dataset = data.snli.load_for_analysis(
        settings.MODEL,
        settings.DATA,
        model_type=settings.MODEL_TYPE,
        cuda=settings.CUDA,
    )
    toks, states, feats, idxs = analyze.extract_features(
        model,
        dataset,
        cache_file=None,
        pack_sequences=False,
    )

    token_masks, tok_feats_vocab = analyze.to_sentence(toks, feats, dataset)

    def reverse_namer(i):
        return tok_feats_vocab["itos"][i]

    quantiles = analyze.get_quantiles(states, 0.01)

    # Now get mask
    breakpoint()
    masks = [FM.parse(x, reverse_namer) for x in result["feature"]]


if __name__ == "__main__":
    main()
