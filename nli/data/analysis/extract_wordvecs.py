"""
Extract word vectors for lemmas that appear in 10k dataset
"""

import spacy
from spacy.tokens.doc import Doc
import numpy as np


def whitespace_tokenizer(text, vocab):
    tokens = text.strip().split(" ")
    return Doc(vocab, tokens)


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description='extract wordvecs',
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('path')

    args = parser.parse_args()

    nlp = spacy.load('en_core_web_lg', disable=['parser', 'tagger', 'ner'])
    nlp.tokenizer = lambda text: whitespace_tokenizer(text, nlp.vocab)

    assert args.path.endswith('.tok')

    with open(args.path) as f:
        lines = [line.strip() for line in list(f)]
        lines = [line for line in lines if line]

    vecs = {}

    docs = nlp.pipe(lines, batch_size=100)
    for doc in docs:
        for token in doc:
            lemma = token.lemma_.lower()
            if lemma not in vecs and not token.is_oov:
                if not np.all(token.vector == 0):
                    vecs[lemma] = token.vector

    with open(args.path.replace('.tok', '.vec'), 'w') as f:
        for tok, v in vecs.items():
            nums = ' '.join(f"{n:f}" for n in v)
            f.write(f"{tok} {nums}\n")
