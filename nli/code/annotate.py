"""
Annotate a corpus with spacy tags
"""

import spacy
from spacy.tokens.doc import Doc
from spacy_wordnet.wordnet_annotator import WordnetAnnotator
from tqdm import tqdm

# For benepar installation, do
# import benepar; benepar.download('benepar_en2')
from benepar.spacy_plugin import BeneparComponent


def whitespace_tokenizer(text, vocab):
    tokens = text.strip().split(" ")
    return Doc(vocab, tokens)


def extract_lemma(token):
    return token.lemma_


def extract_pos(token):
    return token.pos_


def extract_tag(token):
    return token.tag_


def extract_dep(token):
    return token.dep_


def extract_brown_cluster(token):
    cl = token.cluster
    if cl == 0:
        return ""
    return str(cl)


def extract_ent(token):
    """
    Extract wordnet synset if present
    """
    return token.ent_type_


def extract_synset(token):
    """
    Extract wordnet synset if present
    """
    ss = token._.wordnet.synsets()
    if not ss:
        return ""
    return ss[0].name()


def extract_constituents(token):
    """
    Extract the constituents this token is a member of

    NOTE: I don't distinguish toplevel S from nested S. But root dep probably
    does (maybe I should differentiate?)
    """
    consts = []
    parent = token._.parent
    if parent is None:
        return ""
    parent_labels = parent._.labels
    # A span may have multiple labels when there are unary chains in the parse tree.
    while parent_labels[0] != "S":
        consts.extend(parent_labels)
        parent = parent._.parent
        if parent is None:
            break
        parent_labels = parent._.labels
    return ";".join(consts[::-1])


# List of feature names, whether they are open (approx. linear in size of
# vocab) or closed (constant size regardless of vocab), multi (multi-hot) and
# functions
FEATS = [
    ("lemma", "open", extract_lemma),
    #  ("pos", "closed", extract_pos),
    ("tag", "closed", extract_tag),
    ("dep", "closed", extract_dep),
    ("ent", "closed", extract_ent),
    #  ("brown", "closed", extract_brown_cluster),
    ("synset", "open", extract_synset),
    ("const", "multi", extract_constituents),
]


def extract_feats(token):
    """
    Extract features from a token
    """
    feats = []
    for (*name, f) in FEATS:
        feats.append(f(token))
    return feats


def main(args):
    if args.cuda:
        spacy.require_gpu()
    # Load an spacy model (supported models are "es" and "en")
    print("Loading spacy...")
    nlp = spacy.load("en_core_web_lg")
    print("Done")
    nlp.tokenizer = lambda text: whitespace_tokenizer(text, nlp.vocab)
    nlp.add_pipe(WordnetAnnotator(nlp.lang), after="tagger")
    nlp.add_pipe(BeneparComponent("benepar_en2"))

    with open(args.data) as f:
        lines = [line.strip() for line in list(f)]

    all_texts = []
    all_feats = []
    docs = nlp.pipe(lines, batch_size=args.batch_size)
    for doc in tqdm(docs, desc="Extracting feats", total=len(lines)):
        doc_feats = []
        doc_texts = []
        for token in doc:
            t_feats = extract_feats(token)
            doc_feats.append(t_feats)
            doc_texts.append(token.text)
        all_feats.append(doc_feats)
        all_texts.append(doc_texts)

    with open(args.data.replace(".tok", ".feats"), "w") as f:
        f.write("|".join((";".join(fn[:2]) for fn in FEATS)))
        f.write("\n")
        for text, doc_feats in zip(all_texts, all_feats):
            t_feats_joined = ["|".join(tf) for tf in doc_feats]
            line_feats = " ".join(
                ["|".join((t, f)) for t, f in zip(text, t_feats_joined)]
            )
            f.write(line_feats)
            f.write("\n")


if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--data", default="data/analysis/snli_1.0_dev.tok")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)

    args = parser.parse_args()

    main(args)
