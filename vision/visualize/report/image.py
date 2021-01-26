import settings
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageFont
from imageio import imread
import seaborn as sns
import matplotlib.pyplot as plt

plt.switch_backend("agg")


def pairwise_histogram(dist, fname, n=10000):
    # Take a sample of the distances
    samp_n = np.random.randint(len(dist), size=n)
    samp = dist[samp_n]

    plt.figure()
    sns.distplot(samp).set_title("Pairwise Euclidean distances")
    plt.savefig(fname)


def score_histogram(records, fname, title="IoUs"):
    plt.figure()
    scores = [float(r["score"]) for r in records]
    sns.distplot(scores).set_title(title)
    plt.savefig(fname)
