"""
Train a bowman et al-style SNLI model
"""


import os
import torch
import torch.optim as optim
import torch.nn as nn

from torch.utils.data import DataLoader
from data.snli import SNLI, pad_collate
from contextlib import nullcontext
from tqdm import tqdm
import numpy as np
from collections import defaultdict


import models
import util


def run(split, epoch, model, optimizer, criterion, dataloaders, args):
    training = split == "train"
    if training:
        ctx = nullcontext
        model.train()
    else:
        ctx = torch.no_grad
        model.eval()

    ranger = tqdm(dataloaders[split], desc=f"{split} epoch {epoch}")

    loss_meter = util.AverageMeter()
    acc_meter = util.AverageMeter()
    for (s1, s1len, s2, s2len, targets) in ranger:

        if args.cuda:
            s1 = s1.cuda()
            s1len = s1len.cuda()
            s2 = s2.cuda()
            s2len = s2len.cuda()
            targets = targets.cuda()

        batch_size = targets.shape[0]

        with ctx():
            logits = model(s1, s1len, s2, s2len)
            loss = criterion(logits, targets)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        preds = logits.argmax(1)
        acc = (preds == targets).float().mean()

        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc.item(), batch_size)

        ranger.set_description(
            f"{split} epoch {epoch} loss {loss_meter.avg:.3f} acc {acc_meter.avg:.3f}"
        )

    return {"loss": loss_meter.avg, "acc": acc_meter.avg}


def build_model(vocab_size, model_type, embedding_dim=300, hidden_dim=512):
    """
    Build a bowman-style SNLI model
    """
    enc = models.TextEncoder(
        vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim
    )
    if model_type == "minimal":
        model = models.EntailmentClassifier(enc)
    else:
        model = models.BowmanEntailmentClassifier(enc)
    return model


def serialize(model, dataset):
    return {
        "state_dict": model.state_dict(),
        "stoi": dataset.stoi,
        "itos": dataset.itos,
    }


def main(args):
    os.makedirs(args.exp_dir, exist_ok=True)

    # ==== LOAD DATA ====
    if args.debug:
        max_data = 1000
    else:
        max_data = None
    train = SNLI("data/snli_1.0/", "train", max_data=max_data)
    val = SNLI(
        "data/snli_1.0/", "dev", max_data=max_data, vocab=(train.stoi, train.itos)
    )

    dataloaders = {
        "train": DataLoader(
            train,
            batch_size=100,
            shuffle=True,
            pin_memory=False,
            num_workers=0,
            collate_fn=pad_collate,
        ),
        "val": DataLoader(
            val,
            batch_size=100,
            shuffle=False,
            pin_memory=True,
            num_workers=0,
            collate_fn=pad_collate,
        ),
    }

    # ==== BUILD MODEL ====
    model = build_model(
        len(train.stoi),
        args.model_type,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
    )

    if args.cuda:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    metrics = defaultdict(list)
    metrics["best_val_epoch"] = 0
    metrics["best_val_acc"] = 0
    metrics["best_val_loss"] = np.inf

    # Save model with 0 training
    util.save_checkpoint(serialize(model, train), False, args.exp_dir, filename="0.pth")

    # ==== TRAIN ====
    for epoch in range(args.epochs):
        train_metrics = run(
            "train", epoch, model, optimizer, criterion, dataloaders, args
        )
        val_metrics = run("val", epoch, model, optimizer, criterion, dataloaders, args)

        for name, val in train_metrics.items():
            metrics[f"train_{name}"].append(val)

        for name, val in val_metrics.items():
            metrics[f"val_{name}"].append(val)

        is_best = val_metrics["acc"] > metrics["best_val_acc"]

        if is_best:
            metrics["best_val_epoch"] = epoch
            metrics["best_val_acc"] = val_metrics["acc"]
            metrics["best_val_loss"] = val_metrics["loss"]

        util.save_metrics(metrics, args.exp_dir)
        util.save_checkpoint(serialize(model, train), is_best, args.exp_dir)
        if epoch % args.save_every == 0:
            util.save_checkpoint(
                serialize(model, train), False, args.exp_dir, filename=f"{epoch}.pth"
            )


def parse_args():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--exp_dir", default="models/snli/")
    parser.add_argument("--model_type", default="bowman", choices=["bowman", "minimal"])
    parser.add_argument("--save_every", default=1, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--embedding_dim", default=300, type=int)
    parser.add_argument("--hidden_dim", default=512, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
