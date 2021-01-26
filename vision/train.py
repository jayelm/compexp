"""
Train a model for ade20k classification
"""

import settings
from loader.model_loader import loadmodel
from loader.data_loader.ade20k import load_ade20k, to_dataloader
from tqdm import tqdm, trange
import numpy as np
from collections import defaultdict
import contextlib
import warnings

import os
import torch
from torch import optim, nn
from torch.utils.data import DataLoader

from util import train_utils as tutil


if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--dataset", default="ade20k", choices=["ade20k"],
        help="Dataset to train on"
    )
    parser.add_argument("--batch_size", default=32, type=int, help="Train batch size")
    parser.add_argument("--workers", default=4, type=int, help="Train batch size")
    parser.add_argument("--pretrained", action='store_true')
    parser.add_argument("--epochs", default=50, type=int, help="Training epochs")
    parser.add_argument("--save_every", default=1, type=int, help="Save model every n epochs")
    parser.add_argument("--seed", default=42, type=int, help="Default seed")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    data_dir = 'dataset/ADE20K_2016_07_26/'

    model_name = settings.MODEL
    save_dir = f"./zoo/trained/{model_name}_{args.dataset}_finetune{'_pretrained' if args.pretrained else ''}"
    print(f"Train save dir: {save_dir}")

    torch.manual_seed(args.seed)
    random = np.random.seed(args.seed)

    args = parser.parse_args()

    os.makedirs(save_dir, exist_ok=True)
    tutil.save_args(args, save_dir)

    datasets = load_ade20k(
        data_dir, random_state=random, max_classes=5 if args.debug else None
    )
    dataloaders = {
        s: to_dataloader(d, batch_size=args.batch_size, num_workers=args.workers) for s, d in datasets.items()
    }

    # Always load pretrained
    model = loadmodel(None, pretrained_override=args.pretrained)
    # Replace the last layer
    n_classes = datasets["train"].n_classes
    if settings.MODEL == "resnet18":
        model.fc = nn.Linear(512, n_classes)
    elif settings.MODEL == "resnet101":
        model.fc = nn.Linear(2048, n_classes)
    elif settings.MODEL == "alexnet":
        model.classifier[-1] = nn.Linear(4096, n_classes)
    elif settings.MODEL == "vgg16":
        model.classifier[-1] = nn.Linear(4096, n_classes)
    else:
        raise NotImplementedError


    # Re-move the model on/off GPU
    if settings.GPU:
        model = model.cuda()
    else:
        model = model.cpu()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    def run(split, epoch):
        training = split == "train"
        loader = dataloaders[split]
        meters = {m: tutil.AverageMeter() for m in ["loss", "acc"]}
        if training:
            model.train()
            ctx = contextlib.nullcontext()
        else:
            model.eval()
            ctx = torch.no_grad()

        progress_loader = tqdm(loader)
        with ctx:
            for batch_i, (imgs, classes, *_) in enumerate(progress_loader):
                if settings.GPU:
                    imgs = imgs.cuda()
                    classes = classes.cuda()

                batch_size = imgs.shape[0]
                if training:
                    optimizer.zero_grad()

                logits = model(imgs)
                loss = criterion(logits, classes)

                if training:
                    loss.backward()
                    optimizer.step()

                preds = logits.argmax(1)
                acc = (preds == classes).float().mean()
                meters["loss"].update(loss.item(), batch_size)
                meters["acc"].update(acc.item(), batch_size)

                progress_loader.set_description(
                    f"{split.upper():<6} {epoch:3} loss {meters['loss'].avg:.4f} acc {meters['acc'].avg:.4f}"
                )

        return {k: m.avg for k, m in meters.items()}

    metrics = defaultdict(list)
    metrics["best_val_acc"] = 0.0
    metrics["best_val_loss"] = float("inf")
    metrics["best_epoch"] = 0

    splits = ['train', 'val']
    if 'test' in dataloaders.keys():
        splits.append('test')

    for epoch in trange(args.epochs, desc="Epoch"):
        for split in splits:
            split_metrics = run(split, epoch)
            for m, val in split_metrics.items():
                metrics[f"{split}_{m}"].append(val)
        tqdm.write("")

        if metrics["val_acc"][-1] > metrics["best_val_acc"]:
            metrics["best_val_acc"] = metrics["val_acc"][-1]
            metrics["best_val_loss"] = metrics["val_loss"][-1]
            metrics["best_epoch"] = epoch
            tutil.save_model(model, True, save_dir)
        if epoch % args.save_every == 0:
            tutil.save_model(model, False, save_dir, filename=f"{epoch}.pth")

        tutil.save_metrics(metrics, save_dir)
