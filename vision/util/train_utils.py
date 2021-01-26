import subprocess
import os
import torch
import json
import shutil
import contextlib
import warnings


def current_git_hash():
    """
    Get the hash of the latest commit in this repository. Does not account for unstaged changes.
    Returns
    -------
    git_hash : ``str``, optional
        The string corresponding to the current git hash if known, else ``None`` if something failed.
    """
    unstaged_changes = False
    try:
        subprocess.check_output(["git", "diff-index", "--quiet", "HEAD", "--"])
    except subprocess.CalledProcessError as grepexc:
        if grepexc.returncode == 1:
            warnings.warn("Running experiments with unstaged changes.")
            unstaged_changes = True
    except FileNotFoundError:
        warnings.warn("Git not found")
    try:
        git_hash = (
            subprocess.check_output(["git", "describe", "--always"])
            .strip()
            .decode("utf-8")
        )
        return git_hash, unstaged_changes
    except subprocess.CalledProcessError:
        return None, None


def save_metrics(metrics, exp_dir, filename="metrics.json"):
    """
    Load metrics from the given exp directory..
    Parameters
    ----------
    metrics : ``dict``
        Metrics to save
    exp_dir : ``str``
        Folder to load metrics from
    filename : ``str``, optional (default: 'metrics.json')
        Name of metrics file
    """
    with open(os.path.join(exp_dir, filename), "w") as f:
        json.dump(dict(metrics), f, indent=4, separators=(",", ": "), sort_keys=True)


def save_args(args, exp_dir, filename="args.json"):
    """
    Save arguments in the experiment directory. This is REALLY IMPORTANT for
    reproducibility, so you know exactly what configuration of arguments
    resulted in what experiment result! As a bonus, this function also saves
    the current git hash so you know exactly which version of the code produced
    your result (that is, as long as you don't run with unstaged changes).
    Parameters
    ----------
    args : ``argparse.Namespace``
        Arguments to save
    exp_dir : ``str``
        Folder to save args to
    filename : ``str``, optional (default: 'args.json')
        Name of argument file
    """
    args_dict = vars(args)
    args_dict["git_hash"], args_dict["git_unstaged_changes"] = current_git_hash()
    with open(os.path.join(exp_dir, filename), "w") as f:
        json.dump(args_dict, f, indent=4, separators=(",", ": "), sort_keys=True)


class AverageMeter:
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()
        self.reset_running_avg()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset_running_avg(self):
        self.running_val = 0
        self.running_avg = 0
        self.running_sum = 0
        self.running_count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.update_running_avg(val, n)

    def update_running_avg(self, val, n):
        self.running_val = val
        self.running_sum += val * n
        self.running_count += n
        self.running_avg = self.running_sum / self.running_count


def save_model(
    model, is_best, exp_dir, filename="checkpoint.pth", best_filename="model_best.pth"
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.save(model, os.path.join(exp_dir, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(exp_dir, filename), os.path.join(exp_dir, best_filename)
        )
