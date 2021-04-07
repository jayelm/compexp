"""
Utils
"""


import torch
import shutil
import os
import json


class FakePool:
    def __init__(self, *args, **kwargs):
        pass

    def imap_unordered(self, f, args):
        return map(f, args)

    def close(self):
        pass

    def join(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass


class AverageMeter:
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self, running_avg=False):
        self.reset()
        self.compute_running_avg = running_avg
        if self.compute_running_avg:
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
        if self.compute_running_avg:
            self.update_running_avg(val, n)

    def update_running_avg(self, val, n):
        self.running_val = val
        self.running_sum += val * n
        self.running_count += n
        self.running_avg = self.running_sum / self.running_count

    def __str__(self):
        return f"AverageMeter(mean={self.avg:f}, count={self.count:d})"

    def __repr__(self):
        return str(self)


def save_checkpoint(
    state,
    is_best,
    exp_dir,
    filename="checkpoint.pth",
    best_filename="model_best.pth",
    i=None,
):
    """
    Save a checkpoint
    Parameters
    ----------
    state : ``dict``
        State dictionary consisting of model state to save (generally with
        string keys and state_dict values).
    is_best : ``bool``
        Is the model the best one encountered during this training run?
        If so, copy over the model (normally saved to ``filename`` to
        ``best_filename``.)
    exp_dir : ``str``
        Experiment directory to save to.
    filename : ``str``, optional (default: 'checkpoint.pth')
        Model name to save
    best_filename : ``str``, optional (default: 'model_best.pth')
        Model name to save if this model is the best version
    """
    if i is not None:
        filename = "{}.{}".format(filename, i)
        best_filename = "{}.{}".format(best_filename, i)
    torch.save(state, os.path.join(exp_dir, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(exp_dir, filename), os.path.join(exp_dir, best_filename)
        )


def save_metrics(metrics, exp_dir, filename="metrics.json", i=None):
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
    if i is not None:
        filename = "{}.{}".format(filename, i)
    with open(os.path.join(exp_dir, filename), "w") as f:
        json.dump(dict(metrics), f, indent=4, separators=(",", ": "), sort_keys=True)
