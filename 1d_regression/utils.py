import hashlib
import os
import pickle
import random
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from loguru import logger

# plt.rc("text", usetex=True)
plt.rc("font", family="serif")
sns.set_style("ticks")
sns.set_context("talk")


def savefig(name, mode="pdf"):
    FIG_ROOT = "figures"
    os.makedirs(FIG_ROOT, exist_ok=True)
    assert mode in ["pdf", "png"]
    plt.savefig(
        f"{FIG_ROOT}/{name}.{mode}",
        dpi=300,
        quality=100,
        facecolor="none",
        format=mode,
        bbox_inches="tight",
    )


def set_logger(logfile):
    log_format = "{time:MM/DD HH:mm:ss} | {message}"
    logger.remove()
    logger.add(
        sys.stdout, format=log_format, level="INFO", colorize=True,
    )
    logger.add(
        logfile, format=log_format, level="INFO", colorize=True,
    )


def set_seed(seed, cudnn=True):
    """
    Seed everything we can!
    Note that gym environments might need additional seeding (env.seed(seed)),
    and num_workers needs to be set to 1.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # note: the below slows down the code but makes it reproducible
    if (seed is not None) and cudnn:
        torch.backends.cudnn.deterministic = True


def save_obj(obj, name):
    with open(name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + ".pkl", "rb") as f:
        return pickle.load(f)


def get_path_from_args(args):
    """ Returns a unique hash for an argparse object. """
    args_str = str(args)
    path = hashlib.md5(args_str.encode()).hexdigest()
    return path


def get_base_path():
    p = os.path.dirname(os.path.realpath(__file__))
    if os.path.exists(p):
        return p
    raise RuntimeError(
        "I dont know where I am; please specify a path for saving results."
    )


def kernel_regression(xc, yc, xt, gamma=1.0, epsilon=1e-10):
    sqdiff = (xc - xt.transpose(2, 1)).pow(2)
    K = (-sqdiff * gamma).exp() + epsilon
    predictions = (K * yc).sum(1) / K.sum(1)
    return predictions


class Accumulator:
    def __init__(self):
        self.clear()

    def clear(self):
        self.metrics = defaultdict(lambda: [])

    def add(self, key, value):
        self.metrics[key] += value

    def add_dict(self, dict):
        for key, value in dict.items():
            self.add(key, value)

    def mean(self, key):
        return np.mean(self.metrics[key])

    def __getitem__(self, item):
        return self.metrics[item]

    def __setitem__(self, key, value):
        self.metrics[key] = value

    def get_dict(self):
        return copy.deepcopy(dict(self.metrics))

    def items(self):
        return self.metrics.items()

    def __str__(self):
        return str(dict(self.metrics))
