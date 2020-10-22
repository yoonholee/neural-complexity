import argparse
import itertools
import os
from collections import namedtuple
from timeit import default_timer as timer

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from data.get_loader import get_task
from model.neural_complexity import NeuralComplexity1D
from model.nn_learner import get_learner
from utils import set_logger

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default="7")
parser.add_argument("--task", type=str, default="sine")
parser.add_argument("--task-batch-size", type=int, default=128)
parser.add_argument("--test-steps", type=int, default=64)
parser.add_argument("--model-path", type=str)
args, unknown = parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
set_logger("result/regularization.log")

args.batch_size = 512
args.hid_dim = 1024
args.enc_depth = 3
args.pool = "mean"
args.dec_depth = 3
args.num_heads = 8
args.output = "gaussian"
saved_state_dict = torch.load(args.model_path)
model = NeuralComplexity1D(args).cuda()
model.load_state_dict(saved_state_dict)

mse_criterion = nn.MSELoss(reduction="none")
global_timestamp = timer()
tasks = get_task(
    saved=True,
    task=args.task,
    batch_size=args.task_batch_size,
    num_steps=args.test_steps,
)
logger.info(f"Dataset loading took {timer() - global_timestamp:.2f} seconds")


def run_regression(batch, learner_args):
    x_train, y_train = batch["train"][0].cuda(), batch["train"][1].cuda()
    x_test, y_test = batch["test"][0].cuda(), batch["test"][1].cuda()

    bs = x_train.shape[0]
    h = get_learner(
        batch_size=x_train.shape[0],
        layers=learner_args.n_layer,
        hidden_size=learner_args.hid,
        activation=learner_args.act,
        regularizer=learner_args.regularizer,
    ).cuda()
    h_opt = torch.optim.SGD(h.parameters(), lr=learner_args.inner_lr)
    h_crit = nn.MSELoss(reduction="none")
    h.train()

    for _ in range(learner_args.steps):
        preds_train = h(x_train)
        h_loss = h_crit(preds_train.squeeze(), y_train.squeeze()).mean(-1).sum()
        if learner_args.regularizer == "NC":
            preds_test = h(x_test)
            te_xp = torch.cat([x_test, preds_test], dim=-1)
            tr_xp = torch.cat([x_train, preds_train], dim=-1)
            tr_xyp = torch.cat([x_train, y_train, preds_train], dim=-1)
            meta_batch = {"te_xp": te_xp, "tr_xp": tr_xp, "tr_xyp": tr_xyp}

            model_preds = model(meta_batch)
            loc, _scale = model_preds[:, 0], model_preds[:, 1]
            nc_regularization = loc.sum()
            h_loss += nc_regularization
        elif type(learner_args.regularizer) == tuple:
            measure, weight = learner_args.regularizer
            h_loss += h.get_measure(measure).sum() * weight

        h_opt.zero_grad()
        h_loss.backward()
        h_opt.step()
    return h


# steps = [1, 2, 4, 8, 16]
# steps = [4096]
# acts = ["sigmoid", "tanh", "none"]
# hiddens = [5, 10, 20, 80, 160, 320]
# n_layers = [1, 3]
# regularize = itertools.product(["Orthogonal"], [1.0, 1e-1, 1e-2])
# inner_lrs = [1e-5, 1e-4, 1e-3, 1e-1, 1e0]

steps = [16]
n_layers = [2]
hiddens = [40]
acts = ["relu"]
acts = ["sigmoid", "tanh", "none"]
regularize = ["dropout", "v_dropout"]
inner_lrs = [1e-2]

LearnerArgs = namedtuple("LearnerArgs", "steps n_layer hid act regularizer inner_lr")
for _args in itertools.product(steps, n_layers, hiddens, acts, regularize, inner_lrs):
    learner_args = LearnerArgs(*_args)
    logger.info(f"")
    logger.info(learner_args)

    losses_test, losses_train = [], []
    for batch in tasks:
        x_train, y_train = batch["train"][0].cuda(), batch["train"][1].cuda()
        x_test, y_test = batch["test"][0].cuda(), batch["test"][1].cuda()
        h = run_regression(batch, learner_args)
        with torch.no_grad():
            h.eval()
            preds_train = h(x_train)
            preds_test = h(x_test)

        l_test = mse_criterion(preds_test.squeeze(), y_test.squeeze()).mean(-1)
        losses_test.append(l_test)
        l_train = mse_criterion(preds_train.squeeze(), y_train.squeeze()).mean(-1)
        losses_train.append(l_train)

    losses_test = torch.cat(losses_test)
    losses_train = torch.cat(losses_train)
    t_mean = losses_test.mean()
    t_conf = losses_test.std() * 1.96 / np.sqrt(len(losses_test))
    c_mean = losses_train.mean()
    c_conf = losses_train.std() * 1.96 / np.sqrt(len(losses_train))
    logger.info(f"Test {t_mean:.4f} +- {t_conf:.4f}")
    logger.info(f"Train {c_mean:.4f} +- {c_conf:.4f}")
