import argparse
import os
import random
import uuid
from collections import defaultdict
from timeit import default_timer as timer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from data.get_loader import get_task
from model.neural_complexity import NeuralComplexity1D
from model.nn_learner import get_learner
from utils import Accumulator, set_logger


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default="7")
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--task-batch-size", type=int, default=64)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--time-budget", type=int, default=1e10)
parser.add_argument("--task", type=str, default="sine")
parser.add_argument("--nc-regularize", type=str2bool, default=True)

parser.add_argument("--epochs", type=int, default=1000)
parser.add_argument("--train-steps", type=int, default=500)
parser.add_argument("--log-steps", type=int, default=500)
parser.add_argument("--test-steps", type=int, default=250)
parser.add_argument("--learn-freq", type=int, default=10)

parser.add_argument("--inner-lr", type=float, default=1e-2)
parser.add_argument("--inner-steps", type=int, default=16)
parser.add_argument("--nc-weight", type=float, default=1.0)
parser.add_argument("--learner-layers", type=int, default=2)
parser.add_argument("--learner-hidden", type=int, default=40)
parser.add_argument(
    "--learner-act",
    type=str,
    default="relu",
    choices=["relu", "sigmoid", "tanh", "none"],
)

parser.add_argument(
    "--input",
    type=str,
    default="cross_att",
    choices=["cross_att"],
)
parser.add_argument("--enc", type=str, default="fc", choices=["fc"])
parser.add_argument("--pool", type=str, default="mean", choices=["mean", "pma"])
parser.add_argument("--dec", type=str, default="fc", choices=["fc"])

parser.add_argument("--enc-depth", type=int, default=3)
parser.add_argument("--dec-depth", type=int, default=2)
parser.add_argument("--hid-dim", type=int, default=512)
parser.add_argument("--num-heads", type=int, default=8)
args, unknown = parser.parse_known_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

arg_strings = [
    uuid.uuid1().hex[:4],
    f"{args.task}_bs{args.batch_size}lr{args.lr:.1e}",
    f"_tbs{args.task_batch_size}ilr{args.inner_lr:.1e}step{args.inner_steps}",
    f"_lyr{args.learner_layers}h{args.learner_hidden}{args.learner_act}",
    f"_{args.enc_depth}{args.pool}{args.dec_depth}d{args.hid_dim}",
]
if args.nc_weight != 1.0:
    arg_strings.append("w{args.nc_weight}")
if args.pool == "pma":
    arg_strings.append(f"heads{args.num_heads}")
args.log_dir = "result/summary/temp/" + "_".join(arg_strings)
args.model_path = f"{args.log_dir}/model.ckpt"
os.makedirs(args.log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=args.log_dir)
set_logger(f"{args.log_dir}/logs.log")
logger.info(f"unknown={unknown}\n Args: {args}")

model = NeuralComplexity1D(args).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
mse_criterion = nn.MSELoss(reduction="none")
mae_criterion = nn.L1Loss()
accum = Accumulator()
global_timestamp = timer()
global_step = 0

test_tasks = get_task(
    saved=True,
    task=args.task,
    batch_size=args.task_batch_size,
    num_steps=args.test_steps,
)
logger.info(f"Dataset loading took {timer() - global_timestamp:.2f} seconds")


class MemoryBank:
    """
    Memory bank class. Stores snapshots of task learners.
    get_batch() returns a random minibatch of (snapshot, gap) for NC to train on.
    """

    def add(self, te_xp, tr_xp, tr_xyp, gap):
        if not hasattr(self, "te_xp"):
            self.te_xp = te_xp
            self.tr_xp = tr_xp
            self.tr_xyp = tr_xyp
            self.gap = gap
        else:
            self.te_xp = torch.cat([self.te_xp, te_xp], dim=0)
            self.tr_xp = torch.cat([self.tr_xp, tr_xp], dim=0)
            self.tr_xyp = torch.cat([self.tr_xyp, tr_xyp], dim=0)
            self.gap = torch.cat([self.gap, gap], dim=0)

            MEMORY_LIMIT = 1_000_000
            if self.te_xp.shape[0] > MEMORY_LIMIT:
                self.te_xp = self.te_xp[-MEMORY_LIMIT:]
                self.tr_xp = self.tr_xp[-MEMORY_LIMIT:]
                self.tr_xyp = self.tr_xyp[-MEMORY_LIMIT:]
                self.gap = self.gap[-MEMORY_LIMIT:]

    def get_batch(self, batch_size):
        N = self.te_xp.shape[0]
        assert N == self.tr_xp.shape[0]
        assert N == self.tr_xyp.shape[0]
        assert N == self.gap.shape[0]

        idxs = random.sample(range(N), k=batch_size)
        batch = {
            "te_xp": self.te_xp[idxs].cuda(),
            "tr_xp": self.tr_xp[idxs].cuda(),
            "tr_xyp": self.tr_xyp[idxs].cuda(),
        }
        return (batch, self.gap[idxs].cuda())


def run_regression(batch, train=True):
    x_train, y_train = batch["train"][0].cuda(), batch["train"][1].cuda()
    x_test, y_test = batch["test"][0].cuda(), batch["test"][1].cuda()

    h = get_learner(
        batch_size=x_train.shape[0],
        layers=args.learner_layers,
        hidden_size=args.learner_hidden,
        activation=args.learner_act,
    ).cuda()
    h_opt = torch.optim.SGD(h.parameters(), lr=args.inner_lr)
    h_crit = nn.MSELoss(reduction="none")

    for _ in range(args.inner_steps):
        preds_train = h(x_train)
        preds_test = h(x_test)

        te_xp = torch.cat([x_test, preds_test], dim=-1)
        tr_xp = torch.cat([x_train, preds_train], dim=-1)
        tr_xyp = torch.cat([x_train, y_train, preds_train], dim=-1)
        meta_batch = {"te_xp": te_xp, "tr_xp": tr_xp, "tr_xyp": tr_xyp}

        h_loss = h_crit(preds_train.squeeze(), y_train.squeeze()).mean(-1).sum()
        if args.nc_regularize and global_step > args.train_steps * 2:
            model_preds = model(meta_batch)
            # We sum NC outputs across tasks because h_loss is also summed.
            nc_regularization = model_preds.sum()
            h_loss += nc_regularization * args.nc_weight

        h_opt.zero_grad()
        h_loss.backward()
        h_opt.step()

        l_test = mse_criterion(preds_test.squeeze(), y_test.squeeze())
        l_train = mse_criterion(preds_train.squeeze(), y_train.squeeze())
        gap = l_test.mean(-1) - l_train.mean(-1)

        if train:
            memory_bank.add(
                te_xp=te_xp.cpu().detach(),
                tr_xp=tr_xp.cpu().detach(),
                tr_xyp=tr_xyp.cpu().detach(),
                gap=gap.cpu().detach(),
            )
    return h, meta_batch


def test(epoch):
    test_accum = Accumulator()
    for batch in test_tasks:
        h, meta_batch = run_regression(batch, train=False)

        x_train, y_train = batch["train"][0].cuda(), batch["train"][1].cuda()
        x_test, y_test = batch["test"][0].cuda(), batch["test"][1].cuda()
        with torch.no_grad():
            preds_train = h(x_train)
            preds_test = h(x_test)

            l_train = mse_criterion(preds_train.squeeze(), y_train.squeeze())
            l_test = mse_criterion(preds_test.squeeze(), y_test.squeeze())
            gap = l_test.mean(-1) - l_train.mean(-1)

            model_preds = model(meta_batch)
            loss = mse_criterion(model_preds.squeeze(), gap.squeeze()).mean()
            mae = mae_criterion(model_preds.squeeze(), gap.squeeze()).mean()

        test_accum.add_dict(
            {
                "l_test": [l_test.mean(-1).detach().cpu()],
                "l_train": [l_train.mean(-1).detach().cpu()],
                "mae": [mae.item()],
                "loss": [loss.item()],
                "gap": [gap.squeeze().detach().cpu()],
                "pred": [model_preds.squeeze().detach().cpu()],
            }
        )

    all_gaps = torch.cat(test_accum["gap"])
    all_preds = torch.cat(test_accum["pred"])
    R = np.corrcoef(all_gaps, all_preds)[0, 1]
    mean_l_test = torch.cat(test_accum["l_test"]).mean()
    mean_l_train = torch.cat(test_accum["l_train"]).mean()

    writer.add_scalar("test/R", R, epoch)
    writer.add_scalar("test/MAE", test_accum.mean("mae"), epoch)
    writer.add_scalar("test/loss", test_accum.mean("loss"), epoch)
    writer.add_scalar("test/l_test", mean_l_test, epoch)
    writer.add_scalar("test/l_train", mean_l_train, epoch)

    logger.info(f"Test epoch {epoch}")
    logger.info(
        f"mae {test_accum.mean('mae'):.2e} loss {test_accum.mean('loss'):.2e} R {R:.3f} "
        f"l_test {mean_l_test:.2e} l_train {mean_l_train:.2e} "
    )


def train():
    global global_step
    train_loader = get_task(
        saved=False,
        task=args.task,
        batch_size=args.task_batch_size,
        num_steps=args.train_steps,
    )
    for batch in train_loader:
        global_step += 1
        if global_step % args.learn_freq == 0:
            run_regression(batch)

        meta_batch, gap = memory_bank.get_batch(args.batch_size)
        model_preds = model(meta_batch)
        loss = mse_criterion(model_preds.squeeze(), gap.squeeze()).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mae = mae_criterion(model_preds.squeeze(), gap.squeeze())
        accum.add_dict(
            {
                "mae": [mae.item()],
                "loss": [loss.item()],
                "gap": [gap.squeeze().detach().cpu()],
                "pred": [model_preds.squeeze().detach().cpu()],
            }
        )

        if global_step % args.log_steps == 0:
            torch.save(model.state_dict(), args.model_path)

            all_gaps = torch.cat(accum["gap"])
            all_preds = torch.cat(accum["pred"])
            R = np.corrcoef(all_gaps, all_preds)[0, 1]
            logger.info(f"Train Step {global_step}")
            logger.info(
                f"mae {accum.mean('mae'):.2e} loss {accum.mean('loss'):.2e} R {R:.3f} "
            )

            writer.add_scalar("train/R", R, global_step)
            writer.add_scalar("train/MAE", accum.mean("mae"), global_step)
            writer.add_scalar("train/loss", accum.mean("loss"), global_step)
            accum.clear()

        if timer() - global_timestamp > args.time_budget:
            logger.info(f"Stopping at step {global_step}")
            quit()


memory_bank = MemoryBank()
populate_timestamp = timer()
populate_loader = get_task(
    saved=False, task=args.task, batch_size=args.task_batch_size, num_steps=100
)
for batch in populate_loader:
    run_regression(batch)
logger.info(f"Populate time: {timer() - populate_timestamp}")

for epoch in range(args.epochs):
    logger.info(f"Epoch {epoch}")
    logger.info(f"Bank size: {memory_bank.te_xp.shape[0]}")

    test_timestamp = timer()
    out = test(epoch)
    test_elapsed = timer() - test_timestamp

    train_timestamp = timer()
    out = train()
    train_elapsed = timer() - train_timestamp

    writer.add_scalar("time/test_epoch", test_elapsed, epoch)
    writer.add_scalar("time/train_epoch", train_elapsed, epoch)
    logger.info(f"Time: train {train_elapsed:.1f} test {test_elapsed:.1f}")
