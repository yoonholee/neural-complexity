import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


def get_learner(batch_size, layers, hidden_size, activation, regularizer=None):
    if activation == "relu":
        activation = nn.ReLU
    elif activation == "sigmoid":
        activation = nn.Sigmoid
    elif activation == "tanh":
        activation = nn.Tanh
    elif activation == "none":
        activation = nn.Identity
    else:
        raise ValueError(f"activation={activation} not implemented!")

    return ParallelNeuralNetwork(
        batch_size,
        num_layers=layers,
        hidden_size=hidden_size,
        activation=activation,
        regularizer=regularizer,
    )


class ParallelLinear(nn.Module):
    def __init__(self, bs, input_size, output_size):
        super().__init__()
        fcs = [nn.Linear(input_size, output_size) for _ in range(bs)]
        self.weight = Parameter(torch.stack([m.weight for m in fcs]))
        self.bias = Parameter(torch.stack([m.bias for m in fcs]).unsqueeze(1))

    def forward(self, x):
        return torch.einsum("bnd,bmd->bnm", x, self.weight) + self.bias


class ParallelNeuralNetwork(nn.Module):
    """ Equivalent to running args.batch_size neural networks in parallel. No weight sharing. """

    def __init__(self, bs, num_layers, hidden_size, activation, regularizer):
        super().__init__()
        self.bs = bs
        modules = [ParallelLinear(bs, 1, hidden_size)]
        for _ in range(num_layers - 1):
            modules.append(activation())
            modules.append(ParallelLinear(bs, hidden_size, hidden_size))
            if regularizer == "dropout":
                modules.append(nn.Dropout())
            if regularizer == "g_dropout":
                modules.append(GaussianDropout(alpha=1.0))
            if regularizer == "v_dropout":
                modules.append(VariationalDropout(alpha=1.0, dim=hidden_size))
            if regularizer == "alpha_dropout":
                modules.append(nn.AlphaDropout(p=0.5))
            if regularizer == "batchnorm":
                # Parallel batchnorm is equivalent to layernorm in this case
                modules.append(nn.LayerNorm(hidden_size, elementwise_affine=False))
        modules.append(activation())
        modules.append(ParallelLinear(bs, hidden_size, 1))
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        if x.shape[0] != self.bs:
            assert x.shape[0] == 1
            x = x.repeat(self.bs, 1, 1)
        return self.net(x)

    @staticmethod
    def l1(weight):
        return weight.view(weight.shape[0], -1).abs().sum(-1)

    @staticmethod
    def l2(weight):
        return weight.view(weight.shape[0], -1).pow(2).sum(-1)

    @staticmethod
    def norm(weight, p=2, q=2):
        return weight.norm(p=p, dim=2).norm(q, dim=1)

    @staticmethod
    def op_norm(weight, p=float("Inf")):
        _, S, _ = weight.svd()
        return S.norm(p, dim=-1)

    @staticmethod
    def orthogonal_loss(weight):
        bs, n, _ = weight.shape
        sym = torch.bmm(weight, weight.transpose(2, 1))
        eyes = [torch.eye(n, device="cuda") for _ in range(bs)]
        sym -= torch.stack(eyes)
        return sym.abs().sum()

    def get_measure(self, name):
        # https://github.com/bneyshabur/generalization-bounds/blob/master/measures.py
        linears = [p for p in self.modules() if isinstance(p, ParallelLinear)]
        ws = [p.weight for p in linears]
        bs = [p.bias for p in linears]
        ps = ws + bs

        inf = float("Inf")

        if name == "L1":
            return torch.stack([self.l1(p) for p in ps]).sum(0)
        elif name == "L2":
            return torch.stack([self.l2(p) for p in ps]).sum(0)
        elif name == "L_{1,inf}":
            return torch.stack([self.norm(w, p=1, q=inf) for w in ws]).prod(0)
        elif name == "Frobenius":
            return torch.stack([self.norm(w, p=2, q=2) for w in ws]).prod(0)
        elif name == "L_{3,1.5}":
            return torch.stack([self.norm(w, p=3, q=1.5) for w in ws]).prod(0)
        elif name == "Orthogonal":
            # https://arxiv.org/abs/1609.07093
            return torch.stack([self.orthogonal_loss(w) for w in ws]).sum()
        elif name == "Spectral":
            return torch.stack([self.op_norm(w, p=inf) for w in ws]).prod(0)
        elif name == "L_1.5_op":
            return torch.stack([self.op_norm(w, p=1.5) for w in ws]).prod(0)
        elif name == "Trace":
            return torch.stack([self.op_norm(w, p=1) for w in ws]).prod(0)
        else:
            raise ValueError(f"Measure {name} is not implemented.")

    def get_measures(self):
        measure_names = [
            "L1",
            "L2",
            "L_{1,inf}",
            "Frobenius",
            "L_{3,1.5}",
            # "Spectral",
            # "L_1.5_op",
            # "Trace",
        ]
        return {name: self.get_measure(name) for name in measure_names}


class GaussianDropout(nn.Module):
    def __init__(self, alpha=1.0):
        super(GaussianDropout, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        if self.train():
            epsilon = torch.randn(x.size()) * self.alpha + 1
            return x * epsilon.cuda()
        else:
            return x


class VariationalDropout(nn.Module):
    def __init__(self, alpha=1.0, dim=None):
        super(VariationalDropout, self).__init__()

        self.dim = dim
        self.max_alpha = alpha
        log_alpha = (torch.ones(dim) * alpha).log()
        self.log_alpha = nn.Parameter(log_alpha)

    def kl(self):
        c1 = 1.16145124
        c2 = -1.50204118
        c3 = 0.58629921

        alpha = self.log_alpha.exp()

        negative_kl = (
            0.5 * self.log_alpha + c1 * alpha + c2 * alpha ** 2 + c3 * alpha ** 3
        )

        kl = -negative_kl

        return kl.mean()

    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        if self.train():
            # N(0,1)
            epsilon = torch.randn(x.size()).cuda()

            # Clip alpha
            self.log_alpha.data = torch.clamp(self.log_alpha.data, max=self.max_alpha)
            alpha = self.log_alpha.exp()

            # N(1, alpha)
            epsilon = epsilon * alpha

            return x * epsilon
        else:
            return x
