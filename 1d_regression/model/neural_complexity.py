import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .modules import PMA, MultiHeadAttention


def fc_stack(num_layers, input_dim, hidden_dim, output_dim):
    if num_layers == 0:
        return nn.Identity()
    elif num_layers == 1:
        return nn.Linear(input_dim, output_dim)
    else:
        modules = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            modules.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        modules.append(nn.Linear(hidden_dim, output_dim))
        return nn.Sequential(*modules)


class CrossAttEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        dim = args.hid_dim

        self.mlp_v = fc_stack(args.enc_depth, 3, dim, dim)
        self.mlp_qk = fc_stack(args.enc_depth, 2, dim, dim)
        self.attn = MultiHeadAttention(dim, dim, dim, dim, args.num_heads)

    def forward(self, inputs):
        q = self.mlp_qk(inputs["te_xp"])
        k = self.mlp_qk(inputs["tr_xp"])
        v = self.mlp_v(inputs["tr_xyp"])
        out = self.attn(q, k, v)
        return out


class MeanPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert len(x.shape) == 3
        return x.mean(1)


class NeuralComplexity1D(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bs = args.batch_size
        self.encoder = CrossAttEncoder(args)

        if args.pool == "pma":
            self.pool = PMA(dim=args.hid_dim, num_heads=args.num_heads, num_seeds=1)
        elif args.pool == "mean":
            self.pool = MeanPool()

        self.decoder = fc_stack(args.dec_depth, args.hid_dim, args.hid_dim, 1)

    def forward(self, inputs):
        x = self.encoder(inputs)
        x = self.pool(x)
        x = self.decoder(x)
        return x
