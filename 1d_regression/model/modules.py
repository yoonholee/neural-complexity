import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_query, dim_key, dim_value, dim_output, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_query, dim_output, bias=False)
        self.fc_k = nn.Linear(dim_key, dim_output, bias=False)
        self.fc_v = nn.Linear(dim_value, dim_output, bias=False)
        self.fc_o = nn.Linear(dim_output, dim_output)

    def forward(self, query, key, value, mask=None):
        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)

        query_ = torch.cat(query.chunk(self.num_heads, -1), 0)
        key_ = torch.cat(key.chunk(self.num_heads, -1), 0)
        value_ = torch.cat(value.chunk(self.num_heads, -1), 0)

        A_logits = (query_ @ key_.transpose(-2, -1)) / math.sqrt(query.shape[-1])
        if mask is not None:
            mask = torch.stack([mask.squeeze(-1)] * query.shape[-2], -2)
            mask = torch.cat([mask] * self.num_heads, 0)
            A_logits.masked_fill(mask, -float("inf"))
            A = torch.softmax(A_logits, -1)
        else:
            A = torch.softmax(A_logits, -1)

        outs = torch.cat((A @ value_).chunk(self.num_heads, 0), -1)
        outs = query + outs
        outs = outs + F.relu(self.fc_o(outs))
        return outs


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds):
        super().__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mha = MultiHeadAttention(dim, dim, dim, dim, num_heads)

    def forward(self, X):
        batch_size = X.size(0)
        query = self.S.repeat(batch_size, 1, 1)
        return self.mha(query, X, X).squeeze()
