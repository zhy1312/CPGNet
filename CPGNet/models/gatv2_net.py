"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import torch
import torch.nn as nn

import torch.nn.functional as F
import dgl
from dgl.nn.pytorch.conv import GATv2Conv
from dgl.nn.pytorch.glob import (
    SumPooling,
    AvgPooling,
    MaxPooling,
    GlobalAttentionPooling,
)


class Gatv2Net(nn.Module):
    def __init__(
        self,
        n_layers=2,
        in_dim=1026,
        hidden_dim=512,
        out_dim=7,
        head=8,
        activation=F.leaky_relu,
        feat_drop=0.5,
        attn_drop=0.1,
        negative_slope=0.2,
        residual=True,
        graph_pooling_type="mean",
    ):
        super(Gatv2Net, self).__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        self.activation = activation
        heads = ([head] * n_layers) + [1]
        for l in range(n_layers + 1):
            if l == 0:
                # input projection (no residual)
                self.layers.append(
                    GATv2Conv(
                        in_dim,
                        hidden_dim,
                        heads[0],
                        feat_drop,
                        attn_drop,
                        negative_slope,
                        False,
                        self.activation,
                    )
                )
            elif l == n_layers:  # hidden layers
                # output projection
                self.layers.append(
                    GATv2Conv(
                        hidden_dim * heads[-2],
                        out_dim,
                        heads[-1],
                        feat_drop,
                        attn_drop,
                        negative_slope,
                        residual,
                        None,
                    )
                )
            else:
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.layers.append(
                    GATv2Conv(
                        hidden_dim * heads[l - 1],
                        hidden_dim,
                        heads[l],
                        feat_drop,
                        attn_drop,
                        negative_slope,
                        residual,
                        self.activation,
                    )
                )

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = nn.ModuleList()
        self.pools = nn.ModuleList()
        for layer in range(n_layers + 1):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(in_dim, out_dim))
            else:
                self.linears_prediction.append(
                    nn.Linear(hidden_dim * heads[layer - 1], out_dim)
                )

            if graph_pooling_type == "sum":
                self.pools.append(SumPooling())
            elif graph_pooling_type == "mean":
                self.pools.append(AvgPooling())
            elif graph_pooling_type == "max":
                self.pools.append(MaxPooling())
            elif graph_pooling_type == "att":
                if layer == 0:
                    gate_nn = torch.nn.Linear(in_dim, 1)
                else:
                    gate_nn = torch.nn.Linear(hidden_dim * heads[layer - 1], 1)
                self.pools.append(GlobalAttentionPooling(gate_nn))
            else:
                raise NotImplementedError

    def forward(self, g, h=None):
        if h is None:
            h = g.ndata["feat"]

        h_list = []
        for i, layer in enumerate(self.layers):
            pool_h = self.pools[i](g, h)
            pool_h = self.linears_prediction[i](pool_h)
            h_list.append(pool_h)
            h = layer(g, h).flatten(1)

        out = torch.stack(h_list).mean(0)

        return out
