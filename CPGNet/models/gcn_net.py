"""GCN using DGL nn package
References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch.glob import (
    SumPooling,
    AvgPooling,
    MaxPooling,
    GlobalAttentionPooling,
)


class GcnNet(nn.Module):
    def __init__(
        self,
        in_dim=1026,
        hidden_dim=256,
        out_dim=7,
        n_layers=3,
        activation=F.gelu,
        dropout=0.2,
        graph_pooling_type="max",
    ):
        super(GcnNet, self).__init__()

        self.in_feats = in_dim
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_dim, hidden_dim, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(hidden_dim, hidden_dim, activation=activation))
        self.dropout = nn.Dropout(p=dropout)
        self.classify = nn.Linear(hidden_dim, out_dim)

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = nn.ModuleList()
        self.pools = nn.ModuleList()
        for layer in range(n_layers + 1):
            if layer == 0:
                self.linears_prediction.append(nn.Linear(in_dim, out_dim))
            else:
                self.linears_prediction.append(nn.Linear(hidden_dim, out_dim))

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
                    gate_nn = torch.nn.Linear(hidden_dim, 1)
                self.pools.append(GlobalAttentionPooling(gate_nn))
            else:
                raise NotImplementedError

    def forward(self, g, h=None):
        if h is None:
            h = g.ndata["feat"]
        print(h.shape)
        h_list = []
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h_list.append(self.linears_prediction[i](self.pools[i](g, h)))
            h = layer(g, h)

        h_list.append(self.classify(self.pools[-1](g, h)))
        L = torch.stack(h_list)
        out = torch.stack(h_list).mean(0)

        return out


if __name__ == "__main__":
    g1 = dgl.graph(([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]))
    g1.ndata["feat"] = torch.rand(10, 1026)
    g2 = dgl.graph(([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]))
    g2.ndata["feat"] = torch.rand(10, 1026)
    g3 = dgl.graph(([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]))
    g3.ndata["feat"] = torch.rand(10, 1026)
    g = dgl.batch([g1, g2, g3])
    print(g)
    model = GcnNet()
    print(model)
    out = model(g)
    print(out.shape)
