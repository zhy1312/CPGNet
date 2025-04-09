import torch
from .NodeEncoder import ConcatNodeEncoder
from .linear_edge_encoder import LinearEdgeEncoder

__all__ = ["FeatureEncoder"]


class FeatureEncoder(torch.nn.Module):
    def __init__(self, feature_dim, dim_hidden, dim_pe, batch_norm=False) -> None:
        super().__init__()
        self.node_encoder = ConcatNodeEncoder(feature_dim, dim_hidden, dim_pe)
        self.edge_encoder = LinearEdgeEncoder(dim_hidden)

    def forward(self, g):
        h = self.node_encoder(g.ndata["feat"], g.ndata["PE"])
        edge_attr = self.edge_encoder(g.edata["feat"])
        return h, edge_attr
