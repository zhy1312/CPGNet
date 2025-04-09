import torch
import torch.nn as nn
from .linear_node_encoder import LinearNodeEncoder

__all__ = ["ConcatNodeEncoder"]


class RWSENodeEncoder(torch.nn.Module):
    def __init__(self, dim_pe, batch_norm=False) -> None:
        super().__init__()
        self.batch_norm = batch_norm
        if batch_norm:
            self.norm = nn.BatchNorm1d(dim_pe)
        self.pe_encoder = nn.Linear(dim_pe, dim_pe)

    def forward(self, batch):
        if self.batch_norm:
            batch = self.norm(batch)
        batch = self.pe_encoder(batch)
        return batch


class ConcatNodeEncoder(torch.nn.Module):
    def __init__(self, feature_dim, dim_hidden, dim_pe) -> None:
        super().__init__()
        self.dim_pe = dim_pe
        if dim_pe > 0:
            self.encoder1 = LinearNodeEncoder(feature_dim + dim_pe, dim_hidden)
        else:
            self.encoder1 = LinearNodeEncoder(feature_dim, dim_hidden)
        # self.encoder2 = RWSENodeEncoder(dim_pe, batch_norm=batch_norm)

    def forward(self, feature, pe):
        if self.dim_pe > 0:
            h = self.encoder1(torch.concat((feature, pe), dim=1))
        else:
            h = self.encoder1(feature)
        return h
