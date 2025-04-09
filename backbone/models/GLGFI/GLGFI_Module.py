"""
[Recipe for a General, Powerful, Scalable Graph Transformer]
(http://arxiv.org/abs/2205.12454)
"""

from dgl.nn.pytorch.glob import (
    SumPooling,
    AvgPooling,
    MaxPooling,
    GlobalAttentionPooling,
)
import torch.nn as nn
from .encoder.FeatureEncoder import FeatureEncoder
from .layer.GLGFILayer import GLGFILayer


class GLGFIModule(nn.Module):
    def __init__(
        self,
        feature_dim,
        out_size,
        local_gnn,
        global_attn,
        hidden_size=1024,
        pos_enc_size=20,
        num_layers=10,
        num_heads=8,
        dropout=0.1,
        pooling_type="avg",
        heatmap=False,
    ):
        super().__init__()
        self.heatmap = heatmap
        self.encoder = FeatureEncoder(feature_dim, hidden_size, pos_enc_size)

        self.layers = nn.ModuleList(
            [
                GLGFILayer(
                    hidden_size,
                    num_heads,
                    local_gnn,
                    global_attn,
                    dropout,
                    heatmap=self.heatmap,
                )
                for _ in range(num_layers)
            ]
        )
        if pooling_type == "sum":
            self.pooler = SumPooling()
        elif pooling_type == "avg":
            self.pooler = AvgPooling()
        elif pooling_type == "max":
            self.pooler = MaxPooling()
        elif pooling_type == "attn":
            self.pooler = GlobalAttentionPooling(hidden_size)
        else:
            raise ValueError("Invalid pooling type.")
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Linear(hidden_size // 2, out_size),
        )
        self.image_linear = nn.Linear(hidden_size, hidden_size // 2)
        self.text_linear = nn.Linear(512, hidden_size // 2)

        self.heatmap_result = {}

    def forward(self, input):
        if len(input) == 2:
            g = input[0]
            t = input[1][0]
        else:
            g = input[0]
            t = None
        h, e = self.encoder(g)
        for layer in self.layers:
                h, e = layer(g, h, e)
        if self.heatmap:
            self.heatmap_result["feature_map"] = h.detach().cpu().numpy()
        h = self.pooler(g, h)

        if t is not None:
            h = self.image_linear(h)
            t = self.text_linear(t)
            h = h @ t.t()
        else:
            h = self.predictor(h)
        if self.heatmap:
            self.heatmap_result["text_linear_output"] = t.detach().cpu().numpy()
            self.heatmap_result["image_linear_weight"] = self.image_linear.weight.detach().cpu().numpy()
            return h, self.heatmap_result
        else:
            return h
