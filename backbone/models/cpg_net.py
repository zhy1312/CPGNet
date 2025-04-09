from .GLGFI.GLGFI_Module import GLGFIModule
import torch.nn as nn
import torch.nn.init as init
import math


class CpgNet(nn.Module):
    def __init__(
        self,
        feature_dim=1026,
        in_dim=1024,
        out_dim=9,
        n_layers=2,
        n_heads=8,
        dropout=0.1,
        local_gnn="GINE",  # GIN,GINE,GatedGCN,GATv2,GCN,GAT,None
        global_attn="Transformer",  # Transformer,Performer,Mamba,None
        pos_enc_size=0,
        pooling_type="avg",
        heatmap=False,
    ):
        super(CpgNet, self).__init__()
        self.heatmap = heatmap
        self.cpgnet = GLGFIModule(
            feature_dim=feature_dim,
            out_size=out_dim,
            hidden_size=in_dim,
            pos_enc_size=pos_enc_size,
            num_layers=n_layers,
            num_heads=n_heads,
            dropout=dropout,
            local_gnn=local_gnn,
            global_attn=global_attn,
            pooling_type=pooling_type,
            heatmap=heatmap,
        )
        # 对self.gps每层进行初始化
        for layer in self.cpgnet.layers:
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, a=math.sqrt(5))

    def forward(self, input):
        return self.cpgnet(input)
