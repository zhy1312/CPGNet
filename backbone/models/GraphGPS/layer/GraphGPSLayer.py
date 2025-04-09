import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GINConv, GINEConv, GATv2Conv, GATConv, GraphConv

# from performer_pytorch import SelfAttention
from .GatedGCN import GatedGCNConv

# from .SparseMHA import SparseMHA
from .utils import to_dense_batch
import scipy

# from mamba_ssm import Mamba


class GraphGPSLayer(nn.Module):
    """Graph Transformer Layer"""

    def __init__(
        self,
        hidden_size,
        num_heads,
        local_gnn,
        attn_type,
        dropout=0,
        batch_norm=True,
        heatmap=False,
    ):
        super().__init__()
        self.activation = nn.ReLU()
        self.heatmap = heatmap
        # Local GNN model
        self.local_gnn_with_edge_attr = True
        self.local_gnn = local_gnn
        if self.local_gnn == "None":
            self.local_model = None
        elif self.local_gnn == "GIN":
            gin_nn = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                self.activation,
                nn.Linear(hidden_size, hidden_size),
            )
            self.local_model = GINConv(gin_nn)
            self.local_gnn_with_edge_attr = False
        elif self.local_gnn == "GINE":
            gin_nn = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                self.activation,
                nn.Linear(hidden_size, hidden_size),
            )
            self.local_model = GINEConv(gin_nn)
        elif self.local_gnn == "GatedGCN":
            self.local_model = GatedGCNConv(
                hidden_size,
                hidden_size,
                hidden_size,
                dropout,
                batch_norm=True,
                residual=True,
            )
        elif self.local_gnn == "GATv2":
            self.local_model = GATv2Conv(
                in_feats=hidden_size,
                out_feats=hidden_size // num_heads,
                num_heads=num_heads,
                feat_drop=0.5,
                attn_drop=dropout,
                negative_slope=0.2,
                activation=F.leaky_relu,
            )
            self.local_gnn_with_edge_attr = False
        elif self.local_gnn == "GCN":
            self.local_model = GraphConv(hidden_size, hidden_size, activation=F.relu)
            self.local_gnn_with_edge_attr = False
        elif self.local_gnn == "GAT":
            self.local_model = GATConv(
                in_feats=hidden_size,
                out_feats=hidden_size // num_heads,
                num_heads=num_heads,
                feat_drop=0.5,
                attn_drop=dropout,
                negative_slope=0.2,
                activation=F.leaky_relu,
            )
            self.local_gnn_with_edge_attr = False
        else:
            raise ValueError(f"Unsupported Local GNN model {self.local_gnn}")

        # Global attention transformer model
        self.attn_type = attn_type
        if attn_type == "None":
            self.global_attn = None
        elif attn_type == "Transformer":
            self.global_attn = torch.nn.MultiheadAttention(
                hidden_size, num_heads, dropout, batch_first=True
            )

        elif attn_type == "Performer":
            self.global_attn = SelfAttention(
                dim=hidden_size, heads=num_heads, dropout=dropout, causal=False
            )
        elif attn_type == "Mamba":
            self.global_attn = Mamba(
                d_model=hidden_size,  # Model dimension d_model
                d_state=16,  # SSM state expansion factor
                d_conv=4,  # Local convolution width
                expand=2,  # Block expansion factor
            )
        else:
            raise ValueError(
                f"Unsupported Global Attention Transformer model {attn_type}"
            )

        self.batch_norm = batch_norm
        if self.batch_norm:
            self.norm_local = nn.LayerNorm(hidden_size)
            self.norm_attn = nn.LayerNorm(hidden_size)
            self.norm_manba = RMSNorm(hidden_size)
            self.norm_out = nn.LayerNorm(hidden_size)

        self.FFN1 = nn.Linear(hidden_size, hidden_size)
        self.FFN2 = nn.Linear(hidden_size, hidden_size)

        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

    def forward(self, g, h, e):
        h_in = h

        # Local MPNN
        if self.local_model is not None:
            if self.local_gnn == "GatedGCN":
                h_local, e = self.local_model(g, h, e)
            elif self.local_gnn_with_edge_attr:
                h_local = self.local_model(g, h, e)
            elif self.local_gnn == "GATv2" or self.local_gnn == "GAT":
                h_local = self.local_model(g, h).flatten(1)
            else:
                h_local = self.local_model(g, h)
            h_local = self.dropout_local(h_local)
            h_local = h_in + h_local
            if self.batch_norm:
                h_local = self.norm_local(h_local)
                # h_local = self.norm_manba(h_local)

        # Multi-head attention
        if self.global_attn is not None:
            h_attn = self.attn_block(g, h)
            h_attn = self.dropout_attn(h_attn)
            h_attn = h_in + h_attn
            if self.batch_norm:
                h_attn = self.norm_attn(h_attn)
            # h_attn = self.norm_manba(h_attn)

        # Combine the local and global outputs
        if self.local_model is not None and self.global_attn is not None:
            h = h_local + h_attn
        elif self.local_model is None:
            h = h_attn
        elif self.global_attn is None:
            h = h_local
        else:
            raise ValueError("Invalid combination of local or global attention")
        h = h + self.FFN2(F.relu(self.FFN1(h)))
        if self.batch_norm:
            h = self.norm_out(h)

        return h, e

    def attn_block(self, g, h):
        if self.attn_type == "Transformer":
            h_dense, mask = to_dense_batch(
                h, g.batch_num_nodes(), batch_size=g.batch_size, num_nodes=g.num_nodes()
            )
            x= self.global_attn(
                h_dense, h_dense, h_dense, key_padding_mask=~mask, need_weights=False
            )[0]
            h_attn = x[mask]
            # A = A[mask]
        elif self.attn_type == "Performer":
            h_dense, mask = to_dense_batch(
                h, g.batch_num_nodes(), batch_size=g.batch_size, num_nodes=g.num_nodes()
            )
            h_attn = self.global_attn(h_dense, mask=mask)[mask]
        elif self.attn_type == "Mamba":
            h, mask = to_dense_batch(
                h, g.batch_num_nodes(), batch_size=g.batch_size, num_nodes=g.num_nodes()
            )
            h_attn = self.global_attn(h)[mask]

        return h_attn


class RMSNorm(nn.Module):
    """
    均方根归一化： RMS normalization
    """

    def __init__(
        self, d_model: int, eps: float = 1e-5  # hidden dim
    ):  # 防止除以零的小数值
        super().__init__()
        self.eps = eps
        # weight: 可学习的参数，调整归一化后的值
        self.weight = nn.Parameter(
            torch.ones(d_model)
        )  # 初始值为大小为d_model的张量，每个元素的值都是1

    def forward(self, x):
        """
        :param x: 输入张量
        :return: output 均方根规划化后的值

        RMS的计算步骤：
        Step1: 计算每个样本的均方根值
            Step1.1: 先计算x的平方
            Step1.2: 沿着最后一个维度（通常是特征维度）计算平均值，并加上一个很小的数eps
            Step1.3: 最后取平方根
        Step2: 对每个样本进行归一化
            Step2.1：每个特征值除以其所在样本的均方根值
            Step2.2: 最后乘以可以学习的权重weight,得到最终的输出
        """
        output = (
            x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        )
        return output
