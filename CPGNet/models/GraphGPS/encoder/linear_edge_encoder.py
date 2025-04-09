import torch

class LinearEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.encoder = torch.nn.Linear(2, emb_dim)
    def forward(self, edge_attr):
        edge_attr = self.encoder(edge_attr.view(-1, 2))
        return edge_attr