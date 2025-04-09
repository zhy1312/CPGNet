import torch

class LinearNodeEncoder(torch.nn.Module):
    def __init__(self,dim_in,emb_dim):
        super().__init__()
        
        self.encoder = torch.nn.Linear(dim_in, emb_dim)

    def forward(self, x):
        x = self.encoder(x)
        return x