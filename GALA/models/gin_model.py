"""GIN model stub.

You can implement your own GIN/graph backbone here.
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GINConv, global_add_pool
except Exception as e:  # pragma: no cover
    torch = None
    nn = object
    F = None
    GINConv = None
    global_add_pool = None

class GIN(nn.Module):
    def __init__(self, num_features: int, hidden_dim: int, out_dim: int, num_layers: int = 5, dropout: float = 0.5):
        super().__init__()
        if torch is None:
            raise ImportError("PyTorch/PyG not available. Install requirements first.")

        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            in_dim = num_features if i == 0 else hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.readout = nn.Linear(hidden_dim * num_layers, out_dim)

    def forward(self, x, edge_index, batch):
        layer_pooled = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_pooled.append(global_add_pool(x, batch))
        h = torch.cat(layer_pooled, dim=1)
        return self.readout(h)
