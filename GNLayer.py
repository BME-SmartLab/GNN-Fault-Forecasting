import torch
import torch.nn as nn
import torch_geometric.nn as gnn

class GNLayer(gnn.MessagePassing):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim=0):
        super(GNLayer, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * in_channels + edge_dim, hidden_channels),
            nn.PReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.PReLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels + hidden_channels, hidden_channels),
            nn.PReLU(),
            nn.Linear(hidden_channels, out_channels),
        )

    def reset_parameters(self):
        super().reset_parameters()
        for layer in self.edge_mlp:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
        for layer in self.node_mlp:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, size=None):
        if edge_attr is None:
            edge_attr = torch.ones(edge_index.size(1), 0).to(x.device)
        agg = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        return out
    
    def message(self, x_i, x_j, edge_attr=None):
        if edge_attr is None:
            out = torch.cat([x_i, x_j], dim=1)
        else:
            out = torch.cat([x_i, x_j, edge_attr], dim=1)
        out = self.edge_mlp(out)
        return out
