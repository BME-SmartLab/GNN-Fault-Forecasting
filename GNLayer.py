import torch
import torch.nn as nn

class GNLayer(nn.Module):
    def __init__(self, indim, hiddendim, outdim, edgedim=0):
        super(GNLayer, self).__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * indim + edgedim, hiddendim),
            nn.PReLU(),
            nn.Linear(hiddendim, hiddendim),
            nn.PReLU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(indim + hiddendim, hiddendim),
            nn.PReLU(),
            nn.Linear(hiddendim, outdim),
        )

    def segment_sum(self, data, segment_ids, num_segments):
      result_shape = (num_segments, data.size(1))
      result = data.new_full(result_shape, 0)
      segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
      result.scatter_add_(0, segment_ids, data)
      return result

    def edge_model(self, source, target, edge_attr=None):
        if edge_attr is None:
            out = torch.cat([source, target], dim=1)
        else:
          out = torch.cat([source, target, edge_attr], dim=1)
        out = self.edge_mlp(out)
        return out

    def node_model(self, x, edge_index, edge_attr):
        row, col = edge_index
        agg = self.segment_sum(edge_attr, row, num_segments=x.size(0))
        agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        return out

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        edge_attr = self.edge_model(x[row], x[col], edge_attr)
        x = self.node_model(x, edge_index, edge_attr)
        return x
