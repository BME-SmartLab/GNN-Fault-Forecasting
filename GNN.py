import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from GNLayer import GNLayer

class GNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GNLayer(in_channels, out_channels, out_channels, 0)
        self.act = nn.PReLU()
        self.norm = gnn.BatchNorm(out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv(x, edge_index, edge_attr)
        x = self.act(x)
        x = self.norm(x)
        return x

class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
      super().__init__()
      self.in_channels = in_channels
      self.hidden_channels = hidden_channels
      self.out_channels = out_channels
      self.conv1 = GNBlock(in_channels, hidden_channels)
      self.conv2 = GNBlock(hidden_channels, hidden_channels)
      self.conv3 = GNBlock(hidden_channels, hidden_channels)
      self.conv4 = GNBlock(hidden_channels, hidden_channels)
      self.pool = gnn.global_mean_pool
      self.head = gnn.MLP([hidden_channels, hidden_channels, out_channels], norm=None)
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      if self.device != torch.device('cuda'):
        print('WARNING: GPU not available. Using CPU instead.')
      self.to(self.device)
      self.optimizer = None
      self.criterion = F.mse_loss

    def forward(self, data):
      x = data.x
      x = self.conv1(x, data.edge_index)
      x = self.conv2(x, data.edge_index)
      x = self.conv3(x, data.edge_index)
      x = self.conv4(x, data.edge_index)
      x = self.pool(x, data.batch)
      x = self.head(x)
      return x

    def train_batch(self, loader):
      self.train()
      losses = []
      maes = []
      for data in loader:
        self.optimizer.zero_grad()
        data = data.to(self.device)
        out = self(data)
        loss = self.criterion(out, data.y)
        loss.backward()
        self.optimizer.step()
        losses.append(loss.item())
        mae = F.l1_loss(out, data.y)
        maes.append(mae.item())
      return sum(losses)/len(losses), sum(maes)/len(maes)

    @torch.no_grad()
    def test_batch(self, loader):
      self.eval()
      losses = []
      maes = []
      for data in loader:
        data = data.to(self.device)
        out = self(data)
        loss = self.criterion(out, data.y)
        losses.append(loss.item())
        mae = F.l1_loss(out, data.y)
        maes.append(mae.item())
      return sum(losses)/len(losses), sum(maes)/len(maes)

    @torch.no_grad()
    def predict_batch(self, loader):
      self.eval()
      y_preds = []
      y_trues = []
      for data in loader:
        data = data.to(self.device)
        out = self(data)
        y_preds.append(out)
        y_trues.append(data.y)
      y_preds = torch.cat(y_preds).detach().cpu().numpy()
      y_trues = torch.cat(y_trues).detach().cpu().numpy()
      return y_preds, y_trues
