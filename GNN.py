import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from GNLayer import GNLayer
from utilities import tensor_to_list

class MLPBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.PReLU(),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x):
        return self.mlp(x)

class GNBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv = GNLayer(in_channels, hidden_channels, out_channels, 0)
        self.act = nn.PReLU()
        self.norm = gnn.BatchNorm(out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv(x, edge_index, edge_attr)
        x = self.act(x)
        x = self.norm(x)
        return x

class GNN(nn.Module):
    def __init__(self, learning_rate, weight_decay, in_channels, num_layers, hidden_channels, out_channels):
      super().__init__()
      self.learning_rate = learning_rate
      self.weight_decay = weight_decay
      self.in_channels = in_channels
      self.num_layers = num_layers
      self.hidden_channels = hidden_channels
      self.out_channels = out_channels
      self.conv0 = GNBlock(in_channels, hidden_channels, hidden_channels)
      for i in range(num_layers-1):
        setattr(self, f'conv{i+1}', GNBlock(hidden_channels, hidden_channels, hidden_channels))
      self.head = MLPBlock(hidden_channels, hidden_channels, out_channels)
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      if self.device != torch.device('cuda'):
        print('WARNING: GPU not available. Using CPU instead.')
      self.to(self.device)
      self.optimizer =  torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
      self.criterion = F.mse_loss
    
    def forward(self, data):
      x = data.x
      for i in range(self.num_layers):
        x = getattr(self, f'conv{i}')(x, data.edge_index, data.edge_attr)
      x = gnn.global_mean_pool(x, data.batch)
      x = self.head(x)
      return x

    def train_batch(self, loader):
      self.train()
      losses = []
      maes = []
      for data in loader:
        data = data.to(self.device)
        self.optimizer.zero_grad()
        logits = self(data)
        loss = self.criterion(logits, data.y)
        loss.backward()
        self.optimizer.step()
        losses.append(loss.item())
        mae = F.l1_loss(logits, data.y)
        maes.append(mae.item())
      return sum(losses)/len(losses), sum(maes)/len(maes)

    @torch.no_grad()
    def test_batch(self, loader):
      self.eval()
      losses = []
      maes = []
      for data in loader:
        data = data.to(self.device)
        logits = self(data)
        loss = self.criterion(logits, data.y)
        losses.append(loss.item())
        mae = F.l1_loss(logits, data.y)
        maes.append(mae.item())
      return sum(losses)/len(losses), sum(maes)/len(maes)

    @torch.no_grad()
    def predict_batch(self, loader):
      self.eval()
      y_preds = []
      y_trues = []
      for data in loader:
        data = data.to(self.device)
        logits = self(data)
        y_preds.append(logits)
        y_trues.append(data.y)
        y_preds += tensor_to_list(logits)
        y_trues += tensor_to_list(data.y)
      return y_preds, y_trues
