import os
import yaml
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from FaultForecastingDataset import FaultForecastingDataset
from GNN import GNN
from tensorboardX import SummaryWriter
import seaborn as sns
import matplotlib.pyplot as plt

torch.manual_seed(123)

# Load hyperparameters from a YAML file
config_path = os.path.expanduser('~/GNN-Fault-Forecasting/config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

hidden_size = config['hidden_size']
lr = config['lr']
weight_decay = config['weight_decay']
epochs = config['epochs']
batch_size = config['batch_size']
tree = config['tree']

train_split, val_split, test_split = config['splits']

train = config['train']

####################################################################################################

data_folder = 'rest-tree3' if tree else 'rest-tree4'
data_path = f'~/GNN-Fault-Forecasting/data/{data_folder}/data'
log_path = f'~/GNN-Fault-Forecasting/runs/{data_folder}/hidden_size_{hidden_size}_lr_{lr}_weight_decay_{weight_decay}_epochs_{epochs}_batch_size_{batch_size}'

data_path = os.path.expanduser(data_path)
log_path = os.path.expanduser(log_path)

if train:
    writer = SummaryWriter(log_path)

assert train_split + val_split + test_split == 1.0, "Train, validation, and test splits must sum to 1.0"

dataset = FaultForecastingDataset(data_path, tree=tree)

dataset = dataset.shuffle()

train_set, val_set, test_set = torch.utils.data.random_split(dataset,
                                                             [train_split, val_split, test_split])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=len(val_set))
test_loader = DataLoader(test_set, batch_size=len(test_set))

model = GNN(
    learning_rate=lr,
    weight_decay=weight_decay,
    in_channels=2,
    num_layers=4,
    hidden_channels=hidden_size,
    out_channels=1,
)

if train:
    best_val_mae = float('inf')
    for epoch in range(epochs):
        train_loss, train_mae = model.train_batch(train_loader)
        val_loss, val_mae = model.test_batch(val_loader)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), os.path.join(log_path, 'gnn.pt'))
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_mae', train_mae, epoch)
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('val_mae', val_mae, epoch)
        print(f'Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, Val loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}')

    writer.close()

model.load_state_dict(torch.load(os.path.join(log_path, 'gnn.pt')))
val_loss, val_mae = model.test_batch(val_loader)
print(f'Val loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}')
test_loss, test_mae = model.test_batch(test_loader)
print(f'Test loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}')

# Regression plot

y_preds, y_trues = model.predict_batch(test_loader)

plot_df = pd.DataFrame({'y_pred': y_preds[:, 0], 'y_true': y_trues[:, 0]})
figure = plt.figure(figsize=(4, 4))
ax = sns.regplot(x='y_pred', y='y_true', data=plot_df)
ax.set(xlabel='prediction', ylabel='ground truth')
plt.savefig(os.path.join(log_path, 'regression.png'))
