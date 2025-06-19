import os
import yaml
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from FaultForecastingDataset import FaultForecastingDataset
from GNN import GNN
from Explainer import Explainer
from tensorboardX import SummaryWriter
import seaborn as sns
import matplotlib.pyplot as plt

# torch.manual_seed(123)

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

train_baseline = config['train_baseline']
train_explainer = config['train_explainer']
target_sparsity = config['target_sparsity']

####################################################################################################

data_folder = 'rest-tree3' if tree else 'rest-tree4'
data_path = f'~/GNN-Fault-Forecasting/data/{data_folder}/data'
log_path = f'~/GNN-Fault-Forecasting/explainability_runs/{data_folder}/hidden_size_{hidden_size}_lr_{lr}_weight_decay_{weight_decay}_epochs_{epochs}_batch_size_{batch_size}_2'

data_path = os.path.expanduser(data_path)
log_path = os.path.expanduser(log_path)

if train_baseline or train_explainer:
    writer = SummaryWriter(log_path)

assert train_split + val_split + test_split == 1.0, "Train, validation, and test splits must sum to 1.0"

dataset = FaultForecastingDataset(data_path, tree=tree)

dataset = dataset.shuffle()

train_set, val_set, test_set = torch.utils.data.random_split(dataset,
                                                             [train_split, val_split, test_split])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=len(val_set))
test_loader = DataLoader(test_set, batch_size=len(test_set))

baseline = GNN(
    learning_rate=lr,
    weight_decay=weight_decay,
    in_channels=2,
    num_layers=4,
    hidden_channels=hidden_size,
    out_channels=1,
)

if train_baseline:
    best_val_mae = float('inf')
    for epoch in range(epochs):
        train_loss, train_mae, train_r2 = baseline.train_batch(train_loader)
        val_loss, val_mae, val_r2 = baseline.test_batch(val_loader)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(baseline.state_dict(), os.path.join(log_path, 'baseline.pt'))
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_mae', train_mae, epoch)
        writer.add_scalar('train_r2', train_r2, epoch)
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('val_mae', val_mae, epoch)
        writer.add_scalar('val_r2', val_r2, epoch)
        print(f'Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, Train R2: {train_r2:.4f}, Val loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, Val R2: {val_r2:.4f}')

baseline.load_state_dict(torch.load(os.path.join(log_path, 'baseline.pt'), weights_only=False))
val_loss, val_mae, val_r2 = baseline.test_batch(val_loader)
print(f'Val loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, Val R2: {val_r2:.4f}')
test_loss, test_mae, test_r2 = baseline.test_batch(test_loader)
print(f'Test loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}, Test R2: {test_r2:.4f}')

# Regression plot

y_preds, y_trues = baseline.predict_batch(test_loader)

plot_df = pd.DataFrame({'y_pred': np.concatenate(y_preds), 'y_true': np.concatenate(y_trues)})
figure = plt.figure(figsize=(4, 4))
ax = sns.regplot(x='y_pred', y='y_true', data=plot_df)
ax.set(xlabel='prediction', ylabel='ground truth')
plt.savefig(os.path.join(log_path, 'regression.png'))

##################################################################################################

pos_predictor = GNN(
    learning_rate=lr,
    weight_decay=weight_decay,
    in_channels=2,
    num_layers=4,
    hidden_channels=hidden_size,
    out_channels=1,
)

neg_predictor = GNN(
    learning_rate=lr,
    weight_decay=weight_decay,
    in_channels=2,
    num_layers=4,
    hidden_channels=hidden_size,
    out_channels=1,
)

explainer = Explainer(
    baseline=baseline,
    pos_predictor=pos_predictor,
    neg_predictor=neg_predictor,
    target_sparsity=target_sparsity,
)

if train_explainer:
    best_mae_diff = float('-inf')
    for epoch in range(epochs):
        train_loss, train_sparsity, train_pos_loss, train_neg_loss, train_pos_mae, train_neg_mae, train_pos_r2, train_neg_r2 = explainer.train_batch(train_loader)
        val_loss, val_sparsity, val_pos_loss, val_neg_loss, val_pos_mae, val_neg_mae, val_pos_r2, val_neg_r2 = explainer.test_batch(val_loader)
        current_mae_diff = val_neg_mae - val_pos_mae
        if current_mae_diff > best_mae_diff and val_sparsity < target_sparsity:
            best_val_fidelity_diff = current_mae_diff
            print(f'Best validation fidelity difference updated: {best_val_fidelity_diff:.4f}, saving explainer...')
            torch.save(explainer.state_dict(), os.path.join(log_path, 'explainer.pt'))
        print(f'Epoch: {epoch}, Train loss: {train_loss:.4f}, Train sparsity: {train_sparsity:.4f}, Train pos loss: {train_pos_loss:.4f}, Train neg loss: {train_neg_loss:.4f}, Train pos mae: {train_pos_mae:.4f}, Train neg mae: {train_neg_mae:.4f}, Train pos r2: {train_pos_r2:.4f}, Train neg r2: {train_neg_r2:.4f}')
        print(f'Epoch: {epoch}, Val loss: {val_loss:.4f}, Val sparsity: {val_sparsity:.4f}, Val pos loss: {val_pos_loss:.4f}, Val neg loss: {val_neg_loss:.4f}, Val pos mae: {val_pos_mae:.4f}, Val neg mae: {val_neg_mae:.4f}, Val pos r2: {val_pos_r2:.4f}, Val neg r2: {val_neg_r2:.4f}')
        writer.add_scalar('EXPLAINER/train_loss', train_loss, epoch)
        writer.add_scalar('EXPLAINER/train_sparsity', train_sparsity, epoch)
        writer.add_scalar('EXPLAINER/train_pos_loss', train_pos_loss, epoch)
        writer.add_scalar('EXPLAINER/train_neg_loss', train_neg_loss, epoch)
        writer.add_scalar('EXPLAINER/train_pos_mae', train_pos_mae, epoch)
        writer.add_scalar('EXPLAINER/train_neg_mae', train_neg_mae, epoch)
        writer.add_scalar('EXPLAINER/train_pos_r2', train_pos_r2, epoch)
        writer.add_scalar('EXPLAINER/train_neg_r2', train_neg_r2, epoch)
        writer.add_scalar('EXPLAINER/val_loss', val_loss, epoch)
        writer.add_scalar('EXPLAINER/val_sparsity', val_sparsity, epoch)
        writer.add_scalar('EXPLAINER/val_pos_loss', val_pos_loss, epoch)
        writer.add_scalar('EXPLAINER/val_neg_loss', val_neg_loss, epoch)
        writer.add_scalar('EXPLAINER/val_pos_mae', val_pos_mae, epoch)
        writer.add_scalar('EXPLAINER/val_neg_mae', val_neg_mae, epoch)
        writer.add_scalar('EXPLAINER/val_pos_r2', val_pos_r2, epoch)
        writer.add_scalar('EXPLAINER/val_neg_r2', val_neg_r2, epoch)

explainer.load_state_dict(torch.load(os.path.join(log_path, 'explainer.pt'), weights_only=False))
val_loss, val_sparsity, val_pos_loss, val_neg_loss, val_pos_mae, val_neg_mae, val_pos_r2, val_neg_r2 = explainer.test_batch(val_loader)
test_loss, test_sparsity, test_pos_loss, test_neg_loss, test_pos_mae, test_neg_mae, test_pos_r2, test_neg_r2 = explainer.test_batch(test_loader)
print(f'Val loss: {val_loss:.4f}, Val sparsity: {val_sparsity:.4f}, Val pos loss: {val_pos_loss:.4f}, Val neg loss: {val_neg_loss:.4f}, Val pos mae: {val_pos_mae:.4f}, Val neg mae: {val_neg_mae:.4f}, Val pos r2: {val_pos_r2:.4f}, Val neg r2: {val_neg_r2:.4f}')
print(f'Test loss: {test_loss:.4f}, Test sparsity: {test_sparsity:.4f}, Test pos loss: {test_pos_loss:.4f}, Test neg loss: {test_neg_loss:.4f}, Test pos mae: {test_pos_mae:.4f}, Test neg mae: {test_neg_mae:.4f}, Test pos r2: {test_pos_r2:.4f}, Test neg r2: {test_neg_r2:.4f}')

y_probs, y_masks, pos_preds, neg_preds, baseline_preds, y_trues = explainer.predict_batch(test_loader)

pos_plot_df = pd.DataFrame({'y_pred': np.concatenate(pos_preds), 'y_true': np.concatenate(y_trues)})
pos_figure = plt.figure(figsize=(4, 4))
pos_ax = sns.regplot(x='y_pred', y='y_true', data=pos_plot_df)
pos_ax.set(xlabel='prediction', ylabel='ground truth')
plt.savefig(os.path.join(log_path, 'pos_predictor_regression.png'))

neg_plot_df = pd.DataFrame({'y_pred': np.concatenate(neg_preds), 'y_true': np.concatenate(y_trues)})
neg_figure = plt.figure(figsize=(4, 4))
neg_ax = sns.regplot(x='y_pred', y='y_true', data=neg_plot_df)
neg_ax.set(xlabel='prediction', ylabel='ground truth')
plt.savefig(os.path.join(log_path, 'neg_predictor_regression.png'))

if train_baseline or train_explainer:
    writer.close()
