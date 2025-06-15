import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from FaultForecastingDataset import FaultForecastingDataset
from sklearn.linear_model import LinearRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

torch.manual_seed(0)

# Load hyperparameters from a YAML file
config_path = os.path.expanduser('~/GNN-Fault-Forecasting/config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

tree = config['tree']

train_split, val_split, test_split = config['splits']

####################################################################################################

data_folder = 'rest-tree3' if tree else 'rest-tree4'
data_path = f'~/GNN-Fault-Forecasting/data/{data_folder}/data'
data_path = os.path.expanduser(data_path)

assert train_split + val_split + test_split == 1.0, "Train, validation, and test splits must sum to 1.0"

dataset = FaultForecastingDataset(data_path, tree=tree)

dataset = dataset.shuffle()

train_set, val_set, test_set = torch.utils.data.random_split(dataset,
                                                             [train_split, val_split, test_split])


### baseline: linear regression or neural network regression on node feature and spectral feature statistics

def bucketize_feature(values, k):
    """
    Divides sorted values into k buckets and computes the mean of each bucket.
    """
    values_sorted, _ = torch.sort(values)
    n = len(values)
    stats = []
    for i in range(k):
        start = int(i * n / k)
        end = int((i + 1) * n / k)
        if start < end:
            bucket_values = values_sorted[start:end]
            stats.append(bucket_values.mean())
        else:
            stats.append(torch.tensor(0.0))
    return torch.stack(stats)

def compute_spectral_features(data, n_components, num_buckets):
    """
    Computes spectral embedding of the graph and summarizes each component via bucketized means.
    """
    edge_index = data.edge_index.cpu().numpy()
    num_nodes = data.num_nodes
    A = np.zeros((num_nodes, num_nodes))
    for i, j in edge_index.T:
        A[i, j] = 1
        A[j, i] = 1
    embedder = SpectralEmbedding(n_components=n_components, affinity='precomputed')
    embedding = embedder.fit_transform(A)
    embedding = torch.from_numpy(embedding).float()

    bucket_list = []
    for i in range(embedding.shape[1]):
        coord = embedding[:, i]
        buckets = bucketize_feature(coord, k=num_buckets)
        bucket_list.append(buckets)
    spec_feat = torch.cat(bucket_list)

    # spec_feat = torch.cat([
    #     spec_feat,
    #     embedding.mean(axis=0),
    #     embedding.std(axis=0),
    #     embedding.max(axis=0)[0],
    #     embedding.min(axis=0)[0],
    # ])

    return spec_feat

def build_feature_matrix(dataset, num_buckets=6, use_spec=True, num_spec_components=2):
    """
    Builds graph-level feature matrix by bucketizing node features (by node type) and optionally spectral features.
    """
    X_list = []
    for data in dataset:
        intermediary_mask = data.node_type == 0
        intermediary_feat1 = bucketize_feature(data.x[intermediary_mask, 0], k=num_buckets)
        intermediary_feat2 = bucketize_feature(data.x[intermediary_mask, 1], k=num_buckets)
        leaf_mask = data.node_type == 1
        leaf_feat1 = bucketize_feature(data.x[leaf_mask, 0], k=num_buckets)
        leaf_feat2 = bucketize_feature(data.x[leaf_mask, 1],  k=num_buckets)
        feat = torch.cat([
            intermediary_feat1,
            intermediary_feat2,
            leaf_feat1,
            leaf_feat2,
        ])

        # feat = torch.cat([
        #     data.x.mean(dim=0),
        #     data.x.std(dim=0),
        #     data.x.max(dim=0)[0],
        #     data.x.min(dim=0)[0],
        # ])

        if use_spec:
            spec_feat = compute_spectral_features(data, data.num_nodes-1, num_buckets=num_buckets)
            feat = torch.cat([feat, spec_feat], dim=0)
        X_list.append(feat)
    return torch.stack(X_list, dim=0).numpy()

# model definition for a simple neural network

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def train_and_evaluate(train_set, val_set, test_set, config, mode="linreg", num_buckets=6, use_spec=True):
    """
    Train and evaluate a model on the provided datasets using either linear regression or a simple neural network.
    Args:
        train_set: Training dataset.
        val_set: Validation dataset.
        test_set: Test dataset.
        config: Configuration dictionary containing hyperparameters.
        mode: 'linreg' for linear regression, 'nn' for neural network.
        num_buckets: Number of buckets for feature binning.
        use_spec: Whether to use spectral features.
    """
    # Build feature matrices
    X_train = build_feature_matrix(train_set, num_buckets=num_buckets, use_spec=use_spec)
    X_val = build_feature_matrix(val_set, num_buckets=num_buckets, use_spec=use_spec)
    X_test = build_feature_matrix(test_set, num_buckets=num_buckets, use_spec=use_spec)
    y_train = torch.cat([data.y for data in train_set], dim=0).numpy()
    y_val = torch.cat([data.y for data in val_set], dim=0).numpy()
    y_test = torch.cat([data.y for data in test_set], dim=0).numpy()

    if mode == "linreg":
        model = LinearRegression()
        model.fit(X_train, y_train)
        def report(X, y, name):
            y_pred = model.predict(X)
            print(f'Regression with bucketized node and spectral features: {name} \
                  MSE: {mean_squared_error(y, y_pred):.4f}, MAE: {mean_absolute_error(y, y_pred):.4f}, R2: {r2_score(y, y_pred):.4f}')
        report(X_train, y_train, "Train")
        report(X_val, y_val, "Validation")
        report(X_test, y_test, "Test")

    elif mode == "nn":
        train_loader = torch.utils.data.DataLoader(
            list(zip(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())),
            batch_size=config['batch_size'], shuffle=True)
        val_loader = torch.utils.data.DataLoader(
            list(zip(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())),
            batch_size=len(val_set), shuffle=False)
        test_loader = torch.utils.data.DataLoader(
            list(zip(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())),
            batch_size=len(test_set), shuffle=False)
        model = SimpleNN(X_train.shape[1], hidden_dim=config['hidden_size'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        criterion = nn.MSELoss()
        best_val_mae = float('inf')
        # Training loop
        for epoch in range(config['epochs']):
            # Training
            model.train()
            train_losses, train_maes, train_r2s = [], [], []
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                train_mae = mean_absolute_error(y_batch.numpy(), y_pred.detach().numpy())
                train_maes.append(train_mae)
                train_r2 = r2_score(y_batch.numpy(), y_pred.detach().numpy())
                train_r2s.append(train_r2)
            # Validation
            model.eval()
            with torch.no_grad():
                X_val_batch, y_val = next(iter(val_loader))
                y_val_pred = model(X_val_batch)
                val_loss = criterion(y_val_pred, y_val)
                val_mae = mean_absolute_error(y_val.numpy(), y_val_pred.numpy())
                val_r2 = r2_score(y_val.numpy(), y_val_pred.numpy())
            # Save best model
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                torch.save(model.state_dict(), 'best_simple_nn.pth')
            # Logging
            print(f"Epoch {epoch} | Train Loss: {np.mean(train_losses):.4f}, Train MAE: {np.mean(train_maes):.4f}, Train R2: {np.mean(train_r2s):.4f} | "
                    f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, Val R2: {val_r2:.4f}")
        model.load_state_dict(torch.load('best_simple_nn.pth'))
        model.eval()
        with torch.no_grad():
            # Validation
            X_val_batch, y_val = next(iter(val_loader))
            y_val_pred = model(X_val_batch)
            val_loss = criterion(y_val_pred, y_val)
            val_mae = mean_absolute_error(y_val.numpy(), y_val_pred.numpy())
            val_r2 = r2_score(y_val.numpy(), y_val_pred.numpy())
            print(f'Neural Network with bucketized node and spectral features: Validation Loss: {val_loss:.4f}, Validation MAE: {val_mae:.4f}, Validation R2: {val_r2:.4f}')
            # Test
            X_test_batch, y_test = next(iter(test_loader))
            y_test_pred = model(X_test_batch)
            test_loss = criterion(y_test_pred, y_test)
            test_mae = mean_absolute_error(y_test.numpy(), y_test_pred.numpy())
            test_r2 = r2_score(y_test.numpy(), y_test_pred.numpy())
            # print(f'Neural Network with bucketized node and spectral features: Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}, Test R2: {test_r2:.4f}')

for num_buckets in [3, 5, 6]:
    print(f"Training with {num_buckets} buckets:")
    # Linear Regression
    train_and_evaluate(train_set, val_set, test_set, config, mode="linreg", num_buckets=num_buckets, use_spec=True)
    # Neural Network
    train_and_evaluate(train_set, val_set, test_set, config, mode="nn", num_buckets=num_buckets, use_spec=True)
