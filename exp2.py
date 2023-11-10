import os
import pandas as pd
import networkx as nx
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_undirected
from Explainer import GNN, CriticGNN, ActorGNN
from tensorboardX import SummaryWriter
import seaborn as sns
import matplotlib.pyplot as plt

torch.manual_seed(123)

# hyperparameters
hidden = 32
lr = 0.0002
weight_decay = 0.002
epochs = 200
batch_size = 32
lambda_ = 0.001
accepted_mask_size = 1./3.

train_split, val_split, test_split = (0.8, 0.1, 0.1)

train_baseline = True
train_critic = True
train_actor = True

####################################################################################################

data_path = '~/GNN-SLA-Forecast/data/rest-tree3/data'
log_path = '~/GNN-SLA-Forecast/runs/rest-tree3/lambda_{}_hidden_{}_lr_{}_weight_decay_{}_epochs_{}_batch_size_{}'.format(lambda_, hidden, lr, weight_decay, epochs, batch_size)

data_path = os.path.expanduser(data_path)
log_path = os.path.expanduser(log_path)

train = train_baseline or train_critic or train_actor

if train:
    writer = SummaryWriter(log_path)

assert train_split + val_split + test_split == 1.0

class SLADataset(InMemoryDataset):
        def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
            super().__init__(root, transform, pre_transform, pre_filter)
            self.load(self.processed_paths[0])
            self.data, self.slices = torch.load(self.processed_paths[0])

        @property
        def processed_file_names(self):
            return ['data.pt']

        def process(self):
            data_list = []
            for size in ['25_nodes']:
                # read cases.csv
                path = self.root
                exp_path = os.path.join(path, size)
                cases = pd.read_csv(f'{exp_path}/cases.csv')
                # iterate through the cases: each case is a graph, with a corresponding SLA value
                for _, case_ in cases.iterrows():
                    case_no = int(case_['case_no'])
                    subcase_no = int(case_['subcase_no'])
                    sla = [[float(case_['sla'])]] # SLA value -> target
                    case_path = f'{exp_path}/case_{case_no:05}/subcase_{subcase_no:05}'
                    with open(f'{case_path}/case_data.json','r') as f:
                        # read graph.gexf.xml
                        g = nx.read_gexf(f'{case_path}/graph.gexf.xml')
                        p_fail = [-1] + list(nx.get_node_attributes(g, 'p_fail').values()) # -1 is the root
                        retries = [-1] + list(nx.get_node_attributes(g, 'retries').values()) # -1 is the root
                        features = []
                        for in_, out_, f1, f2 in zip(g.in_degree, g.out_degree, p_fail, retries):
                            if out_[1] == 0:
                                features.append([f1, 0]) # leaf node
                            else:
                                features.append([0, f2]) # non-leaf node
                        data = Data()
                        data.di_edge_index = torch.tensor(list((int(edge[0]), int(edge[1])) for edge in g.edges)).T
                        data.edge_index = to_undirected(data.di_edge_index)
                        data.case = case_no
                        data.subcase = subcase_no
                        # data.x = torch.ones((len(g), 1)) # dummy features instead
                        data.x = torch.tensor(features)
                        data.y = torch.tensor(sla)
                        data.batch = torch.full([len(g)], len(data_list) - 1)
                        data_list.append(data)
            self.save(data_list, self.processed_paths[0])
            torch.save(self.collate(data_list), self.processed_paths[0])

dataset = SLADataset(data_path)

dataset = dataset.shuffle()

train_set, val_set, test_set = torch.utils.data.random_split(dataset,
                                                             [train_split, val_split, test_split])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=len(val_set))
test_loader = DataLoader(test_set, batch_size=len(test_set))

baseline = GNN(2, hidden, 1)
baseline.optimizer = torch.optim.AdamW(baseline.parameters(), lr=lr, weight_decay=weight_decay)
print(baseline)

critic = CriticGNN(2, hidden, 1)
critic.optimizer = torch.optim.AdamW(critic.parameters(), lr=lr, weight_decay=weight_decay)
print(critic)

actor = ActorGNN(critic, baseline, lambda_)
actor.optimizer = torch.optim.AdamW(actor.parameters(), lr=lr, weight_decay=weight_decay)
print(actor)

if train_baseline:
    best_val_mae = float('inf')
    for epoch in range(epochs):
        train_loss, train_mae = baseline.train_batch(train_loader)
        val_loss, val_mae = baseline.test_batch(val_loader)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(baseline.state_dict(), os.path.join(log_path, 'baseline.pt'))
        writer.add_scalar('BASELINE/train_loss', train_loss, epoch)
        writer.add_scalar('BASELINE/train_mae', train_mae, epoch)
        writer.add_scalar('BASELINE/val_loss', val_loss, epoch)
        writer.add_scalar('BASELINE/val_mae', val_mae, epoch)
        print(f'Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, Val loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}')

baseline.load_state_dict(torch.load(os.path.join(log_path, 'baseline.pt')))
val_loss, val_mae = baseline.test_batch(val_loader)
print(f'Val loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}')
test_loss, test_mae = baseline.test_batch(test_loader)
print(f'Test loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}')

# Regression plot

y_preds, y_trues = baseline.predict_batch(test_loader)

plot_df = pd.DataFrame({'y_pred': y_preds[:, 0], 'y_true': y_trues[:, 0]})
figure = plt.figure(figsize=(4, 4))
ax = sns.regplot(x='y_pred', y='y_true', data=plot_df)
ax.set(xlabel='prediction', ylabel='ground truth')
plt.savefig(os.path.join(log_path, 'regression.png'))


if train_critic:
    best_val_mae = float('inf')
    for epoch in range(epochs):
        train_loss, train_mae = critic.train_batch(train_loader)
        val_loss, val_mae = critic.test_batch(val_loader)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(critic.state_dict(), os.path.join(log_path, 'critic.pt'))
        writer.add_scalar('CRITIC/train_loss', train_loss, epoch)
        writer.add_scalar('CRITIC/train_mae', train_mae, epoch)
        writer.add_scalar('CRITIC/val_loss', val_loss, epoch)
        writer.add_scalar('CRITIC/val_mae', val_mae, epoch)
        print(f'Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, Val loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}')

critic.load_state_dict(torch.load(os.path.join(log_path, 'critic.pt')))
val_loss, val_mae = critic.test_batch(val_loader)
print(f'Val loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}')
test_loss, test_mae = critic.test_batch(test_loader)
print(f'Test loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}')

# Regression plot

y_preds, y_trues = critic.predict_batch(test_loader)

plot_df = pd.DataFrame({'y_pred': y_preds[:, 0], 'y_true': y_trues[:, 0]})
figure = plt.figure(figsize=(4, 4))
ax = sns.regplot(x='y_pred', y='y_true', data=plot_df)
ax.set(xlabel='prediction', ylabel='ground truth')
plt.savefig(os.path.join(log_path, 'regression.png'))


if train_actor:
    best_val_mae = float('inf')
    for epoch in range(epochs):
        train_loss, train_mask_size, train_critic_pos_mae, train_critic_neg_mae, train_baseline_mae = actor.train_batch(train_loader)
        val_loss, val_mask_size, val_critic_pos_mae, val_critic_neg_mae, val_baseline_mae = actor.test_batch(val_loader)
        if val_critic_pos_mae < best_val_mae and val_mask_size < accepted_mask_size:
            best_val_mae = val_critic_pos_mae
            torch.save(actor.state_dict(), os.path.join(log_path, 'actor.pt'))
        writer.add_scalar('ACTOR/train_loss', train_loss, epoch)
        writer.add_scalar('ACTOR/train_mask_size', train_mask_size, epoch)
        writer.add_scalar('ACTOR/train_critic_pos_mae', train_critic_pos_mae, epoch)
        writer.add_scalar('ACTOR/train_critic_neg_mae', train_critic_neg_mae, epoch)
        writer.add_scalar('ACTOR/train_baseline_mae', train_baseline_mae, epoch)
        writer.add_scalar('ACTOR/val_loss', val_loss, epoch)
        writer.add_scalar('ACTOR/val_mask_size', val_mask_size, epoch)
        writer.add_scalar('ACTOR/val_critic_pos_mae', val_critic_pos_mae, epoch)
        writer.add_scalar('ACTOR/val_critic_neg_mae', val_critic_neg_mae, epoch)
        writer.add_scalar('ACTOR/val_baseline_mae', val_baseline_mae, epoch)
        print(f'Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Train mask size: {train_mask_size:.4f},\
              Train critic pos MAE: {train_critic_pos_mae:.4f}, Train critic neg MAE: {train_critic_neg_mae:.4f}, Train baseline MAE: {train_baseline_mae:.4f},\
                Val loss: {val_loss:.4f}, Val mask size: {val_mask_size:.4f},\
                    Val critic pos MAE: {val_critic_pos_mae:.4f}, Val critic neg MAE: {val_critic_neg_mae:.4f}, Val baseline MAE: {val_baseline_mae:.4f}')

actor.load_state_dict(torch.load(os.path.join(log_path, 'actor.pt')))
val_loss, val_mask_size, val_critic_pos_mae, val_critic_neg_mae, val_baseline_mae = actor.test_batch(val_loader)
print(f'Val loss: {val_loss:.4f}, Val mask size: {val_mask_size:.4f}, Val critic pos MAE: {val_critic_pos_mae:.4f}, Val critic neg MAE: {val_critic_neg_mae:.4f}, Val baseline MAE: {val_baseline_mae:.4f}')
test_loss, test_mask_size, test_critic_pos_mae, test_critic_neg_mae, test_baseline_mae = actor.test_batch(test_loader)
print(f'Test loss: {test_loss:.4f}, Test mask size: {test_mask_size:.4f}, Test critic pos MAE: {test_critic_pos_mae:.4f}, Test critic neg MAE: {test_critic_neg_mae:.4f}, Test baseline MAE: {test_baseline_mae:.4f}')

if train:
    writer.close()
