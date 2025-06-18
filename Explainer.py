import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from GNN import GNBlock, MLPBlock
from utilities import apply_mask, tensor_to_list, tensor_batch_to_list

class Explainer(nn.Module):
    def __init__(self, baseline, pos_predictor, neg_predictor, target_sparsity):
        super(Explainer, self).__init__()
        self.target_sparsity = target_sparsity
        self.baseline = baseline
        self.pos_predictor = pos_predictor
        self.neg_predictor = neg_predictor
        self.learning_rate = baseline.learning_rate
        self.weight_decay = baseline.weight_decay
        self.in_channels = baseline.in_channels
        self.num_layers = baseline.num_layers
        self.hidden_channels = baseline.hidden_channels
        self.conv0 = GNBlock(
            self.in_channels,
            self.hidden_channels,
            self.hidden_channels,
        )
        for i in range(1, self.num_layers):
            setattr(self, f'conv{i}', GNBlock(
                self.hidden_channels,
                self.hidden_channels,
                self.hidden_channels,
            ))
        self.head = MLPBlock(self.hidden_channels, self.hidden_channels, 1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device != torch.device('cuda'):
            print('WARNING: GPU not available. Using CPU instead.')
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.criterion = self.loss
    
    def loss(self, reward, y_true, y_pred, batch):
        # the reward is calculated for each graph
        # binary cross-entropy between node selection probabilities and the node selection mask, averaged over the nodes for each graph
        cross_entropy = gnn.global_mean_pool(
            -torch.sum(y_true * torch.log(y_pred + 1e-8) + (1 - y_true) * torch.log(1 - y_pred + 1e-8), dim=1),
            batch
        )
        # L1 norm of the probability so that the model learns to use as few nodes as possible
        # instead of calculation for each graph, we calculate the mean over the whole batch
        L1_norm = torch.mean(y_pred) # gnn.global_mean_pool(y_pred, batch)[batch]
        assert 0 <= L1_norm <= 1
        # the selection budget is the difference between the sparsity target and the L1 norm
        selection_budget = torch.abs(self.target_sparsity - L1_norm)
        # the custom actor loss is the sum of the reward and the selection budget
        custom_actor_loss = (reward + 0.01 * selection_budget) * cross_entropy
        return torch.mean(custom_actor_loss)

    def forward(self, data):
        x = data.x
        for i in range(self.num_layers):
            x = getattr(self, f'conv{i}')(x, data.edge_index, data.edge_attr)
        x = self.head(x)
        return x

    def train_batch(self, loader):
        self.baseline.eval()
        self_losses, sparsities = [], []
        pos_losses, neg_losses, pos_maes, neg_maes = [], [], [], []
        for data in loader:
            data = data.to(self.device)
            with torch.no_grad():
                baseline_logits = self.baseline(data)
            self.eval()
            self.pos_predictor.train()
            self.pos_predictor.optimizer.zero_grad()
            self.neg_predictor.train()
            self.neg_predictor.optimizer.zero_grad()
            with torch.no_grad():
                probs = torch.sigmoid(self(data))
                mask = torch.bernoulli(probs)
            pos_logits = self.pos_predictor(apply_mask(data, mask))
            pos_loss = self.pos_predictor.criterion(pos_logits, baseline_logits, reduction='none')
            pos_loss.mean().backward()
            self.pos_predictor.optimizer.step()
            neg_logits = self.neg_predictor(apply_mask(data, 1.0 - mask))
            neg_loss = self.neg_predictor.criterion(neg_logits, baseline_logits, reduction='none')
            neg_loss.mean().backward()
            self.neg_predictor.optimizer.step()
            with torch.no_grad():
                reward = -(pos_loss - neg_loss)
                self_loss = self.criterion(reward, mask, probs, data.batch)
            self.pos_predictor.eval()
            self.neg_predictor.eval()
            self.train()
            self.optimizer.zero_grad()
            with torch.no_grad():
                probs = torch.sigmoid(self(data))
                mask = torch.bernoulli(probs)
                pos_logits = self.pos_predictor(apply_mask(data, mask))
                neg_logits = self.neg_predictor(apply_mask(data, 1.0 - mask))
                pos_loss = self.pos_predictor.criterion(pos_logits, baseline_logits, reduction='none')
                neg_loss = self.neg_predictor.criterion(neg_logits, baseline_logits, reduction='none')
                reward = -(pos_loss - neg_loss)
            probs = torch.sigmoid(self(data))
            self_loss = self.criterion(reward, mask, probs, data.batch)
            self_loss.backward()
            self.optimizer.step()
        self_losses.append(self_loss.item())
        sparsities.append(mask.mean().item())
        pos_losses.append(pos_loss.mean().item())
        neg_losses.append(neg_loss.mean().item())
        pos_mae = F.l1_loss(pos_logits, baseline_logits)
        neg_mae = F.l1_loss(neg_logits, baseline_logits)
        pos_maes.append(pos_mae.item())
        neg_maes.append(neg_mae.item())
        return sum(self_losses) / len(self_losses), \
            sum(sparsities) / len(sparsities), \
                sum(pos_losses) / len(pos_losses), \
                    sum(neg_losses) / len(neg_losses), \
                        sum(pos_maes) / len(pos_maes), \
                            sum(neg_maes) / len(neg_maes)
    
    @torch.no_grad()
    def test_batch(self, loader):
        self.pos_predictor.eval()
        self.neg_predictor.eval()
        self.baseline.eval()
        self.eval()
        self_losses, sparsities = [], []
        pos_losses, neg_losses, pos_maes, neg_maes = [], [], [], []
        for data in loader:
            data = data.to(self.device)
            baseline_logits = self.baseline(data)
            probs = torch.sigmoid(self(data))
            mask = (probs > 0.5).float()
            pos_logits = self.pos_predictor(apply_mask(data, mask))
            neg_logits = self.neg_predictor(apply_mask(data, 1.0 - mask))
            pos_loss = self.pos_predictor.criterion(pos_logits, baseline_logits, reduction='none')
            neg_loss = self.neg_predictor.criterion(neg_logits, baseline_logits, reduction='none')
            reward = -(pos_loss - neg_loss)
            self_loss = self.criterion(reward, mask, probs, data.batch)
            self_losses.append(self_loss.item())
            sparsities.append(mask.mean().item())
            pos_losses.append(pos_loss.mean().item())
            neg_losses.append(neg_loss.mean().item())
            pos_mae = F.l1_loss(pos_logits, baseline_logits)
            neg_mae = F.l1_loss(neg_logits, baseline_logits)
            pos_maes.append(pos_mae.item())
            neg_maes.append(neg_mae.item())
        return sum(self_losses) / len(self_losses), \
            sum(sparsities) / len(sparsities), \
                sum(pos_losses) / len(pos_losses), \
                    sum(neg_losses) / len(neg_losses), \
                        sum(pos_maes) / len(pos_maes), \
                            sum(neg_maes) / len(neg_maes)

    @torch.no_grad()
    def predict_batch(self, loader):
        self.pos_predictor.eval()
        self.neg_predictor.eval()
        self.baseline.eval()
        self.eval()
        y_probs, y_masks = [], []
        pos_preds, neg_preds, baseline_preds, y_trues = [], [], [], []
        for data in loader:
            data = data.to(self.device)
            probs = torch.sigmoid(self(data))
            mask = (probs > 0.5).float()
            y_probs += tensor_batch_to_list(probs, data.batch)
            y_masks += tensor_batch_to_list(mask, data.batch)
            pos_pred = self.pos_predictor(apply_mask(data, mask))
            neg_pred = self.neg_predictor(apply_mask(data, 1.0 - mask))
            baseline_pred = self.baseline(data)
            pos_preds += tensor_to_list(pos_pred)
            neg_preds += tensor_to_list(neg_pred)
            baseline_preds += tensor_to_list(baseline_pred)
            y_trues += tensor_to_list(data.y)
        return y_probs, y_masks, pos_preds, neg_preds, baseline_preds, y_trues
