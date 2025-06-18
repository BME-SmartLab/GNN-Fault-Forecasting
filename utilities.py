import torch
from torch_geometric.data import Data

def apply_mask(data, mask):
    x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
    x = (x * mask).float()
    mask = mask.squeeze().bool()
    edge_mask = (mask[edge_index[0]]) & (mask[edge_index[1]])
    edge_index = edge_index[:, edge_mask]
    if edge_attr is not None:
        edge_attr = edge_attr[edge_mask]
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=data.batch, y=data.y)

def tensor_to_list(tensor):
    result = []
    for i in range(len(tensor)):
        result.append(tensor[i].detach().cpu().numpy())
    return result

def tensor_batch_to_list(tensor, batch):
    result = []
    for i in range(batch.max() + 1):
        result.append(tensor[batch == i].detach().cpu().numpy())
    return result
