import os
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_undirected

class FaultForecastingDataset(InMemoryDataset):
    """Dataset for fault forecasting in service meshes.
    Each graph represents a service mesh with nodes as services and edges as dependencies.
    The features of the nodes are the probability of failure for leaf nodes and the number of retries for intermediary nodes.
    The target value is the system-level fault probability value.
    """
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, tree=True):
        self.tree = tree
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = []
        sizes = ['25_nodes'] if self.tree else ['25_nodes_downedges_noprio', '26_nodes_downedges_noprio']
        for size in sizes:
            # read cases.csv
            path = self.root
            exp_path = os.path.join(path, size)
            cases = pd.read_csv(f'{exp_path}/cases.csv')
            # cases correspond to the graphs / service meshes
            # we iterate through the cases and create a graph for each case
            for _, case_ in cases.iterrows():
                case_no = int(case_['case_no'])
                subcase_no = int(case_['subcase_no'])
                p_fault = [[float(case_['sla'])]] # system-level fault probability value
                case_path = f'{exp_path}/case_{case_no:05}/subcase_{subcase_no:05}'
                with open(f'{case_path}/case_data.json','r') as f:
                    # read graph.gexf.xml
                    g = nx.read_gexf(f'{case_path}/graph.gexf.xml')
                    # the probability of failure for each leaf node in the graph
                    p_fail = [-1] + list(nx.get_node_attributes(g, 'p_fail').values()) # -1 is the root node
                    # the number of retries for each intermediary node in the graph
                    retries = [-1] + list(nx.get_node_attributes(g, 'retries').values()) # -1 is the root node
                    features = []
                    for in_, out_, f1, f2 in zip(g.in_degree, g.out_degree, p_fail, retries):
                        if out_[1] == 0:
                            features.append([f1, 0]) # leaf nodes only have p_fail, no retries
                        else:
                            features.append([0, f2]) # intermediary nodes only have retries, no p_fail
                    # create a PyG Data object for the graph
                    data = Data()
                    # for visualization purposes, we keep the directed edge index
                    data.di_edge_index = torch.tensor(list((int(edge[0]), int(edge[1])) for edge in g.edges)).T
                    # but for training, we convert it to an undirected edge index
                    data.edge_index = to_undirected(data.di_edge_index)
                    data.case = case_no
                    data.subcase = subcase_no
                    # data.x = torch.ones((len(g), 1)) # if using dummy features instead of p_fail and retries
                    data.x = torch.tensor(features)
                    data.y = torch.tensor(p_fault)
                    data.batch = torch.full([len(g)], len(data_list) - 1)
                    data_list.append(data)
        self.save(data_list, self.processed_paths[0])
        torch.save(self.collate(data_list), self.processed_paths[0])
