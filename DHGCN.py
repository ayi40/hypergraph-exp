import torch.nn as nn
import torch
import torch.nn.functional as F
from util.utils import MatrixMethod

## Graphsafe
##should we need dropout?
class Aggregator(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Aggregator, self).__init__()
        self.MLP = nn.Embedding(in_dim*3, out_dim)
        self.activation = nn.LeakyReLU()
        self.MLPweight = self.MLP.weight

    def forward(self, ego_embed, neighbor_in, neighbor_out):
        neighbor_in = MatrixMethod.matrix_dot(neighbor_in, ego_embed)
        neighbor_out = MatrixMethod.matrix_dot(neighbor_out, ego_embed)

        #graphsage
        embed = torch.concatenate([ego_embed, neighbor_in, neighbor_out], axis=1)
        embed = torch.chunk(embed, 100, dim=0)
        embed = MatrixMethod.matrix_dot(embed, self.MLPweight)
        embed = self.activation(embed)
        return embed




class EEHGCN(nn.Module):
    def __init__(self, config):
        super(EEHGCN, self).__init__()
        self._init_arg(config)
        self._init_embedding()

        # initial model
        self.aggregator_layers = nn.ModuleList()
        for k in range(self.dhgcn_layer):
            self.aggregator_layers.append(Aggregator(self.weight_size_list[k], self.weight_size_list[k + 1]))

    def _init_arg(self,config):
        self.n_relations = config['n_relations']
        self.n_entities = config['n_entities']
        self.n_edges = config['n_edges']
        self.ns_entities = config['ns_entities']
        self.ns_edges = config['ns_edges']
        self.H_in_fold = config['H_in_fold']
        self.H_out_fold = config['H_out_fold']
        self.hedge_wei = config['hedge_wei']
        self.edge_attr = config['edge_attr']

        # args
        self.embed_dim = 64
        self.dhgcn_layer = 2
        self.weight_size_list = [64, 64, 64]

    def _init_embedding(self):
        self.all_embedding = nn.Embedding(self.n_entities, self.embed_dim)
        nn.init.xavier_uniform_(self.all_embedding.weight)
        self.edge_embedding = nn.Embedding(self.ns_edges, self.embed_dim)
        nn.init.xavier_uniform_(self.edge_embedding.weight)

    def _cal_kg_egde(self):
        ego_embed = self.edge_embedding.weight
        all_embedding = [ego_embed]

        for layer in self.aggregator_layers:
            ego_embed = layer(ego_embed, self.H_in_fold, self.H_out_fold)
            norm_embed = F.normalize(ego_embed, p=2, dim=1)
            all_embedding.append(norm_embed)
        all_embedding = torch.concatenate(all_embedding, axis=1)
        kg_embed = []
        for i in range(self.n_relations):
            kg_embed.append(torch.sum(all_embedding[self.edge_attr == i], dim=0, keepdim=True))
        kg_embed = torch.concatenate(kg_embed, axis=0)
        print(kg_embed.shape)


















from LoadData import LoadData
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = LoadData(datapath='./dataset/last-fm.txt',simple_kg_file='./dataset/last-fm-simple.txt')
print('Load Data Success')
config={}
config['n_relations'] = data.n_relations
config['n_entities'] = data.n_entities
config['n_edges'] = data.n_edges
config['ns_entities'] = data.ns_entities
config['ns_edges'] = data.ns_edges
config['H_in_fold'] = [i.to(device) for i in data.H_in_fold]
config['H_out_fold'] = [i.to(device) for i in data.H_out_fold]
config['hedge_wei'] = data.hedge_wei.to(device)
config['edge_attr'] = data.edge_attr

model = EEHGCN(config)
model.to(device)
model._cal_kg_egde()





