import gc

import torch.nn as nn
import torch
import torch.nn.functional as F

## Graphsafe
##should we need dropout?
class Aggregator(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Aggregator, self).__init__()
        self.MLP = nn.Embedding(in_dim*3,out_dim)
        self.activation = nn.LeakyReLU()

    def forward(self, neighbor_in, neighbor_out, ego_embed):
        neighbor_in = torch.matmul(neighbor_in, ego_embed)
        neighbor_out = torch.matmul(neighbor_out, ego_embed)

        #graphsage
        embed = torch.cat([ego_embed, neighbor_in, neighbor_out], dim=1)
        embed = self.activation(self.MLP(embed))
        return embed




class EEHGCN(nn.Module):
    def __init__(self, config):
        super(EEHGCN, self).__init__()
        self._init_arg(config)
        self._init_embedding()
        self._cal_kg_egde()

        # initial model
        self.aggregator_layers = nn.ModuleList()
        for k in range(self.n_layers):
            self.aggregator_layers.append(Aggregator(self.weight_size_list[k], self.weight_size_list[k + 1]))

        self._cal_kg_egde()
    def _init_arg(self,config):
        self.n_relations = config['n_relations']
        self.n_entities = config['n_entities']
        self.n_edge = config['n_edge']
        self.in_norm = config['in_norm']
        self.outT_norm = config['outT_norm']
        self.hedge_wei = config['hedge_wei']
        self.in_norm_hat = config['in_norm_hat']
        self.out_norm_hat = config['out_norm_hat']

        # args
        self.embed_dim = 64
        self.dhgcn_layer = 2
        self.weight_size_list=[64,64,64]

    def _init_embedding(self):
        self.all_embedding = nn.Embedding(self.n_entities, self.embed_dim)
        nn.init.xavier_uniform_(self.all_embedding.weight)
        self.edge_embedding = nn.Embedding(self.n_edge, self.embed_dim)
        nn.init.xavier_uniform_(self.edge_embedding.weight)

    def _cal_kg_egde(self):
        ego_embed = self.edge_embedding.weight
        all_embedding = [ego_embed]
        neighbor_in = self._matrix_dot(self.in_norm_hat, self.outT_norm)
        neighbor_out = self._matrix_dot(self.out_norm_hat,self.in_norm.T)

        print(self.in_norm.shape,self.hedge_wei.shape,self.outT_norm.shape)

        for layer in self.aggregator_layers:
            ego_embed = layer(ego_embed, neighbor_in, neighbor_out)
            norm_embed = F.normalize(ego_embed, p=2, dim=1)
            all_embedding.append(norm_embed)

    def _matrix_dot(self, A, B):
        ego_embed = self.edge_embedding.weight.to(device)
        n_fold = len(A)
        res=[]
        for i in range(n_fold):
            r = torch.matmul(A[i], B)
            print(r.shape, i)
            res.append(r)
            del r
            gc.collect()
        res = torch.concatenate(res, axis=1)
        print(res.shape)








from LoadData import LoadData
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = LoadData(datapath='./dataset/amazon.txt')
config={}
config['n_relations'] = data.n_relations
config['n_entities'] = data.n_entities
config['n_edge'] = data.n_edge
config['in_norm'] = data.in_norm.to(device)
config['outT_norm'] = data.outT_norm.to(device)
config['hedge_wei'] = data.hedge_wei.to(device)
config['in_norm_hat'] = [i.to(device) for i in data.in_norm_hat]
config['out_norm_hat'] = [i.to(device) for i in data.out_norm_hat]
model = EEHGCN(config)





