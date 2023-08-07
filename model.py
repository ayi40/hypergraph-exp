import torch.nn as nn
import torch
import torch.nn.functional as F
from util.utils import MatrixMethod
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Graphsafe
##should we need dropout?
class Aggregator(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Aggregator, self).__init__()
        self.MLP = nn.Embedding(in_dim*2, out_dim)
        self.activation = nn.LeakyReLU()
        self.MLPweight = self.MLP.weight

    def forward(self, ego_embed, neighbor_in):
        neighbor_in = MatrixMethod.matrix_dot(neighbor_in, ego_embed)
        # neighbor_out = MatrixMethod.matrix_dot(neighbor_out, ego_embed)

        #graphsage
        embed = torch.concatenate([ego_embed, neighbor_in], axis=1)
        embed = torch.chunk(embed, 100, dim=0)
        embed = MatrixMethod.matrix_dot(embed, self.MLPweight)
        embed = self.activation(embed)
        return embed





class MODEL(nn.Module):
    def __init__(self, config):
        super(MODEL, self).__init__()
        self._init_arg(config)
        self._init_embedding()

        # initial model
        # Agrregator
        self.aggregator_layers = nn.ModuleList()
        for k in range(self.dhgcn_layer):
            self.aggregator_layers.append(Aggregator(self.weight_size_list[k], self.weight_size_list[k + 1]))
        # MLP for fuse relation embedding of several layers
        self.MLPfuse = nn.Linear((self.dhgcn_layer+1)*self.embed_dim, self.embed_dim)
        self.activationfuse = nn.LeakyReLU()
        # LSTM
        self.LSTM = nn.LSTM(input_size=self.embed_dim, hidden_size=64, num_layers=2, batch_first=True)
        self.activationlstm = nn.Sigmoid()


    def _init_arg(self,config):
        self.n_relations = config['n_relations']
        self.n_entities = config['n_entities']
        self.n_edges = config['n_edges']
        self.n_users = config['n_users']
        self.n_items = config['n_items']
        self.ns_entities = config['ns_entities']
        self.ns_edges = config['ns_edges']
        self.padding_idx = config['padding_idx']
        self.H_in_fold = config['H_in_fold']
        self.hedge_wei = config['hedge_wei']
        self.edge_attr = config['edge_attr']

        # args
        self.embed_dim = 64
        self.dhgcn_layer = 2
        self.weight_size_list = [64, 64, 64]
        self.batchsize = config['bz']
        self.pathlen = 7

    def _init_embedding(self):
        # entities 中已经包括user，item， 和其他entities
        self.all_embedding = nn.Embedding(self.n_entities+1, self.embed_dim, padding_idx=self.padding_idx)
        nn.init.xavier_uniform_(self.all_embedding.weight)
        self.edge_embedding = nn.Embedding(self.ns_edges, self.embed_dim)
        nn.init.xavier_uniform_(self.edge_embedding.weight)
        # personal relation
        self.user_relation = nn.Embedding(self.n_relations*self.n_users, self.embed_dim)
        nn.init.xavier_uniform_(self.edge_embedding.weight)



    def _cal_kg_egde(self):
        ego_embed = self.edge_embedding.weight
        all_embedding = [ego_embed]

        for layer in self.aggregator_layers:
            ego_embed = layer(ego_embed, self.H_in_fold)
            norm_embed = F.normalize(ego_embed, p=2, dim=1)
            all_embedding.append(norm_embed)
        all_embedding = torch.concatenate(all_embedding, axis=1)
        kg_embed = []
        for i in range(self.n_relations):
            kg_embed.append(torch.sum(all_embedding[self.edge_attr == i], dim=0, keepdim=True))
        kg_embed = torch.concatenate(kg_embed, axis=0)
        kg_embed = self.activationfuse(self.MLPfuse(kg_embed))
        return kg_embed

    def _co_atten(self, kgr_embed, users):
        per_embed=[]
        kgr_embed = [kgr_embed for _ in range(self.batchsize)]
        kgr_embed = torch.stack(kgr_embed, axis=0)
        for u in users:
            per_embed.append(self.user_relation.weight[int(u * self.n_relations):int((u + 1) * self.n_relations)])
        per_embed = torch.stack(per_embed, axis=0)
        # print(per_embed.shape,kgr_embed.shape, kgr_embed)
        co_embed = kgr_embed*per_embed
        return co_embed

    def _cal_pro(self, co_embed, path, path_idx):
        mask = torch.arange(0, path.shape[1])
        relation_mask = (mask % 2) == 1
        relation_mask = relation_mask.to(device)
        ######user-item咋办
        relation_mask = (path != self.padding_idx) & relation_mask

        relation_embedding = path[relation_mask]
        relation_attr = path_idx.reshape(path.shape[0],1).repeat(1, path.shape[1])
        relation_attr = relation_attr[relation_mask]
        print(relation_attr,relation_embedding)
        relation_embedding = [co_embed[relation_attr[i], relation_embedding[i]] for i in range(len(relation_embedding))]
        print(relation_embedding.shape, relation_embedding)
        # relation_embedding = co_embed[relation_embedding]

        
        # k = torch.zeros(path.shape)
        # a=torch.ones(k[relation_mask].shape)
        # print(k,a)
        # k[relation_mask]=a
        # print(k)
        exit()





    def forward(self, users, pos_sample, neg_sample, pos_path, neg_path, pos_path_idx, neg_path_idx):
        kgr_embed = self._cal_kg_egde()
        co_embed = self._co_atten(kgr_embed, users)
        self._cal_pro(co_embed, pos_path, pos_path_idx)

















