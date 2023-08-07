import torch
from model import MODEL
from LoadData import LoadData
from torch.utils.data import DataLoader
from Dataset import Dataset
from util.utils import utils
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = LoadData(kg_file='./dataset/last-fm-ckg-oppo.txt',simple_kg_file='./dataset/last-fm-simple.txt')

print('Load Data Success')
config={}
config['n_relations'] = data.n_relations
config['n_entities'] = data.n_entities
config['n_edges'] = data.n_edges
config['n_users'] = data.n_users
config['n_items'] = data.n_items
config['ns_entities'] = data.ns_entities
config['ns_edges'] = data.ns_edges
config['padding_idx'] = data.padding_idx
config['H_in_fold'] = [i.to(device) for i in data.H_in_fold]
config['hedge_wei'] = data.hedge_wei.to(device)
config['edge_attr'] = data.edge_attr
config['path'] = data.path
config['path_dic'] = data.path_dic

# config
config['bz'] = 2

# initial model
model = MODEL(config)
model.to(device)

# Load DataLoader
train_dataset = Dataset(data.train_data, data.train_user_dict, data.n_items, 'train')
test_dataset = Dataset(data.test_data, data.test_user_dict, data.n_items,  'test')
train_loader = DataLoader(train_dataset, batch_size=config['bz'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config['bz'])


for batch_ndx, train_data in enumerate(train_loader):
    u, pos_i, neg_i = train_data
    pos_path, pos_path_idx = utils.find_uipath_batch(u, pos_i, data.path_dic, data.pathlen, data.padding_idx)
    neg_path, neg_path_idx = utils.find_uipath_batch(u, neg_i, data.path_dic, data.pathlen, data.padding_idx)
    u, pos_i, neg_i, pos_path, neg_path, pos_path_idx, neg_path_idx = (u.to(device), pos_i.to(device), neg_i.to(device),
                                                                       pos_path.to(device), neg_path.to(device),
                                                                       pos_path_idx.to(device), neg_path_idx.to(device))
    model(u, pos_i, neg_i, pos_path, neg_path, pos_path_idx, neg_path_idx)

















