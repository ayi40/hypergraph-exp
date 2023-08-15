import torch
from model import MODEL
from LoadData import LoadData
from torch.utils.data import DataLoader
from Dataset import Dataset
from util.utils import utils
import torch.nn.functional as F
import os


def cal_loss(pos_score, neg_score):
    return torch.mean((-1.0) * F.logsigmoid(neg_score - pos_score))


def train():
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
    lr=1e-5
    n_epoch = 100

    # initial model
    model = MODEL(config)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=0.0001)

    # Load DataLoader
    train_dataset = Dataset(data.train_data, data.train_user_dict, data.n_items, 'train')
    test_dataset = Dataset(data.test_data, data.test_user_dict, data.n_items,  'test')
    train_loader = DataLoader(train_dataset, batch_size=config['bz'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['bz'])

    n_batch = data.n_train // config['bz'] + 1

    for epoch in range(1, n_epoch + 1):
        total_loss=0
        for batch_ndx, train_data in enumerate(train_loader):
            u, pos_i, neg_i = train_data
            pos_path, pos_path_idx = utils.find_uipath_batch(u, pos_i, data.path_dic, data.pathlen, data.padding_idx, data.virtual_relation)
            neg_path, neg_path_idx = utils.find_uipath_batch(u, neg_i, data.path_dic, data.pathlen, data.padding_idx, data.virtual_relation)
            u, pos_i, neg_i, pos_path, neg_path, pos_path_idx, neg_path_idx = (u.to(device), pos_i.to(device), neg_i.to(device),
                                                                               pos_path.to(device), neg_path.to(device),
                                                                               pos_path_idx.to(device), neg_path_idx.to(device))
            pos_score, neg_score = model(u, pos_i, neg_i, pos_path, neg_path, pos_path_idx, neg_path_idx)
            loss = cal_loss(pos_score, neg_score)
            print(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            total_loss += loss
            print('1')

        print(
            'Training: Epoch {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, total_loss / n_batch))


train()












