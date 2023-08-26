import torch
from model import MODEL
from LoadData import LoadData
from torch.utils.data import DataLoader
from Dataset import Dataset
from util.utils import utils
import torch.nn.functional as F
import os
import gc
from random import sample
from time import time
from util.metrics import *
from torch.utils.tensorboard import SummaryWriter


def cal_loss(pos_score, neg_score):
    return torch.mean((-1.0) * F.logsigmoid(pos_score - neg_score))

def eval(model, data, bs, device, Ks, mode):
    # get testing dataset
    test_batch_size = bs
    train_user_dict = data.train_user_dict
    test_user_dict = data.test_user_dict

    model.eval()
    user_ids = list(test_user_dict.keys())
    if mode=='mini-test':
        # user_ids = sample(user_ids,50)
        user_ids = user_ids[:30]
    item_ids = torch.arange(data.n_items, dtype=torch.long)
    item_ids+=data.n_users

    item_batches = [item_ids[i: i + test_batch_size] for i in range(0, len(item_ids), test_batch_size)]
    item_batches = [torch.LongTensor(d) for d in item_batches]

    cf_scores = []
    metric_names = ['precision', 'recall', 'ndcg']
    metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}
    for user in user_ids:
        t=time()
        for batch_idx, testdata in enumerate(item_batches):
            with torch.no_grad():
                i = testdata
                u = torch.tensor([user]*(i.shape[0]))

                path, path_idx = utils.find_uipath_batch(u, i, data.path_dic, data.pathlen, data.padding_idx,
                                                                 data.virtual_relation)
                u, i, path, path_idx = u.to(device), i.to(device), path.to(device), path_idx.to(device)
                batch_scores = model(u, path, path_idx, mode='predict').reshape(1,-1)

            batch_scores = batch_scores.cpu()
            batch_metrics = calc_metrics_at_k(batch_scores, train_user_dict, test_user_dict,
                                              np.array([user]),
                                              i.cpu().numpy(), Ks, data.n_users)

            cf_scores.append(batch_scores.numpy())
            for k in Ks:
                for m in metric_names:
                    metrics_dict[k][m].append(batch_metrics[k][m])

            del batch_scores, batch_metrics
            gc.collect()

    cf_scores = np.concatenate(cf_scores, axis=0)
    for k in Ks:
        for m in metric_names:
            metrics_dict[k][m] = np.concatenate(metrics_dict[k][m]).mean()
    print(metrics_dict)
    return cf_scores, metrics_dict



def train():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = LoadData(kg_file='./dataset/ckg_final.txt',simple_kg_file='./dataset/last-fm-simple.txt')

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
    config['bz'] = 3000
    lr=1e-4
    n_epoch = 20

    # initial model
    model = MODEL(config)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma = 0.8)

    # Load DataLoader
    train_dataset = Dataset(data.train_data, data.train_user_dict, data.n_items, 'train')
    train_loader = DataLoader(train_dataset, batch_size=config['bz'], shuffle=True, drop_last=True)

    n_batch = data.n_train // config['bz'] + 1
    writer = SummaryWriter('./result/whole_data')
    print(n_batch)
    _, metrics_dict = eval(model, data, 50000, device=device, Ks=[20, 40], mode='mini-test')
    writer.add_scalar('ndcg40', metrics_dict[40]['ndcg'], 0)
    writer.add_scalar('precision40', metrics_dict[40]['precision'], 0)
    for epoch in range(1, n_epoch + 1):

        start_time = time()
        total_loss=0
        model.train()
        for batch_ndx, train_data in enumerate(train_loader):
            u, pos_i, neg_i = train_data
            # print(u.shape, pos_i.shape, neg_i.shape)
            pos_path, pos_path_idx = utils.find_uipath_batch(u, pos_i, data.path_dic, data.pathlen, data.padding_idx, data.virtual_relation)
            neg_path, neg_path_idx = utils.find_uipath_batch(u, neg_i, data.path_dic, data.pathlen, data.padding_idx, data.virtual_relation)
            u, pos_i, neg_i, pos_path, neg_path, pos_path_idx, neg_path_idx = (u.to(device), pos_i.to(device), neg_i.to(device),
                                                                               pos_path.to(device), neg_path.to(device),
                                                                               pos_path_idx.to(device), neg_path_idx.to(device))

            pos_score, neg_score = model(u, pos_path, neg_path, pos_path_idx, neg_path_idx, mode='train')
            loss = cal_loss(pos_score, neg_score)
            writer.add_scalar('loss', loss, (epoch-1)*n_batch+batch_ndx)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            total_loss += loss

            if (batch_ndx+1) % 100 == 0:
                print('Training: Epoch {:04d} batch {:04d}|  Iter Mean Loss {:.4f}'.format(epoch, batch_ndx,
                                                                                                 loss))

        print(time()-start_time)
        print(
            'Training: Epoch {:04d} |  Iter Mean Loss {:.4f}'.format(epoch, total_loss / n_batch))
        cf_scores, metrics_dict = eval(model, data, 50000, device=device, Ks=[20, 40], mode='mini-test')
        writer.add_scalar('ndcg40', metrics_dict[40]['ndcg'], epoch)
        writer.add_scalar('precision40', metrics_dict[40]['precision'], epoch)
    writer.close()





train()












