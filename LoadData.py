import random

import numpy as np
import collections
import scipy.sparse as sp

from copy import deepcopy
import torch
import os


class LoadData(object):
    def __init__(self, kg_file, simple_kg_file):
        # config
        kg_file = kg_file
        simple_kg_file = simple_kg_file
        path = kg_file.rsplit('/', 1)[0]
        self.pathlen = 7



        # load ui-graph
        self.train_data, self.train_user_dict = self._load_uigraph(os.path.join(path, 'train.txt'))
        self.test_data, self.test_user_dict = self._load_uigraph(os.path.join(path, 'test.txt'))
        self.n_users, self.n_items, self.n_train, self.n_test = self.statistic()

        # load kg and simple kg
        self.n_relations, self.n_entities, self.n_edges = 0, 0, 0
        self.kg_data, self.kg_dict_head, self.kg_dict_tail, self.relation_dict = self._load_kg(kg_file,simple=False)
        self.virtual_relation = self.n_relations

        self.simple_kg, self.ns_entities, self.ns_edges = self._load_kg(simple_kg_file, simple=True)

        # load path
        self.padding_idx = self.n_entities
        self.path,self.path_dic = self._load_kgpath(os.path.join(path, 'path.txt'))


        print('relation num:{}, entities num:{}, edges num:{}'.format(self.n_relations, self.n_entities, self.n_edges))
        # print(self.ns_entities, self.ns_edges)

        # generate the hypergraph adjacency
        # hyper-head=hyper-out, hyper-tail=hyper-in
        self.hyper_in, self.hyper_out, self.edge_attr = self.build_simple_hypergraph()
        print(self.hyper_in.shape, self.hyper_out.shape, len(self.hyper_in.data))

        # weight of hyperedge - default weight of all edge is 1
        self.hedge_wei = self._set_hyperedge_weight()

        # without multi edge_weight
        self.H_in = self.set_norm(self.hyper_in, self.hyper_out)
        # self.H_out = self.set_norm(self.hyper_out, self.hyper_in)
        self.H_in_fold = self._cut_matrix(self.H_in, 100)
        self.hedge_wei = self.convert_coo2tensor(self.hedge_wei)

        # self.in_norm, self.outT_norm, self.hedge_wei= self.convert_coo2tensor(self.in_norm.tocoo()), \
        #                                                self.convert_coo2tensor(self.outT_norm.tocoo()),\
        #                                               self.convert_coo2tensor(self.hedge_wei.tocoo())

        # self.cal = self._matrix_dot(self.in_norm, self.outT_norm, 10000)
        # print(len(self.cal.data))
        # print(self.hyper_in.shape, self.hyper_in_T)


    def _load_kg(self, file_name, simple):
        if not simple:
            kg_np = np.loadtxt(file_name, dtype=np.int32)
            kg_np = np.unique(kg_np, axis=0)

            self.n_relations = max(kg_np[:, 1]) + 1
            self.n_entities = max(max(kg_np[:, 0]), max(kg_np[:, 2])) + 1
            self.n_edges = len(kg_np)

            kg_dict_head = collections.defaultdict(list)
            kg_dict_tail = collections.defaultdict(list)
            relation_dict = collections.defaultdict(list)

            for head, relation, tail in kg_np:
                kg_dict_head[head].append((head, relation, tail))
                kg_dict_tail[tail].append((head, relation, tail))
                relation_dict[relation].append((head, relation, tail))
            return kg_np, kg_dict_head, kg_dict_tail, relation_dict
        else:
            kg_np = np.loadtxt(file_name, dtype=np.int32)
            kg_np = np.unique(kg_np, axis=0)
            n_entities = max(max(kg_np[:, 0]), max(kg_np[:, 2])) + 1
            n_edges = len(kg_np)

            return kg_np, n_entities, n_edges

    def _load_uigraph(self, file_name):
        data = list()
        user_dict = dict()

        lines = open(file_name, 'r').readlines()
        for l in lines:
            l = l.strip()
            l = [int(i) for i in l.split(' ')]
            u_id, v_id = l[0], l[1:]
            v_id = list(set(v_id))

            for v in v_id:
                data.append([u_id, v])
            if len(v_id)>0:
                user_dict[u_id] = v_id
        return np.array(data), user_dict

    def _load_kgpath(self,file_name):
        all_path=list()
        path_dict = collections.defaultdict(lambda: collections.defaultdict(list))
        lines = open(file_name,'r').readlines()
        for l in lines:
            l = l.strip()
            l = [int(i) for i in l.split(' ')]
            h, t = l[0], l[-1]
            if len(l) < self.pathlen:
                l = l+[self.padding_idx]*(self.pathlen-len(l))
            all_path.append(l)
            path_dict[h][t].append(l)
        return all_path, path_dict

    def statistic(self):
        n_users = max(max(self.train_data[:, 0]), max(self.test_data[:, 0])) + 1
        n_items = max(max(self.train_data[:, 1]), max(self.test_data[:, 1])) + 1
        n_train = len(self.train_data)
        n_test = len(self.test_data)
        return n_users, n_items, n_train, n_test


    def build_simple_hypergraph(self):
        cols = np.arange(0, self.ns_edges, 1)
        rows_in = self.simple_kg[:, 0]
        rows_out = self.simple_kg[:, 2]
        vals = [1.] * self.ns_edges
        edge_attr = self.simple_kg[:, 1]
        hyper_in = sp.coo_matrix((vals, (cols, rows_in)), shape=(self.ns_edges, self.ns_entities))
        hyper_out = sp.coo_matrix((vals, (cols, rows_out)), shape=(self.ns_edges, self.ns_entities))
        return hyper_in, hyper_out, edge_attr


    def _set_hyperedge_weight(self):
        return sp.diags([1],[0],shape=(self.ns_entities,self.ns_entities))

    def set_norm(self, hyper_in, hyper_out):
        # for propagation dierection
        rowsum_in = np.array((hyper_in.dot(self.hedge_wei)).sum(1))
        d_in = np.power(rowsum_in, -1).flatten()
        d_in[np.isinf(d_in)] = 0
        D_in = sp.diags(d_in)

        colsum_out = np.array((hyper_out.sum(0)))
        d_out = np.power(colsum_out,-1).flatten()
        d_out[np.isinf(d_out)] = 0
        D_out = sp.diags(d_out)

        A_in =D_in.dot(hyper_in).dot(self.hedge_wei).dot(D_out).dot(hyper_out.T)
        return A_in

    def convert_coo2tensor(self, X):
        coo = X.tocoo()
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        return torch.sparse.FloatTensor(i, v, torch.Size(coo.shape))

    def _cut_matrix(self, matrix, n_fold):
        matrix_fold = []
        fold_len = matrix.shape[0] // n_fold

        for i in range(n_fold):
            start = i * fold_len
            if i == n_fold - 1:
                end = self.n_edges
            else:
                end = (i + 1) * fold_len
            matrix_fold.append(self.convert_coo2tensor(matrix[start:end]))
        return matrix_fold




