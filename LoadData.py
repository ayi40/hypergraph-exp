import numpy as np
import collections
import scipy.sparse as sp
from copy import deepcopy
import torch


class LoadData(object):
    def __init__(self, datapath):
        # config
        kg_file = datapath

        self.n_relations, self.n_entities, self.n_edge = 0, 0, 0
        self.kg_data, self.kg_dict, self.relation_dict = self._load_kg(kg_file)
        print(self.n_relations, self.n_entities, self.n_edge)

        # generate the hypergraph adjacency
        # hyper-head=hyper-out, hyper-tail=hyper-in
        self.hyper_in, self.hyper_out, self.edge_attr = self.build_hypergraph()
        print(self.hyper_in.shape, self.hyper_out.shape, len(self.hyper_in.data))

        # weight of hyperedge - default weight of all edge is 1
        self.hedge_wei = self._set_hyperedge_weight()

        # without multi edge_weight
        self.in_norm, self.outT_norm = self.set_norm()
        self.in_norm_hat = self._cut_matrix(self.in_norm.dot(self.hedge_wei), 10000)
        self.out_norm_hat = self._cut_matrix(self.outT_norm.T.dot(self.hedge_wei), 10000)
        self.in_norm, self.outT_norm, self.hedge_wei= self.convert_coo2tensor(self.in_norm.tocoo()), \
                                                       self.convert_coo2tensor(self.outT_norm.tocoo()),\
                                                      self.convert_coo2tensor(self.hedge_wei.tocoo())

        # self.cal = self._matrix_dot(self.in_norm, self.outT_norm, 10000)
        # print(len(self.cal.data))
        # print(self.hyper_in.shape, self.hyper_in_T)


    def _load_kg(self, file_name):
        kg_np = np.loadtxt(file_name, dtype=np.int32)
        kg_np = np.unique(kg_np, axis=0)

        self.n_relations = max(kg_np[:, 1]) + 1
        self.n_entities = max(max(kg_np[:, 0]), max(kg_np[:, 2])) + 1
        self.n_edge = len(kg_np)

        kg_dict = collections.defaultdict(list)
        relation_dict = collections.defaultdict(list)

        for head, relation, tail in kg_np:
            kg_dict[head].append((tail, relation))
            relation_dict[relation].append((head, tail))
        return kg_np, kg_dict, relation_dict

    def build_hypergraph(self):
        cols = np.arange(0, self.n_edge, 1)
        rows_in = self.kg_data[:, 0]
        rows_out = self.kg_data[:, 2]
        vals = [1.] * self.n_edge
        edge_attr = self.kg_data[:, 1]
        hyper_in = sp.coo_matrix((vals, (cols, rows_in)), shape=(self.n_edge, self.n_entities))
        hyper_out = sp.coo_matrix((vals, (cols, rows_out)), shape=(self.n_edge, self.n_entities))
        return hyper_in, hyper_out, edge_attr


    def _set_hyperedge_weight(self):
        return sp.diags([1],[0],shape=(self.n_entities,self.n_entities))

    def set_norm(self):
        # for propagation dierection
        rowsum_in = np.array((self.hyper_in.dot(self.hedge_wei)).sum(1))
        d_in = np.power(rowsum_in, -1).flatten()
        d_in[np.isinf((d_in))] = 0
        D_in = sp.diags(d_in)

        colsum_out = np.array((self.hyper_out.sum(0)))
        d_out = np.power(colsum_out,-1).flatten()
        d_out[np.isinf(d_out)] = 0
        D_out = sp.diags(d_out)
        return D_in.dot(self.hyper_in), D_out.dot(self.hyper_out.T)

    def convert_coo2tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        values = coo.data
        indices = np.vstack((coo.row, coo.col))

        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        return torch.sparse.FloatTensor(i, v, torch.Size(coo.shape))


    def _matrix_dot(self, A, B, n_fold):
        A_fold = self._cut_matrix(A, n_fold)
        for i in range(n_fold):
            r = i.dot(B)
            print(len(A_fold[i].data))
            sp.save_npz('test.npz', r, compressed=True)
            print(i)
            del r
        # res = np.concatenate(res, axis=1)
        # print(res.shape)


    def _cut_matrix(self, matrix, n_fold):
        matrix_fold = []
        fold_len = matrix.shape[0] // n_fold

        for i in range(n_fold):
            start = i * fold_len
            if i == n_fold - 1:
                end = self.n_edge
            else:
                end = (i + 1) * fold_len
            matrix_fold.append(self.convert_coo2tensor(matrix[start:end]))
        return matrix_fold
















# data = LoadData(datapath='./dataset/amazon.txt')