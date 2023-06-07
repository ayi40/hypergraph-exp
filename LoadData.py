import numpy as np
import collections
import scipy.sparse as sp

class LoadData(object):
    def __init__(self, datapath):
        kg_file = datapath+'/yelp2018-kg.txt'
        # kg_file = datapath + '/test.txt'

        self.n_relations, self.n_entities, self.n_hyperedge = 0, 0, 0
        self.kg_data, self.kg_dict, self.relation_dict = self._load_kg(kg_file)

        # generate the hypergraph adjacency
        self.a_adj, self.hyper_adj, self.edge_attr = self.build_hypergraph()

        # weight of hyperedge - default weight of all edge is 1
        self.hedge_wei = self._set_hyperedge_weight()

        self.hyper_in = self.set_norm()
        print(self.a_adj.toarray(), self.hyper_adj.toarray())
        print(self.hyper_in.toarray())




    def _load_kg(self, file_name):
        kg_np = np.loadtxt(file_name, dtype=np.int32)
        kg_np = np.unique(kg_np, axis=0)

        self.n_relations = max(kg_np[:, 1]) + 1
        self.n_entities = max(max(kg_np[:, 0]), max(kg_np[:, 2])) + 1
        self.n_hyperedge = len(kg_np)

        kg_dict = collections.defaultdict(list)
        relation_dict = collections.defaultdict(list)

        for head, relation, tail in kg_np:
            kg_dict[head].append((tail, relation))
            relation_dict[relation].append((head, tail))
        return kg_np, kg_dict, relation_dict

    def build_hypergraph(self):
        a_cols = np.hstack((np.arange(0, self.n_hyperedge, 1), np.arange(0, self.n_hyperedge, 1)))
        a_rows = np.hstack((self.kg_data[:, 0] , self.kg_data[:, 2]))
        a_vals = [1.] * (self.n_hyperedge * 2)


        a_adj = sp.coo_matrix((a_vals, (a_rows, a_cols)), shape=(self.n_entities, self.n_hyperedge))
        hyper_adj = sp.coo_matrix((a_vals, (a_cols, a_rows)), shape=(self.n_hyperedge, self.n_entities))
        edge_attr = self.kg_data[:, 1]
        # rowsum = np.array(a_adj.sum(1))
        # print(np.mean(rowsum))
        #
        # print(a_adj.shape, hyper_adj.shape, edge_attr.shape)
        return a_adj, hyper_adj, edge_attr

    def _set_hyperedge_weight(self):
        return sp.diags([1],[0],shape=(self.n_entities,self.n_entities))

    def set_norm(self):
        rowsum = np.array((self.hyper_adj.dot(self.hedge_wei)).sum(1))
        d_inv_sqrt = np.power(rowsum, -1).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
        D = sp.diags(d_inv_sqrt)

        colsum = np.array(self.hyper_adj.sum(0))
        b_inv_sqrt = np.power(colsum,-1).flatten()
        b_inv_sqrt[np.isinf(b_inv_sqrt)] = 0
        B = sp.diags(b_inv_sqrt)
        # res = D.dot(self.hyper_adj).dot(self.hedge_wei).dot(B).astype(np.float32).dot(self.hyper_adj.T.astype(np.float32))
        res = self._matrix_dot(D.dot(self.hyper_adj).dot(self.hedge_wei).dot(B), self.hyper_adj.T, 200)
        return res

    def _matrix_dot(self, A, B, n_fold):
        print(A.shape, B.shape)
        A_fold = []
        fold_len = self.n_hyperedge // n_fold

        for i in range(n_fold):
            start = i * fold_len
            if i ==n_fold-1:
                end = self.n_hyperedge
            else:
                end = (i+1) * fold_len
            A_fold.append(A[start:end].astype(np.float32))
        temp_res = []
        for i in range(n_fold):
            print(i)
            temp_res.append(A_fold[i].dot(B))
            print(temp_res[i].shape)
        res = np.concatenate(temp_res, axis=1)
        print(res.shape)

        return None











data = LoadData(datapath='./dataset')