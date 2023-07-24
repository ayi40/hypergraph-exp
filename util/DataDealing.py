import numpy as np
import scipy.sparse as sp
import random
import os
import collections

class LoadUIGraph:
    def __init__(self, path):
        self.train_data, self.train_user_dict = self._load_uigraph(os.path.join(path, 'train.txt'))
        self.test_data, self.test_user_dict = self._load_uigraph(os.path.join(path, 'test.txt'))
        self.statistic()
        print('user num:{}, item num:{}, train data num:{}, test data num:{}'.format(self.n_users,self.n_items,self.n_train,self.n_test))

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

    def statistic(self):
        self.n_users = max(max(self.train_data[:, 0]), max(self.test_data[:, 0])) + 1
        self.n_items = max(max(self.train_data[:, 1]), max(self.test_data[:, 1])) + 1
        self.n_train = len(self.train_data)
        self.n_test = len(self.test_data)


class LoadKG:
    def __init__(self, path):
        self.load_kg(path)
        self.statistic()

    def load_kg(self, path):
        kg = np.loadtxt(path, dtype=np.int32)
        kg = np.unique(kg, axis=0)
        self.data = kg

    def statistic(self):
        self.n_entities = max(max(self.data[:, 0]), max(self.data[:, 2])) + 1
        self.n_relations = max(self.data[:, 1]) + 1
        self.n_edge = len(self.data)
        print(self.n_entities, self.n_relations, self.n_edge)

        self.kg_dict_head = collections.defaultdict(list)
        self.kg_dict_tail = collections.defaultdict(list)
        self.relation_dict = collections.defaultdict(list)

        for head, relation, tail in self.data:
            self.kg_dict_head[head].append((head, relation, tail))
            self.kg_dict_tail[tail].append((head, relation, tail))
            self.relation_dict[relation].append((head, relation, tail))

    @staticmethod
    def save_kg(path, kgdata):
        np.savetxt(path, kgdata, fmt='%d')

class DealingData:
    def __init__(self):
        pass

    # check for duplicate relations
    @staticmethod
    def check_reverse_equal(kg):
        relation_matrix = []
        reverse = []
        for i in range(kg.n_relations):
            data = kg.data[kg.data[:, 1] == i]
            print(i,len(data))
            row = data[:, 0]
            col = data[:, 2]
            val = [1]*len(data)
            matrix = sp.coo_matrix((val, (row, col)), shape=(kg.n_entities, kg.n_entities))
            relation_matrix.append(matrix)
        for i in range(len(relation_matrix)):
            for j in range(i,len(relation_matrix)):
                # .nnz() number of non zero
                if (relation_matrix[i] != relation_matrix[j].T).nnz == 0:
                    reverse.append([i,j])
        print((relation_matrix[0] != relation_matrix[1].T).nnz)
        print((relation_matrix[0] != relation_matrix[3]).nnz)
        print(reverse)

    #only deal with relation, without change the entity id
    @staticmethod
    def del_relation(kg, del_list:list):
        new_index = 0
        res_kg = []
        for i in range(kg.n_relations):
            if i in del_list:
                continue
            data = kg.data[kg.data[:, 1] == i]
            data[:, 1] = new_index
            new_index += 1
            res_kg.append(data)
        res_kg = np.concatenate(res_kg, axis=0)
        kg.data = res_kg
        print('Delete success!')
        kg.statistic()

    @staticmethod
    def simple_kg(kg, seed_num):
        kg_seed = []
        for i in kg.relation_dict:
            if len(kg.relation_dict[i]) < seed_num:
                kg_seed += kg.relation_dict[i]
            else:
                kg_seed += random.sample(kg.relation_dict[i], seed_num)
        kg_simple = set(kg_seed)
        for (h, r, t) in kg_seed:
            for i in kg.kg_dict_head[h]:
                if i[1] != r:
                    kg_simple.add(i)
            for i in kg.kg_dict_tail[t]:
                if i[1] != r:
                    kg_simple.add(i)
            for i in kg.kg_dict_head[t]:
                kg_simple.add(i)
            for i in kg.kg_dict_tail[h]:
                kg_simple.add(i)
        kg_simple = list(kg_simple)

        # re-order
        simple_entities = set()
        for i in kg_simple:
            simple_entities.add(i[0])
            simple_entities.add(i[2])
        simple_entities = list(simple_entities)
        simple_entities.sort()
        print(len(simple_entities))
        transfer_dict = {}
        for index, e in enumerate(simple_entities):
            transfer_dict[e]=index
        kg_simple.sort()
        for i in range(len(kg_simple)):
            h, r, t = kg_simple[i]
            kg_simple[i] = [transfer_dict[h], r, transfer_dict[t]]
        kg_simple = np.array(kg_simple)
        LoadKG.save_kg('../dataset/last-fmtest.txt',kg_simple)
        return kg_simple

    @staticmethod
    def get_user2item_path(kg, ui, pathlen):
        path_dic = collections.defaultdict(list)
        for i in range(1):
            unfind = [[i]]
            find=[]
            for l in range(pathlen):
                path = unfind.pop()
                for (h, r ,t) in kg.kg_dict_head[path[-1]]:
                    new_path = path+[r]+[t]
                    if new_path[-1]!=i:
                        find.append(new_path)
                    if new_path[-1] in ui.train_user_dict[i]:
                        path[i].append(new_path)

















kg = LoadKG('../dataset/last-fm.txt')
ui = LoadUIGraph('../dataset/')
DealingData.get_user2item_path(kg, ui, 6)




