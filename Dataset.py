import torch
from torch.utils.data import Dataset
import numpy as np

class Dataset(Dataset):
    def __init__(self, datas, datas_user_dict,  n_items, dataset_type):
        self.datas = datas
        self.datas_user_dict = datas_user_dict
        self.dataset_type = dataset_type
        self.n_items = n_items

    def __getitem__(self, item):
        u, pos_i = self.datas[item][0], self.datas[item][1]
        if self.dataset_type == 'test':
            return u, pos_i
        elif self.dataset_type == 'train':
            return u, pos_i, self._sample_neg(u)

    def _sample_neg(self, user):
        while True:
            neg_i_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
            if neg_i_id not in self.datas_user_dict[user]:
                return neg_i_id

    def __len__(self):
        return len(self.datas)
