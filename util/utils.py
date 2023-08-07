import torch
import numpy as np
class MatrixMethod:
    def __init__(self):
        pass

    @staticmethod
    def matrix_dot(A, B, return_list=False):
        n_fold = len(A)
        res=[]
        for i in range(n_fold):
            r = torch.matmul(A[i], B)
            res.append(r)
        if return_list:
            return res
        res = torch.concatenate(res, axis=0)
        return res

class utils:
    def __init__(self):
        pass

    @staticmethod
    def find_uipath(user, item, pathdic, pathlen, padding_idx):
        # get pos path
        ui_path = pathdic[user][item]
        # add [user,item] path
        newpath = [user, item]
        if len(newpath) < pathlen:
            newpath = newpath + [padding_idx] * (pathlen - len(newpath))
        ui_path.append(newpath)
        return ui_path

    @staticmethod
    def find_uipath_batch(user_batch, item_batch, pathdic, pathlen, padding_idx):
        user_batch = user_batch.numpy()
        item_batch = item_batch.numpy()
        path=[]
        path_idx = []
        for idx in range(len(user_batch)):
            u = user_batch[idx]
            i = item_batch[idx]
            p = utils.find_uipath(u, i, pathdic, pathlen, padding_idx)
            path.append(p)
            path_idx+=[idx]*len(p)
        path_idx = np.array(path_idx)
        path = np.concatenate(path, axis=0)
        return torch.tensor(path), torch.tensor(path_idx)


