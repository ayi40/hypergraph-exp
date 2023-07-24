import torch
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
