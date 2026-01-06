import numpy as np
import torch


def inv_sherman_morrison(u, A_inv):
    """Inverse of a matrix with rank 1 update.
    """
    Au = np.dot(A_inv, u)
    A_inv -= np.outer(Au, Au) / (1 + np.dot(u.T, Au))
    return A_inv


def CB(alpha, x, M):
    return alpha * np.sqrt(np.dot(np.dot(x.T, np.linalg.inv(M)), x))


def probCal(realvalue_vec, rank, args):
    prob = 1.
    for i in range(args.card):
        if i == 0:
            denominator = torch.cat([torch.exp(realvalue_vec[j]) for j in list(set(list(range(args.solDim))) - set(rank[:(j - 1)]))], 0).sum()
        else:
            denominator = torch.cat([torch.exp(realvalue_vec[j]) for j in list(set(list(range(args.solDim))) - set(rank[:(j - 1)]))], 0).sum()
        prob = prob * torch.exp(realvalue_vec[rank[i]]) / denominator
    return prob


def checkFea(x, legalList):
    cnt = 0
    cntT = 0
    for i in range(len(legalList.keys())):
        for j in legalList[i]:
            cntT += 1
            if x[j] + x[i] > 1:
                cnt += 1
    return cnt / cntT / 2


def networkOutput_to_posdata(realvalue_vec):
    if type(realvalue_vec) != np.ndarray:
        realvalue_vec = realvalue_vec.detach().numpy()
    u = np.random.uniform(size=np.shape(realvalue_vec))
    z = -np.log(-np.log(u))
    rank = np.argsort(z + realvalue_vec)[::-1]
    return rank


def posdata_to_onehot(rank):
    matrix = np.zeros([args.card, args.solDim])
    for i in range(args.card):
        matrix[i][rank[i]] = 1
    return matrix.reshape(args.card * args.solDim)


def posdata_to_binaryvec_withoutpos(rank):
    vec = np.zeros(args.solDim)
    for i in rank:
        vec[i] = 1
    return vec


def onehot_to_binaryvec_withoutpos(onehot):
    if type(onehot) != np.ndarray:
        onehot = onehot.detach().numpy()
    vec = np.zeros(args.solDim)
    for i in range(args.card):
        vec[np.argwhere(onehot[i] > 0.1)] = 1
    return vec


def onehot_to_rank(onehot):
    if type(onehot) != np.ndarray:
        onehot = onehot.detach().numpy()
    onehot = onehot.reshape([args.card, args.solDim])
    vec = []
    for i in range(args.card):
        if len(np.argwhere(onehot[i] > 0.1)) != 0:
            vec.append(np.argwhere(onehot[i] > 0.1)[0])
    return np.array(vec)
