"""
K best assignment using murty&Gibbs

Note:Gibbs sampling can't solve the matrix that row == col
The L is :
[l11, l12, ..., l1n, l01, inf, ..., inf
 l21, l22, ..., l2n, inf, l02, ..., inf
  .    .         .    .    .         .
  .    .         .    .    .         .
  .    .         .    .    .         .
 lm1, lm2, ..., lmn, inf, inf, ..., l0n]
"""

import numpy as np


def kbestbygibbs(L, K, num_Iteration=100):
    K = int(K)
    n = L.shape[0]
    m = L.shape[1] - n
    assignment = np.empty((n, num_Iteration), dtype=int)
    cost = []
    currsolu = np.arange(m, n + m)
    assignment[:, [0]] = currsolu.reshape((-1, 1))

    cost_i = 0
    for i in range(n):
        cost_i += L[i, currsolu[i]]
    cost.append(cost_i)

    for solu in range(1, num_Iteration):
        cost_i = 0
        for row in range(n):
            tempsamp = np.exp(-L[row, :])
            tempsamp[currsolu[0:row]] = 0
            tempsamp[currsolu[row + 1:n]] = 0

            idxold = np.nonzero(tempsamp)[0]
            tempsamp = tempsamp[idxold]
            tempsamp = np.append(0, np.cumsum(tempsamp) / np.sum(tempsamp))
            index = tempsamp <= np.random.rand()

            idxold = idxold[index[:-1]]
            currsolu[row] = idxold[-1]
            cost_i += L[row, currsolu[row]]

        cost.append(cost_i)
        assignment[:, [solu]] = currsolu.reshape((-1, 1))

    assignment, index = np.unique(assignment, axis=1, return_index=True)
    cost = np.array(cost)[index]
    index = np.argsort(cost)

    cost = cost[index]
    assignment = assignment[:, index]

    if K < num_Iteration and K < len(cost):
        assignment = assignment[:, 0:K]
        cost = cost[0:K]

    return cost, assignment


def kbestbymurty(L, K):
    pass
