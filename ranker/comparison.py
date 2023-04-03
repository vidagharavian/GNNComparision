import numpy as np
import scipy.sparse as sp


def davidScore(A):
    # s = davidScore(A)
    #   INPUTS:
    # A is a NxN matrix representing a directed network
    #   A can be weighted (integer or non-integer)
    #   A(i,j) = # of dominance interactions by i toward j. 
    #   A(i,j) = # of times that j endorsed i.
    #   OUTPUTS:
    # s is the Nx1 vector of Davids Score
    P = A / (A + A.transpose())  # Pij = Aij / (Aij + Aji)
    P = sp.lil_matrix(np.nan_to_num(P))
    P.setdiag(0)
    P = sp.csr_matrix(P)
    P.eliminate_zeros()
    w = P.sum(1)
    l = P.sum(0).transpose()
    w2 = P.dot(w)
    l2 = P.transpose().dot(l)
    s = w + w2 - l - l2
    return np.array(s).flatten()


def serialRank_matrix(A):
    # In serialRank, C(i,j) = 1 if j was preferred over i, so we need to transpose A.
    A = A.transpose()
    C = (A - A.transpose()).sign()
    n = A.shape[0]
    S = C.dot(C.transpose()) / 2
    S.data += n / 2
    return S
