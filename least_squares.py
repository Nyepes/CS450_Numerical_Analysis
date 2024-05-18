import numpy as np
from linear_equations import backSubstitution

def gram_schmidt(A, reduced = False):
    n = A.shape[0]
    m = A.shape[1]
    if (reduced):
        Q = np.eye(N = n, M = m)
        R = np.zeros((m, m))
    else:
        Q = np.eye(n)
        R = np.zeros((n, m))

    for i in range(m):
        q = A[:, i]
        for j in range(i):
            comp = np.dot(A[:, i], Q[:, j])
            q = q - comp * Q[:, j]
            R[j, i] = comp
        norm =  np.linalg.norm(q)
        if (i < m): 
            Q[:, i] = q / norm
            R[i, i] = norm

    return Q, R
def modified_gram_schmidt(A, reduced = False):
    n = A.shape[0]
    m = A.shape[1]
    if (reduced):
        Q = np.eye(N = n, M = m)
        R = np.zeros((m, m))
    else:
        Q = np.eye(m)
        R = np.zeros((m, n))
    for i in range(m):
        q = A[:, i]
        for j in range(i):
            comp = np.dot(q, Q[:, j])
            q -= comp * Q[:, j]
            R[j, i] = comp
        norm = np.linalg.norm(q)
        if (i < m):
            Q[:, i] = q / norm
            R[i, i] = norm
    return Q, R
def householder(A, reduced = False):
    n = A.shape[0]
    m = A.shape[1]
    Q = np.eye(n)
    R = A.copy()
    for i in range(min(n, m - 1)):
        ei = np.zeros(n - i)
        ei[0] = 1
        z = R[i:, i]
        znorm = np.linalg.norm(z)
        if (R[i, i] < 0):
            u = (z - znorm * ei) 
        else:
            u = (z + znorm * ei)
        unorm = u.T @ u
        for j in range(i, m):
            R[i:, j] = R[i:, j] - 2 * np.dot(u, R[i:, j]) * u / unorm
        for j in range(n):
            Q[i:, j] = Q[i:, j] - 2 * np.dot(u, Q[i:, j]) * u / unorm
    if (reduced):
        Q = Q[:m, :n]
        R = R[:m, :m]
    return Q.T, R 
def givens_rotations(A, reduced = False):
    n = A.shape[0]
    m = A.shape[1]
    Q = np.eye(n)
    R = A.copy()
    for j in range(m - 1, -1, -1):
        for i in range(n - 2, j - 1, -1):
            hyp = np.linalg.norm(R[i: i + 2, j])
            if (hyp == 0): continue
            cos = R[i, j] / hyp
            sin = R[i + 1, j] / hyp
            rot = np.array([
                [cos, sin],
                [-sin, cos]
            ])
            R[i: i + 2, j:] = rot @ R[i: i + 2, j:] 
            Q[i: i + 2, :] = rot @ Q[i: i + 2, :] 
            R[i + 1, j] = 0
    return Q.T, R
def solve_llsq(A, b):
    Q, R = householder(A, reduced= True)
    return backSubstitution(R, Q.T @ b)
def qr_from_upper_hessenberg(A_param, symmetric = False):
    n, m = A_param.shape
    Q = np.eye(n)
    A = A_param.copy()
    for i in range(m - 1):
        hyp = np.linalg.norm(A[i:i+2, i])
        cos = A[i, i] / hyp
        sin = A[i + 1, i] / hyp
        G = np.array([[cos, sin], [-sin, cos]])
        if (symmetric):
            A[i: i + 2, i: i + 4] = G @ A[i: i + 2, i: i + 4] 
            Q[i: i + 2, : ] = G @ Q[i: i + 2, : ]
        else:
            A[i: i + 2, i:] = G @ A[i: i + 2, i:] 
            Q[i: i + 2, :] = G @ Q[i: i + 2, :]
    return Q.T, A

import numpy as np
import scipy.linalg as sla
import matplotlib.pyplot as plt

# 

def getRankKCrossA(A, rank):
    n = A.shape[0]
    m = A.shape[1]
    QA, RA, PA = sla.qr(A, pivoting = True)
    QAt, RAt, PAt = sla.qr(A.T, pivoting = True)
    A_r = np.zeros((n, m))
    Arow = np.zeros((rank, m))
    Acol = np.zeros((n, rank))
    A_rr = np.zeros((rank, rank))
    Arow = (A.T[:, PAt[:rank]]).T
    Acol = A[:, PA[:rank]]
    A_rr = A[PAt[:rank], :]
    A_rr = A_rr[:, PA[:rank]]
    return Acol @ np.linalg.inv(A_rr) @ Arow, A_rr
