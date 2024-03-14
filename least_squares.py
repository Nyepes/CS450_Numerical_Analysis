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
        for j in range(m):
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
    print(Q)
    print(R)
    print()
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
            print(Q)
            print(R)
            print()
    return Q.T, R
def solve_llsq(A, b):
    Q, R = householder(A, reduced= True)
    return backSubstitution(R, Q.T @ b)
def solve_rank_deficient(A, b):
    n, m = A.shape
    mat = np.zeros((n + m, m))
    mat[:n, :] = A
    mat[n:, :] = np.eye(m)
    return solve_llsq(A, b)


A = np.array([
    [1, -1, 0, 0],
    [1, -1, 1, 0],
    [0, -1, 1, 1],
    [0, 0, 0.25, 1],
])
Q, R = givens_rotations(A)
print()
print(Q)
# A = np.random.rand(, 3)
