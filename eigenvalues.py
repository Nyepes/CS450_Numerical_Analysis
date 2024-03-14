import numpy as np
import linear_equations
import least_squares

def rayleigh_quotient(A, x):
    return x @ A @ x / (np.linalg.norm(x) ** 2)
def power_iteration(A, k = 30, normalized = True):
    x = np.random.rand(A.shape[1])
    for i in range(k):
        x = A @ x 
        if (normalized): x = x / np.linalg.norm(x, ord=float('inf'))
    if (not normalized): return x
    else: return x, np.linalg.norm(A @ x, ord=float('inf'))
def inverse_iteration(A, k = 30, normalized = True):
    x = np.random.rand(A.shape[1])
    P, L, U = linear_equations.LuPartialPivots(A)
    val = 0
    for i in range(k): 
        x = P.T @ x
        y = linear_equations.forwardSubstitution(L, x)
        x = linear_equations.backSubstitution(U, y)
        val = np.linalg.norm(x, ord=float('inf'))
        x = x / val
    return x, val
def inverse_shift_iteration(A, x, k = 30, closestEigVal = 5):
    A_copy = A.copy()
    n, m = A.shape
    val = 0
    shift = 0
    for i in range(k):
        shift_amount = closestEigVal
        shift += shift_amount
        for j in range(m):
            A_copy[j, j] -= shift_amount
        try:
            P, L, U = linear_equations.LuPartialPivots(A_copy)
        except:
            return x, shift
        x = P.T @ x
        y = linear_equations.forwardSubstitution(L, x)
        x = linear_equations.backSubstitution(U, y)
        val = np.linalg.norm(x, ord=float('inf'))
        x = x / val
    return x, shift
def rayleigh_iteration(A, x, k = 30):
    A_copy = A.copy()
    n, m = A.shape
    val = 0
    shift = 0
    for i in range(k):
        r_quotient = rayleigh_quotient(A_copy, x)
        shift += r_quotient
        for j in range(m):
            A_copy[j, j] -= r_quotient
        try:
            P, L, U = linear_equations.LuPartialPivots(A_copy)
        except:
            return x, shift
        x = P.T @ x
        y = linear_equations.forwardSubstitution(L, x)
        x = linear_equations.backSubstitution(U, y)
        val = np.linalg.norm(x, ord=float('inf'))
        x = x / val
    return x, val + shift
def simulatenous_iteration(A, k = 30):
    n, m = A.shape
    X = np.random.rand(n, m)

    for i in range(k):
        X = A @ X
    return X
def orthogonal_iteration(A, k = 30):
    n, m = A.shape
    X = A.copy()
    R = np.eye(N = n, M = m)
    Q = np.zeros((n, m))
    for i in range(k):
        Q, R = least_squares.householder(X, reduced = True)
        X = A @ Q
    return X, R, Q
def to_hessenberg(A_param): 
    A = A_param.copy()
    n, m = A.shape
    Q = np.eye(n)
    for i in range(m - 1):
        e_i = np.zeros(n - i - 1)
        e_i[0] = 1
        z = A[i + 1: , i]
        u = z
        if (z[0] < 0):
            u = z - np.linalg.norm(z) * e_i
        else:
            u = z + np.linalg.norm(z) * e_i
        u = u / np.linalg.norm(u)
        for j in range(m):
            A[i + 1:, j] = A[i + 1:, j] - 2 * np.dot(u, A[i + 1:, j]) * u
            Q[i + 1:, j] = Q[i + 1:, j] - 2 * np.dot(u, Q[i + 1:, j]) * u


        for j in range(n):
            A[j, i + 1:] = A[j, i + 1:] - 2 * np.dot(u, A[j, i + 1:]) * u

    A[np.abs(A) < 1e-15] = 0
    return Q, A
def qr_iteration(A_param, k = 30):
    A = A_param.copy()
    Q_hessen, H = to_hessenberg(A)
    Q_hessen = Q_hessen.T
    for i in range(k):
        Q, R = least_squares.qr_from_upper_hessenberg(H, symmetric= True)
        H = R @ Q
        Q_hessen = Q_hessen @ Q
    return Q_hessen, H
    

A = np.array([
    [2.9766, 0.3945, 0.4198, 1.1159],
    [0.3945, 2.7328, -0.3097, 0.1129],
    [0.4198, -0.3097, 2.5675, 0.6079],
    [1.1159, 0.1129, 0.6079, 1.7231]
])
qr_iteration(A)
X, R, Q = orthogonal_iteration(A)
print(Q)
# A = np.random.rand(4, 4)
# x = np.array([0.1, 1.0, 0,0])
# vecs, val, Q = orthogonal_iteration(A)
# print(vecs)
# print(val)

# print(np.linalg.eig(A))
# a = np.zeros(4)
# a[0:2] = -vecs[0, 1:3] / vecs[0, 0]
# a[2] = 1
# a = a / np.linalg.norm(a)
# print(a)