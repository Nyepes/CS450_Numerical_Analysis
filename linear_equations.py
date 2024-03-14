import numpy as np


def lu(X):
    A = X.copy()
    n = len(A)
    def recurse(A, L, U, i):
        if (i == n):
            return L, U
        U[i, i:] = A[i, i:]
        L[i:, i] = A[i:, i] / U[i, i]
        A[i+1:, i+1:] -= np.outer(L[i + 1:, i], U[i, i + 1:])
        return recurse(A, L, U, i + 1)
    
    L = np.zeros_like(A)
    U = np.zeros_like(A)
    return recurse(A, L, U, 0)
def LuPartialPivots(X):
    def swapRows(mat, i, j):
        for col in range(len(mat)):
            temp = mat[i, col]
            mat[i, col] = mat[j, col]
            mat[j, col] = temp
    def findMaxRowVal(mat, col):
        idxMax = col
        for i in range(col, len(mat)):
            if (np.abs(mat[i, col]) > np.abs(mat[idxMax, col])):
                idxMax = i
        return idxMax
    A = X.copy()
    n = len(A)
    P = np.eye(n)
    L = np.zeros_like(A)
    U = np.zeros_like(A)
    for i in range(n):
        idxMax = findMaxRowVal(A, i)
        swapRows(A, idxMax, i)
        swapRows(P, idxMax, i)
        swapRows(L, idxMax, i)
        swapRows(U, idxMax, i)
        U[i, i:] = A[i, i:]
        if(A[i, i] == 0): raise Exception("Singular")
        L[i:, i] = A[i:, i] / A[i, i]
        A[i+1:, i+1:] -= np.outer(L[i + 1:, i], U[i, i + 1:])

    return P, L, U
def checkLUExistence(A):
    n = len(A)
    for i in range(n):
        if (np.linalg.det(A[0:i, 0:i]) == 0): return False
    return True
def LUBanded(A, b):
    n = len(A)
    remaining = A.copy()
    L = np.zeros_like(A)
    U = np.zeros_like(A)
    for i in range(n):
        U[i, i : i + b] = remaining[i, i : i + b]
        L[i : i + b, i] = remaining[i : i + b, i] / remaining[i, i]
        remaining[i + 1: i + b, i + 1 : i + b] -= np.outer(L[i + 1 : i + b, i], U[i, i + 1: i + b])
    return L, U
def forwardSubstitution(L, y):
    n = len(L)
    for i in range(n):
        y[i] = y[i] / L[i, i]
        y[i + 1:] -= y[i] * L[i + 1:, i]
    return y 
def backSubstitution(U, b):
    n = len(b)
    for i in range(n - 1, -1, -1):
        b[i] = b[i] / U[i, i]
        b[:i] -= b[i] * U[:i, i]
    return b
def cholesky(A):
    n = len(A)
    L = np.zeros_like(A)
    for i in range(n):
        L[i, i] = np.sqrt(A[i, i])
        L[i + 1:, i] = A[i + 1:, i] / L[i, i]
        A[i+1:, i+1:] -= np.outer(L[i+1:, i], L[i+1:, i])
    return L

