import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    A = np.asarray(A, dtype=float)
    n, m = A.shape
    A_t = np.zeros((m, n))
    # A_t = np.zeros((len(A[0]), len(A)))

    for i in range (n):
        for j in range(m):
            A_t[j, i] = A[i, j]
            
    return A_t

