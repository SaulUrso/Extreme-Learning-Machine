import numpy as np


def cholensky(A):
    A = np.array(A)
    # noqa
    # if not np.all(np.linalg.eigvals(A) > 0):
    #    raise ValueError("Matrix is not positive definite")
    # if not (A == A.T).all():
    #    raise ValueError("Matrix is not symmetric")
    L = np.zeros_like(A)
    n = np.size(A, 0)
    for j in range(0, n):
        for i in range(j, n):
            if i == j:
                L[i][j] = np.sqrt(A[i][j] - np.dot(L[i, :i], L[i, :i]))
            else:
                L[i][j] = (A[i][j] - np.dot(L[i, :i], L[j, :i])) / L[j][j]
    return L, L.T



# if __name__ == "__main__":
#     total_error = 0
#     for i in range(1000):
#         n = 4
#         A = np.random.rand(n, n)
#         M = np.matmul(A, A.T)
#         L, U = cholensky(M)

#         """print("M:")
#         print(M)
#         print("Cholesky decomposition of M:")
#         print(L)
#         print(U)
#         print("A:")
#         print(np.matmul(L, U))"""

#         error = np.linalg.norm(M - np.matmul(L, U))
#         total_error += error

#     print("Total Error:")
#     print(total_error)
