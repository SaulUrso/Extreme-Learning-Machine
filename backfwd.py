import time

import numpy as np

from cholensky import cholensky


def forward_substitution(L, B):
    # Final matrix dimensions (n, m)
    n = L.shape[0]
    m = B.shape[1]

    X = np.zeros_like(B, dtype=np.double)

    # Resolve the system for each column of B
    for k in range(m):
        for i in range(n):
            # Summation of the terms already calculated
            sum = 0
            for j in range(i):
                sum += L[i, j] * X[j, k]
            # Computing x variable
            X[i, k] = (B[i, k] - sum) / L[i, i]

    return X


def back_substitution(U, B):
    n = U.shape[0]
    m = B.shape[1]
    X = np.zeros_like(B, dtype=np.double)

    for k in range(m):
        for i in range(n - 1, -1, -1):
            sum = 0
            for j in range(i + 1, n):
                sum += U[i, j] * X[j, k]
            X[i, k] = (B[i, k] - sum) / U[i, i]
    return X


def solve_system(M, B):
    start = time.process_time()
    L, U = cholensky(M)
    end = time.process_time()
    Z = forward_substitution(L, B)
    W = back_substitution(U, Z)
    return W, end - start


# if __name__ == "__main__":
#     n = 4
#     total_error = 0
#     for i in range(1000):
#         A = np.random.rand(n, n)
#         M = np.matmul(A, A.T)
#         B = np.random.rand(n, 1)

#         W = solve_system(M, B)
#         X = np.linalg.solve(M, B)

#         error = np.linalg.norm(W - X)

#         if error > 1:
#             print("Error:")
#             print(error)
#             print("M:")
#             print(M)
#             print("B:")
#             print(B)
#             print("W:")
#             print(W)
#             print("X:")
#             print(X)

#         total_error += error
#     print("Total Error:")
#     print(total_error)
