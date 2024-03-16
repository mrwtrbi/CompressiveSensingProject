import numpy as np

# Algorithme OMP
def OMP(x, D, epsilon, iterMax):
    n, k = np.shape(D)
    alpha = np.zeros(k)
    R = x
    index = []
    A = np.empty((n, 0))
    iterations = 0
    ps = np.zeros(k)

    while np.linalg.norm(R) > epsilon and iterations < iterMax:
        for j in range(k):
            ps[j] = np.abs(np.dot(D[:, j].T, R)) / np.linalg.norm(D[:, j])

        m = np.argmax(ps)
        index.append(m)
        A = np.column_stack((A, D[:, m]))
        alpha[index] = np.dot(np.linalg.pinv(A), x)
        R = x - np.dot(A, alpha[index])
        iterations += 1

    return alpha, R, iterations, index