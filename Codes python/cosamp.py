import numpy as np
from numpy.linalg import norm
from numpy.linalg import pinv

def CoSaMP(x, D, epsilon, iterMax, s):
    n, k = np.shape(D)
    alpha = np.zeros(k)
    R = x
    index = []
    A = np.empty((n, 0))
    it = 0
    ps = np.zeros(k)

    while np.linalg.norm(R) > epsilon and it < iterMax:
        for j in range(k):
            ps[j] = np.abs(np.dot(D[:, j].T, R)) / np.linalg.norm(D[:, j])

        # Sélection des atomes avec la plus grande contribution
        m = np.argpartition(ps, -2*s)[-2*s:]
        V = set(index)|set(m)
        index = list(V)
        # Application des moindes carrés
        A = D[:, index]
        alpha[index] = np.dot(np.linalg.pinv(A), x)
        index = np.argpartition(np.abs(alpha), -s)[-s:]
        # On recommence avec les moindres carrés avec les s atomes retenus
        alpha = np.zeros(k)
        A = D[:,index]
        alpha[index] = np.dot(np.linalg.pinv(A), x)
        # Actualisation des résidus
        R = x - np.dot(A, alpha[index])
        it += 1

    return alpha, R, it, index
