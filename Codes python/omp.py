import numpy as np
from numpy import linalg as LA

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

def omp_compressed(y, Phi, D, eps, iterMax):
    # y est le signal compressé, Phi est la matrice de mesure, et D est le dictionnaire.
    M, N = Phi.shape  # Récupération des dimensions de la matrice de mesure
    _, k = D.shape  # Récupération des dimensions du dictionnaire
    alpha = np.zeros(k)  # Initialisation des coefficients
    R = y  # Initialisation du résidu avec le signal compressé
    index = []  # Liste pour stocker les indices des atomes sélectionnés
    A = np.empty((M, 0))  # Matrice initialement vide qui va contenir les atomes sélectionnés projetés

    it = 0  # Compteur d'itérations
    while LA.norm(R) > eps and it < iterMax:  # Critère d'arrêt
        ps = np.zeros(k)
        for j in range(k):
            D_j = np.dot(Phi, D[:, j])  # Application de la matrice de mesure au j-ème atome
            ps[j] = np.abs(np.dot(D_j.T, R)) / LA.norm(D_j)
        m = np.argmax(ps)
        index.append(m)
        A = np.column_stack([A, np.dot(Phi, D[:, m])])  # Ajouter l'atome projeté
        alpha_temp = LA.lstsq(A, y, rcond=None)[0]  # Solution des moindres carrés
        alpha[index] = alpha_temp
        R = y - np.dot(A, alpha_temp)  # Mise à jour du résidu
        it += 1
    return alpha, R, it, index