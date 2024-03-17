import pandas as pd
import numpy as np

import numpy as np
from numpy import linalg as LA

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

# Import du dictionnaire appris par k-SVD
D = np.array(pd.read_csv('/content/CompressiveSensingProject/Dico-appris.csv'))

# Import des signaux test
X = np.array(pd.read_excel("/content/CompressiveSensingProject/DonneesProjet.xlsx", sheet_name="vecteurs pour valider"))
X = X[1:,:]

# Construction de la matrice de mesure aléatoire sélectionnée précédemment
n = X.shape[0]
i = 0.25 # Mesure sélectionnée précédemment
m = int(i*n)
phi = phi4(m,n)

# Echantillonage compressif
y = np.dot(phi, X)

# Reconstruction des signaux
s = 10

# Initialiser la matrice reconstruite
X_reconstruit = np.zeros_like(X)  

# Reconstruction de chaque signal compressé
for i in range(y.shape[1]):
  alpha, R, it, index = omp_compressed(y[:, i], phi, D, eps, iterMax)
  # Reconstruction du signal à partir des coefficients
  X_reconstruit[:, i] = np.dot(D, alpha)

# Convertir le dictionnaire numpy en DataFrame pandas
df = pd.DataFrame(X)

# Enregistrer le tableau NumPy dans un fichier CSV
df.to_csv("X_reconstruit.csv", index=False, header=True)  

X_reconstruit = pd.read_csv("./X_reconstruit.csv")

# Calcul du MSE pour chaque colonne
MSE = ((X - X_reconstruit) ** 2).mean(axis=0)

print("La Mean Square Error pour chaque atome")
for i, mse in enumerate(MSE):
    print(f"Signal {i + 1}: {mse:.4f}")