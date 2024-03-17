# Import des librairies nécessaires
import numpy as np
from numpy.linalg import pinv
from numpy.linalg import norm
import pandas as pd
from sklearn.preprocessing import normalize
from math import*
import time
import matplotlib.pyplot as plt

# Étape 1 : Charger le dictionnaire appris
D = pd.read_csv('Dico-appris.csv', header=None, dtype=float).values


# Charger les signaux depuis le fichier Excel, en utilisant la première ligne comme en-tête
df_signals = pd.read_excel('Classeur1.xlsx', header=0)  # header=0 est la valeur par défaut


X1 = df_signals.iloc[:, 0].values
X2 = df_signals.iloc[:, 1].values
X3 = df_signals.iloc[:, 2].values


X1 = pd.to_numeric(X1, errors='coerce')
X2 = pd.to_numeric(X2, errors='coerce')
X3 = pd.to_numeric(X3, errors='coerce')


# Définition des algorithmes
# Fonction OMP corrigée
def OMP(x, D, eps, iterMax=1000):
    n, k = D.shape
    alpha = np.zeros(k)
    R = x.copy()
    index = []
    A = np.empty((n, 0))
    iterations = 0

    while np.linalg.norm(R) > eps and iterations < iterMax:
        ps = np.abs(D.T @ R)
        m = np.argmax(ps)
        index.append(m)
        A = D[:, index]
        alpha_solved = np.linalg.lstsq(A, x, rcond=None)[0]
        alpha = np.zeros(k)
        for i, ind in enumerate(index):
            alpha[ind] = alpha_solved[i]
        R = x - A @ alpha_solved
        iterations += 1

    return alpha, np.linalg.norm(R), iterations, index

# Fonction StOMP corrigée
def StOMP(x, D, eps, iterMax=1000, t=2):
    n, k = D.shape
    alpha = np.zeros(k)
    R = x.copy()
    index = []
    iterations = 0

    while np.linalg.norm(R) > eps and iterations < iterMax:
        ps = np.abs(D.T @ R)
        threshold = t * np.max(ps)
        indices_to_add = np.where(ps > threshold)[0]
        index = list(set(index) | set(indices_to_add))
        A = D[:, index]
        alpha_solved = np.linalg.lstsq(A, x, rcond=None)[0]
        alpha = np.zeros(k)
        for i, ind in enumerate(index):
            alpha[ind] = alpha_solved[i]
        R = x - A @ alpha_solved
        iterations += 1

    return alpha, np.linalg.norm(R), iterations, index

import numpy as np

def CoSaMP(x, D, eps, iterMax, s=9):
    n, k = np.shape(D)
    alpha = np.zeros(k)
    R = x.copy()
    iterations = 0

    while np.linalg.norm(R) > eps and iterations < iterMax:
        # Étape 1 : Corrélation
        ps = np.abs(D.T @ R)

        # Étape 2 : Sélection des 2s indices les plus significatifs
        m = np.argpartition(-ps, 2*s)[:2*s]

        # Étape 3 : Union des nouveaux indices avec les précédents
        # Dans votre version originale, 'index' était utilisé mais pas initialisé correctement pour le cycle.
        # 'V' doit contenir les indices actuels plus les nouveaux indices les plus significatifs.
        V = set(m)
        index = list(V)

        # Étape 4 : Résolution des moindres carrés sur l'ensemble d'indices sélectionnés
        A = D[:, index]
        alpha_solved = np.linalg.lstsq(A, x, rcond=None)[0]

        # Étape 5 : Garder les s plus grandes valeurs en magnitude et mettre à jour alpha
        idx_large_coefs = np.argsort(-np.abs(alpha_solved))[:s]
        alpha = np.zeros(k)
        for idx in idx_large_coefs:
            alpha[index[idx]] = alpha_solved[idx]

        # Étape 6 : Mise à jour du résidu
        R = x - D @ alpha
        iterations += 1

    return alpha, np.linalg.norm(R), iterations,None



# Fonction IRLS corrigée
def IRLS(x, D, eps, iterMax, p):
  n, k = np.shape(D)
  alpha0 = np.zeros(k)
  alpha = np.zeros(k)
  Q = np.zeros((k,k))
  it = 0

  # Initialisation de alpha
  alpha = D.T@np.linalg.inv(D@D.T)@x
  test = True
  # Boucle principale
  while test and it<iterMax:
    # Construction de la matrice Q
    for i in range(k):
      z = (np.abs(alpha[i]**2+eps))**(p/2-1)
      Q[i,i] = 1/z
    # Calcul de alpha
    alphaQ = Q@D.T@np.linalg.inv(D@Q@D.T)@x
    # Critère d'arrêt
    if np.linalg.norm(alpha-alpha0)<np.sqrt(eps)/100 and eps<10**(-8):
      test = False
    else:
      eps = eps/10
      alpha0=alpha
      it += 1

    return alpha, np.linalg.norm(x - D @ alpha),it,None

algos = {
    'OMP': lambda x, D: OMP(x, D, eps=1e-4, iterMax=1000),
    'StOMP': lambda x, D: StOMP(x, D, eps=1e-4, iterMax=1000, t=2),
    'CoSaMP': lambda x, D: CoSaMP(x, D, eps=1e-4, iterMax=1000, s=9),
    'IRLS': lambda x, D: IRLS(x, D, eps=1e-4, iterMax=1000, p=0.5)
}

# Initialisez des listes pour collecter les résultats
results_iterations = []
results_norm = []
results_error = []
results_sparsity = []

signals = [X1, X2, X3]
signal_names = ['X1', 'X2', 'X3']

# Exécutez les algorithmes et collectez les métriques
for signal_name, x in zip(signal_names, signals):
    for algo_name, algo_func in algos.items():
        alpha, norm_R, iterations, _ = algo_func(x, D)  # Exécutez l'algorithme
        error = np.linalg.norm(x - D @ alpha)  # Erreur de reconstruction
        sparsity = np.count_nonzero(alpha)  # Sparsité de la solution

        # Collecte des résultats
        results_iterations.append({
            'Signal': signal_name,
            'Algorithme': algo_name,
            'Itérations': iterations
        })
        results_norm.append({
            'Signal': signal_name,
            'Algorithme': algo_name,
            'Norme du Résidu': norm_R
        })
        results_error.append({
            'Signal': signal_name,
            'Algorithme': algo_name,
            'Erreur de Reconstruction': error
        })
        results_sparsity.append({
            'Signal': signal_name,
            'Algorithme': algo_name,
            'Sparsité': sparsity
        })

# Conversion des résultats en DataFrame
df_results_iterations = pd.DataFrame(results_iterations)
df_results_norm = pd.DataFrame(results_norm)
df_results_error = pd.DataFrame(results_error)
df_results_sparsity = pd.DataFrame(results_sparsity)

# Fonction pour tracer les métriques
def plot_metric(df, metric, title, ylabel):
    plt.figure(figsize=(10, 6))
    for signal_name in signal_names:
        df_signal = df[df['Signal'] == signal_name]
        plt.bar(df_signal['Algorithme'] + ' ' + signal_name, df_signal[metric])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Algorithme')
    plt.xticks(rotation=45)
    plt.legend(title='Signal')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Visualisation pour chaque métrique
plot_metric(df_results_iterations, 'Itérations', 'Nombre d\'itérations par algorithme', 'Itérations')
plot_metric(df_results_norm, 'Norme du Résidu', 'Norme du résidu par algorithme', 'Norme du Résidu')
plot_metric(df_results_error, 'Erreur de Reconstruction', 'Erreur de reconstruction par algorithme', 'Erreur')
plot_metric(df_results_sparsity, 'Sparsité', 'Sparsité de la solution par algorithme', 'Sparsité')