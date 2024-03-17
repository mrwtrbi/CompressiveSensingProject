import numpy as np
from numpy.linalg import norm

def StaMP(x, D, eps, iterMax, t):
  n, k = np.shape(D)
  alpha = np.zeros(k)
  R = x
  index = []
  A = np.empty((n,0))
  it = 0
  ps = np.zeros(k)
  while np.linalg.norm(R)>eps and it<iterMax:
    for j in range(k):
      ps[j] = np.abs(np.dot(D[:,j].T,R))/np.linalg.norm(D[:,j])
    # Calcul du seuil
    Seuil = t*np.linalg.norm(R)/np.sqrt(k)
    # Sélection des atomes dont la contribution est supérieure au seuil
    m = np.where(ps>Seuil)[0]
    # Ajout des indices aux anciens
    V = set(index)|set(m)
    index = list(V)
    # Matrice formée des atomes sélectionnés
    A = D[:,index]
    # Application des moindres carrés
    alpha[index] = np.dot(np.linalg.pinv(A),x)
    # Actualisation des résidus
    R = x - np.dot(A, alpha[index])
    it += 1
  return alpha, R, it, index
