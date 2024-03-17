import numpy as np
from numpy.linalg import norm
from numpy.linalg import inv

def IRLS(x, D, eps, iterMax, p):
  n, k = np.shape(D)
  alpha0 = np.zeros(k)
  alpha = np.zeros(k)
  Q = np.zeros((k,k))
  it = 0

  # Initialisation de alpha
  alpha0 = D.T@np.linalg.inv(D@D.T)@x
  test = True
  # Boucle principale
  while test and it<iterMax:
    # Construction de la matrice Q
    for i in range(k):
      z = (np.abs(alpha0[i]**2+eps))**(p/2-1)
      Q[i,i] = 1/z
    # Calcul de alpha
    alpha = Q@D.T@np.linalg.inv(D@Q@D.T)@x
    # Critère d'arrêt
    if np.linalg.norm(alpha-alpha0)<np.sqrt(eps)/100 and eps<10**(-8):
      test = False
    else:
      eps = eps/10
      alpha0=alpha
      it += 1

  return alpha, it