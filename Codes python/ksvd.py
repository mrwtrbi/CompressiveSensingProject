import numpy as np

# Algorithme k-SVD
def ksvd(X, D0, m):
  n, l = np.shape(X)
  n, k = np.shape(D0)
  # Matrice Lambda des représentations parcimonieuses
  A = np.zeros((k,l))
  # Initialisation du dictionnaire
  D = D0
  # Boucle principale de l'algorithme
  for j in range(m):
    # Utilisation de OMP pour calculer chaque représentation parci de Lambda
    for p in range(l):
      alpha, R, it, index = OMP(X[:,p], D, eps, iterMax)
      # Par soucis de notation, on note la matrice Lambda → A
      A[:, p] = alpha
    # Actualisation des atomes du dictionnaire l'un après l'autre
    for i in range(k):
      # Erreur isolée de l'atome i
      Ei = X - np.dot(D,A) + np.dot(np.matrix(D[:,i]).T, np.matrix(A[i,:]))
      # Si les coefficients sont nuls
      if A[i,:].all() == 0:
        # Appliquer des coefficients aléatoires
        v = np.random.rand(n)
        D[:,i] = v/np.linalg.norm(v)
      # Si les coefficients sont non nuls
      else:
        # Construction de la matrice Omega i
        wi = np.nonzero(A[i,:])
        c = len(wi)
        Omega_i = np.zeros((l,c))
        for q in range(c):
          Omega_i[wi[q],q] = 1
        # Calcul de l'erreur de reconstruction
        EiR = np.dot(Ei, Omega_i)
        U,S,V = np.linalg.svd(EiR)
        # Actualisation de l'atome i
        D[:,i] = U[:,0].T
        A[i,wi] = np.dot(S[0],V[:,0].T)

  return D, A