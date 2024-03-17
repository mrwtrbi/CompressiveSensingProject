import numpy as np
from numpy.linalg import norm
import pandas as pd

# Construction des matrices de mesure
def phi1(m, n):
  p = np.random.rand(m, n)
  return p

def phi2(m, n):
  proba = 0.25
  p = 2*np.random.binomial(1, proba, size=(m,n))-np.ones((m,n))
  return p

def phi3(m,n):
  proba = 0.25
  p = np.random.binomial(1, proba, size=(m,n))
  return p

def phi4(m,n):
  p = np.random.normal(0, 1/np.sqrt(m), size=(m,n))
  return p

# Fonction calculant la cohérence mutuelle
def coherence(phi, D):
  m, n = np.shape(phi)
  n, k = np.shape(D)
  coher = np.zeros((m,k))
  for i in range(m):
    for j in range(k):
      z = np.dot(phi[i,:].T, D[:,j])/(np.linalg.norm(phi[i,:])*np.linalg.norm(D[:,j]))
      # Calcul de la valeur absolue
      coher[i,j] = np.abs(z)
  a = np.max(coher)
  c = a*np.sqrt(n)
  return c

# Import du dictionnaire appris par k-SVD
D = np.array(pd.read_csv('./Dico-appris.csv'))

# Taille du dictionnaire
n = D.shape[0]

# Calcul des cohérences mutuelles
mesures = [0.15, 0.2, 0.25, 0.3, 0.5]

for i in mesures:
  m = int(i*n)
  # Calcul des matrices de mesure
  p1 = phi1(m, n)
  p2 = phi2(m, n)
  p3 = phi3(m, n)
  p4 = phi4(m, n)

  # Calcul des cohérences
  coher1 = coherence(p1, D)
  coher2 = coherence(p2, D)
  coher3 = coherence(p3, D)
  coher4 = coherence(p4, D)

  # Afficher les cohérences mutuelles
  print(f" Pour {i}% les cohérences mutuelles sont en utilisant :\nPhi1 = {coher1}, Phi2 = {coher2}, Phi3 = {coher3} et Phi4 = {coher4}\n")
