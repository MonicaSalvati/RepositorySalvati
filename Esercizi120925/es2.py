import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

## PUNTO 1
#Caricamento dataset
df = pd.read_csv("Wholesale customers data.csv")

# EscludO Channel e Region
X = df.drop(columns=["Channel", "Region"])

## PUNTO 2
# Standardizzazione
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

## PUNTO 3
#K-distance plot per stimare eps 
k = 5
nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
distances, indices = nbrs.kneighbors(X_scaled)

# Distanza dal k-esimo vicino
distances = np.sort(distances[:, k-1])

plt.figure(figsize=(10,6))
plt.plot(range(len(distances)), distances, marker='.', linestyle='-', color='b')
plt.title(f"K-distance plot (k={k})", fontsize=14)
plt.xlabel("Punti ordinati", fontsize=12)
plt.ylabel(f"{k}-esima distanza", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)

plt.legend()
plt.show()

# PUNTO 4
# DBSCAN con parametri scelti 
eps_value = 2  
min_samples_value = 5

db = DBSCAN(eps=eps_value, min_samples=min_samples_value)
labels = db.fit_predict(X_scaled)

df["Cluster"] = labels

## PUNTO 5
# Valutazione 
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_outliers = list(labels).count(-1)

print("Numero cluster trovati:", n_clusters)
print("Numero outlier:", n_outliers)

if n_clusters > 1:
    sil_score = silhouette_score(X_scaled, labels)
    print("Silhouette Score:", sil_score)
else:
    print("Silhouette Score non calcolabile (solo 1 cluster trovato)")

