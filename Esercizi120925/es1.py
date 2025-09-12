import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

## PUNTO 1 
# Caricamento del dataset
df = pd.read_csv("Mall_Customers.csv")
# Selezione delle colonne Annual Income (k$) e Spending Score (1-100)
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

## PUNTO 2
# Standardizzazione delle feature
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

## PUNTO 3 
# Scelta di k in base al silhouette score
silhouette_scores=[]
K = range(2,11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
# Grafico silhouette score per diversi k
plt.figure(figsize=(8, 5))
sns.lineplot(x=list(K), y=silhouette_scores, marker='o')
plt.title("Silhouette Score al variare di k")
plt.xlabel("Numero di cluster (k)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.tight_layout()

## PUNTO 4
# KMeans con k = 5, silhouette_score più alto
kmeans = KMeans(n_clusters=5, random_state= 42)
labels = kmeans.fit_predict(X_scaled)

## PUNTO 5
# Visualizzazione con i cluster trovati
plt.figure(figsize=(6, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='red', marker='X', s=200, label='Centroidi')
plt.title("Cluster trovati con k-Means")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()
plt.grid(True)
plt.show()

## PUNTO 6 
# Interpretazione
# Guardando i cluster risultanti questi risultano molto "sparsi". Ciò probabilmente è dovuto al fatto che non è stato eseguito un feature engineer ovvero  
# una riformulazione delle feature oppure le feature scelte non permettevano una buona divisione dei dati. 