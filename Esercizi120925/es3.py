import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import time

## PUNTO 1 
# Caricamento del dataset
df = pd.read_csv("Datasets/train.csv")
print(df.head())
# Divisione in X e y
X = df.drop(columns=["label"])
y = df["label"]

## PUNTO 2
# Standardizzazione dei dati
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled.shape[1])

## PUNTO 3
# PCA per mantenere il 95% della varianza 
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
print(f'Numero di componenti che mantengono il 95% della varianza: {X_pca.shape[1]}')

## PUNTO 4 
# Split 80/20
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, stratify=y, test_size=0.2, random_state=42)

## PUNTO 5
# Decision Tree con PCA
tree = DecisionTreeClassifier(random_state=42)
start_pca = time.time()
tree.fit(X_train, y_train)
end_pca = time.time()
print(f'Tempo di training con PCA {round(end_pca-start_pca,2)}')

## PUNTO 6
# Valutazione Decision Tree con PCA
y_pred = tree.predict(X_test)
print(f"Accuracy (con PCA): {accuracy_score(y_test, y_pred)} s")

## PUNTO 7 
# Decision Tree senza PCA
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=42)
tree_no_pca = DecisionTreeClassifier(random_state=42)
start = time.time()
tree_no_pca.fit(X_train_1, y_train_1)
end = time.time()
print(f'Tempo di training senza PCA {round(end-start,2)} s')
y_pred_1 = tree_no_pca.predict(X_test_1)
print(f"Accuracy (senza PCA): {accuracy_score(y_test_1, y_pred_1)}")

## PUNTO 8 
# Il tempo di training con la PCA è maggiore rispetto al tempo di training senza la PCA. 
# L'accuracy con la PCA sembra minore rispetto all'accuracy senza la PCA.




