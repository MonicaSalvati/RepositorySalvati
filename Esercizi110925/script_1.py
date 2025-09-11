import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

## PUNTO 1
# Caricamento del dataset
df = pd.read_csv("Iris.csv")
# Prime cinque righe del dataset
print(df.head(5))

## PUNTO 2
#Divisione dei dati in train/test(80%-20%)
X = df.drop("Species", axis=1)
y = df["Species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.20, random_state=42)

## PUNTO 3
# Modello Decision Tree Classifier
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
# Addestramento del modello
tree.fit(X_train, y_train)
# Predizioni del modello
y_pred_tree = tree.predict(X_test)

# PUNTO 4 e 5 
# Classification report
print("Classification report:")
print(classification_report(y_test, y_pred_tree, digits=3))