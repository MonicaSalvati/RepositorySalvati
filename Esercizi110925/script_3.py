import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Caricamento del dataset
df = pd.read_csv("Iris.csv")

# Divisione features e target
X = df.drop("Species", axis=1)
y = df["Species"]

# Prima si separa il test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)

# Poi si separa il validation dal resto
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42)


# Modello Decision Tree Classifier
tree = DecisionTreeClassifier(max_depth=2, random_state=42)
# Addestramento del modello
tree.fit(X_train, y_train)

# Predizioni sul validation set
y_pred_val = tree.predict(X_val)
print("Classification report su validation set:")
print(classification_report(y_val, y_pred_val, digits=3))


# Predizioni sul test set
y_pred_tree = tree.predict(X_test)
print("Classification report su test set:")
print(classification_report(y_test, y_pred_tree, digits=3))