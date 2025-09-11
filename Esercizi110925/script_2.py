import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

## PUNTO 1
# Caricamento dataset Credit Card 
df = pd.read_csv("creditcard.csv")

# Separazione feature e target
X = df.drop("Class", axis=1)
y = df["Class"]

## PUNTO 2 
# Divisione del dataset in train e test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

## PUNTi 3 e 4
# -------------------------
# Modello 1 – Decision Tree
# -------------------------
tree = DecisionTreeClassifier(class_weight="balanced", random_state=42)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)

print("Decision Tree:")
print(classification_report(y_test, y_pred_tree, digits=3))

# -------------------------
# Modello 2 – Random Forest
# -------------------------
forest = RandomForestClassifier(class_weight="balanced", random_state=42)
forest.fit(X_train, y_train)
y_pred_forest = forest.predict(X_test)

print("Random Forest:")
print(classification_report(y_test, y_pred_forest, digits=3))

# -------------------------
# Applicazione di SMOTE
# -------------------------
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
# -------------------------
# Modello 1 – Decision Tree
# -------------------------
tree_r = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_r.fit(X_train_resampled, y_train_resampled)
y_pred_tree_r = tree_r.predict(X_test)

print("Decision Tree with SMOTE:")
print(classification_report(y_test, y_pred_tree_r, digits=3))

# -------------------------
# Modello 2 – Random Forest
# -------------------------
forest_r = RandomForestClassifier(max_depth=5, random_state=42)
forest_r.fit(X_train_resampled, y_train_resampled)
y_pred_forest_r = forest_r.predict(X_test)

print("Random Forest with SMOTE:")
print(classification_report(y_test, y_pred_forest_r, digits=3))

