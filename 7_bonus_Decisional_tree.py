import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = Path("ai_student_impact_dataset.csv")

df = pd.read_csv(DATA_PATH)


target = "Burnout_Risk_Level"
X = df.drop(columns=["Student_ID", target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Taille de X_train :", X_train.shape) # (40000, 14) au lieu de 15
print("Taille de X_test  :", X_test.shape)  # (10000, 14) au lieu de 15

cat_cols = X.select_dtypes(include=["object", "str", "bool"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object", "str", "bool"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

clf_tree = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("tree", DecisionTreeClassifier(max_depth=4, random_state=42))
])

clf_tree.fit(X_train, y_train)

y_pred = clf_tree.predict(X_test)
print("\nAccuracy sur le jeu de test :", round(accuracy_score(y_test, y_pred), 4))
print("\nRapport de classification :\n", classification_report(y_test, y_pred))