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

X = df.drop(columns=["Student_ID"])
target = "Burnout_Risk_Level"
y = df[target]


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Taille de X_train :", X_train.shape)
print("Taille de X_test  :", X_test.shape)


cat_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object", "bool"]).columns.tolist()

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

y_pred = clf_tree.predict(X_test)
print("\nAccuracy sur le jeu de test :", round(accuracy_score(y_test, y_pred), 4))
print("\nRapport de classification :\n", classification_report(y_test, y_pred))
