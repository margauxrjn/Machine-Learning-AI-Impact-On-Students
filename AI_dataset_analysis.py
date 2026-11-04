import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

DATA_PATH = Path("ai_student_impact_dataset.csv")

df = pd.read_csv(DATA_PATH)

#Suppression de la colonne Student_ID
df = df.drop(columns=["Student_ID"])
print("\nColumn Student_ID is deleted !")

print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst five rows:")
display(df.head())


continuous_features = ["Pre_Semester_GPA","Weekly_GenAI_Hours","Traditional_Study_Hours","Post_Semester_GPA","Skill_Retention_Score"]
categorical_features = [col for col in df.columns if col not in continuous_features]

print("Continuous features:\n", continuous_features)
print("Categorical features:\n", categorical_features)

#Je t'aime Margaux

print("Dataset information:")
df.info()
print("\nMissing values per column:")
display(df.isna().sum().to_frame("Missing values"))

# 1. Standardisation directe des variables continues (sans imputation)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[continuous_features])

# 2. Recherche du nombre optimal de clusters (k de 2 à 10)
K_range = range(2, 20)
inertias = []
silhouette_scores = []
db_scores = []

for k in K_range:
    print(f"Calcul en cours pour k = {k}...")  # Permet de suivre l'avancement dans la console !
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    inertias.append(kmeans.inertia_)
    # sample_size=2000 évite que Python bloque pendant 5 minutes sur les gros fichiers
    silhouette_scores.append(silhouette_score(X_scaled, labels, sample_size=2000, random_state=42))
    db_scores.append(davies_bouldin_score(X_scaled, labels))

# 3. Affichage des 3 graphiques (Elbow, Silhouette, Davies-Bouldin)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Elbow Method (Inertie)
axes[0].plot(K_range, inertias, marker='o', color='b')
axes[0].set_title("Elbow Method (Inertie)")
axes[0].set_xlabel("Nombre de clusters (k)")
axes[0].set_ylabel("Inertie (WCSS)")
axes[0].grid(True)

# Silhouette Score (Plus c'est HAUT, mieux c'est)
axes[1].plot(K_range, silhouette_scores, marker='s', color='g')
axes[1].set_title("Silhouette Score (Max = Meilleur)")
axes[1].set_xlabel("Nombre de clusters (k)")
axes[1].set_ylabel("Score de Silhouette")
axes[1].grid(True)

# Davies-Bouldin Index (Plus c'est BAS, mieux c'est)
axes[2].plot(K_range, db_scores, marker='^', color='r')
axes[2].set_title("Davies-Bouldin Index (Min = Meilleur)")
axes[2].set_xlabel("Nombre de clusters (k)")
axes[2].set_ylabel("Indice de Davies-Bouldin")
axes[2].grid(True)

plt.tight_layout()
plt.savefig("kmeans_evaluation.png", dpi=300) # Sauvegarde le graphique dans votre dossier projet
print("Graphique sauvegardé sous 'kmeans_evaluation.png' !")
plt.show()

# Tableau récapitulatif des scores
metrics_df = pd.DataFrame({
    "k": list(K_range),
    "Inertia": inertias,
    "Silhouette_Score": silhouette_scores,
    "Davies_Bouldin_Index": db_scores
})
display(metrics_df)

# 4. Entraînement du modèle final et profilage pour l'interprétation
# (Modifiez optimal_k selon les résultats de vos graphiques)
optimal_k = 3

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["Cluster"] = kmeans_final.fit_predict(X_scaled)

print("\nRépartition des étudiants par cluster :")
display(df["Cluster"].value_counts().to_frame("Nombre d'étudiants"))

print("\nProfil moyen de chaque cluster (valeurs réelles pour interprétation) :")
cluster_profile = df.groupby("Cluster")[continuous_features].mean().round(2)
display(cluster_profile)