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


print("Dataset information:")
df.info()
print("\nMissing values per column:")
display(df.isna().sum().to_frame("Missing values"))


scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[continuous_features])


K_range = range(2, 20)
inertias = []
silhouette_scores = []
db_scores = []

for k in K_range:
    print(f"Computing k = {k}...")
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto')
    labels = kmeans.fit_predict(X_scaled)

    inertias.append(kmeans.inertia_)

    silhouette_scores.append(silhouette_score(X_scaled, labels, sample_size=2000, random_state=None))
    db_scores.append(davies_bouldin_score(X_scaled, labels))


fig, axes = plt.subplots(1, 3, figsize=(18, 5))


axes[0].plot(K_range, inertias, marker='o', color='b')
axes[0].set_title("Elbow Method (Within-cluster inertia)")
axes[0].set_xlabel("Number of clusters (k)")
axes[0].set_ylabel("Within cluster (Sum of squares)")
axes[0].grid(True)


axes[1].plot(K_range, silhouette_scores, marker='s', color='g')
axes[1].set_title("Silhouette Score (Max = Best)")
axes[1].set_xlabel("Number of clusters (k)")
axes[1].set_ylabel("Silhouette score")
axes[1].grid(True)


axes[2].plot(K_range, db_scores, marker='^', color='r')
axes[2].set_title("Davies-Bouldin Index (Min = Best)")
axes[2].set_xlabel("Number of clusters (k)")
axes[2].set_ylabel("Davies-Bouldin Index")
axes[2].grid(True)

plt.tight_layout()
plt.savefig("kmeans_evaluation.png", dpi=300)
print("Graph saved as 'kmeans_evaluation.png' !")
plt.show()


metrics_df = pd.DataFrame({
    "k": list(K_range),
    "Within-cluster inertia": inertias,
    "Silhouette_Score": silhouette_scores,
    "Davies_Bouldin_Index": db_scores
})
display(metrics_df)


optimal_k = 3

kmeans_final = KMeans(n_clusters=optimal_k, random_state=None, n_init='auto')
df["Cluster"] = kmeans_final.fit_predict(X_scaled)

print("\nStudents repartition per cluster :")
display(df["Cluster"].value_counts().to_frame("Number of students"))

print("\nCluster global profile :")
cluster_profile = df.groupby("Cluster")[continuous_features].mean().round(2)
display(cluster_profile)