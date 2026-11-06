# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("ai_student_impact_dataset.csv")
df = pd.read_csv(DATA_PATH)


df = df.drop(columns=["Student_ID"])
print("\nColumn Student_ID is deleted !")


target_col = "Burnout_Risk_Level"
y = df[target_col]

cols_to_drop = [target_col]
X = df.drop(columns=cols_to_drop)


X_encoded = pd.get_dummies(X,drop_first=True)


X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42,stratify=y)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set size: {X_train_scaled.shape}")
print(f"Test set size: {X_test_scaled.shape}")
print("\n")
print("Training characteristics:\n", X_train_scaled[:105])
print("Training labels:\n", y_train[:105])
print("Test characteristics:\n", X_train_scaled[:45])
print("Test labels:\n", y_train[:45])

#--------------------------

k_values = list(range(1, 91, 2))
accuracy_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    accuracy_scores.append(accuracy_score(y_test, y_pred))


best_k = k_values[np.argmax(accuracy_scores)]
best_acc = max(accuracy_scores)
print(f"\nBest k: {best_k} with accuracy: {best_acc * 100:.2f}%")


plt.figure(figsize=(9, 5), dpi=150)
plt.plot(k_values, accuracy_scores, marker='o', color='tab:blue', linewidth=2)
plt.axvline(best_k, color='tab:red', linestyle='--', label=f'Best k={best_k}')
plt.title("KNN Accuracy vs. k (Number of Neighbors)", fontsize=13)
plt.xlabel("k Value", fontsize=11)
plt.ylabel("Test Accuracy", fontsize=11)
plt.xticks(k_values[::2])  # Show every second tick for readability
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()


final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train_scaled, y_train)
final_pred = final_knn.predict(X_test_scaled)

print(f"Final Model Test Accuracy: {accuracy_score(y_test, final_pred) * 100:.2f}%")


y_pred = knn.predict(X_test)
print("Predicted labels:\n", y_pred)
print("Real labels:\n", y_test)


print("\nAccuracy :", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification report :\n", classification_report(y_test, y_pred))