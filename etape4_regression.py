# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

DATA_PATH = Path("ai_student_impact_dataset.csv")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH).drop(columns=["Student_ID"])

target = "Post_Semester_GPA"
simple_features = ["Pre_Semester_GPA"]
multiple_features = [
    "Pre_Semester_GPA",
    "Weekly_GenAI_Hours",
    "Traditional_Study_Hours",
    "Skill_Retention_Score",
]
y = df[target]

# %%
X_simple = df[simple_features]
model_simple = LinearRegression().fit(X_simple, y)
pred_simple = model_simple.predict(X_simple)
mse_simple = mean_squared_error(y, pred_simple)
r2_simple = r2_score(y, pred_simple)

print(f"Intercept : {model_simple.intercept_:.4f}")
print(f"Slope (Pre_Semester_GPA) : {model_simple.coef_[0]:.4f}")
print(f"MSE : {mse_simple:.4f}  |  R2 : {r2_simple:.4f}")

fig, ax = plt.subplots(figsize=(6, 5))
sns.regplot(
    data=df.sample(5000, random_state=0),
    x="Pre_Semester_GPA",
    y=target,
    scatter_kws={"alpha": 0.2, "s": 8},
    line_kws={"color": "red"},
    ax=ax,
)
ax.set_title("Simple regression: Post vs Pre GPA")
plt.tight_layout()
plt.savefig(FIG_DIR / "regression_simple.png", dpi=150)
plt.show()

# %%
X_multiple = df[multiple_features]
model_multiple = LinearRegression().fit(X_multiple, y)
pred_multiple = model_multiple.predict(X_multiple)
mse_multiple = mean_squared_error(y, pred_multiple)
r2_multiple = r2_score(y, pred_multiple)
n, p = X_multiple.shape
r2_adjusted = 1 - (1 - r2_multiple) * (n - 1) / (n - p - 1)

coefficients = pd.DataFrame(
    {
        "coefficient": model_multiple.coef_,
        "standardized": model_multiple.coef_ * X_multiple.std().values / y.std(),
    },
    index=multiple_features,
).round(4)

print(f"Intercept : {model_multiple.intercept_:.4f}")
print(coefficients)
print(f"MSE : {mse_multiple:.4f}  |  R2 : {r2_multiple:.4f}  |  adjusted R2 : {r2_adjusted:.4f}")

# %%
comparison = pd.DataFrame(
    {
        "MSE": [mse_simple, mse_multiple],
        "RMSE": [mse_simple ** 0.5, mse_multiple ** 0.5],
        "R2": [r2_simple, r2_multiple],
    },
    index=["Simple", "Multiple"],
).round(4)
print(comparison)

residuals = y - pred_multiple
sample = df.sample(5000, random_state=0).index

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(pred_multiple[sample], y.loc[sample], alpha=0.2, s=8)
axes[0].plot([1, 4], [1, 4], color="red")
axes[0].set_xlabel("Predicted Post_Semester_GPA")
axes[0].set_ylabel("Actual Post_Semester_GPA")
axes[0].set_title("Multiple regression: predicted vs actual")
sns.histplot(residuals, bins=40, ax=axes[1])
axes[1].set_xlabel("Residual")
axes[1].set_title("Multiple regression: residuals")
plt.tight_layout()
plt.savefig(FIG_DIR / "regression_multiple.png", dpi=150)
plt.show()
