# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_PATH = Path("ai_student_impact_dataset.csv")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH).drop(columns=["Student_ID"])

numeric_features = ["Pre_Semester_GPA", "Post_Semester_GPA", "Weekly_GenAI_Hours",
                    "Traditional_Study_Hours", "Skill_Retention_Score",
                    "Tool_Diversity", "Perceived_AI_Dependency", "Anxiety_Level_During_Exams"]
burnout_order = ["Low", "Medium", "High"]

# %%
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
for ax, method in zip(axes, ["pearson", "spearman"]):
    corr = df[numeric_features].corr(method=method)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True, ax=ax)
    ax.set_title(method.capitalize())
plt.tight_layout()
plt.savefig(FIG_DIR / "heatmap_correlations.png", dpi=200)
plt.show()

# %%
pairs = df[numeric_features].corr(method="spearman").abs().unstack()
pairs = pairs[pairs.index.get_level_values(0) < pairs.index.get_level_values(1)]
top_pairs = pairs.sort_values(ascending=False).head(6)
top_pairs

# %%
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, (a, b) in zip(axes.ravel(), top_pairs.index):
    sns.regplot(data=df.sample(5000, random_state=0), x=a, y=b, ax=ax,
                scatter_kws={"alpha": 0.2, "s": 8}, line_kws={"color": "red"})
    ax.set_title(f"{a} vs {b}")
plt.tight_layout()
plt.savefig(FIG_DIR / "scatter_top_pairs.png", dpi=200)
plt.show()

# %%
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
for ax, col in zip(axes.ravel(), numeric_features):
    sns.boxplot(data=df, x="Burnout_Risk_Level", y=col, order=burnout_order, ax=ax)
    ax.set_title(col)
plt.tight_layout()
plt.savefig(FIG_DIR / "boxplot_burnout.png", dpi=200)
plt.show()

# %%
df.assign(Burnout_Code=df["Burnout_Risk_Level"].map({"Low": 0, "Medium": 1, "High": 2}))[
    numeric_features + ["Burnout_Code"]].corr(method="spearman")["Burnout_Code"].drop("Burnout_Code").round(3).sort_values()

#bonjour