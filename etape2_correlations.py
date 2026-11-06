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
plt.figure(figsize=(10, 8))

corr = df[numeric_features].corr(method="spearman")
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True)

plt.title("Spearman")
plt.tight_layout()
plt.savefig(FIG_DIR / "heatmap_correlations.png", dpi=200)
plt.show()

# %%
pairs = df[numeric_features].corr(method="spearman").abs().unstack()
pairs = pairs[pairs.index.get_level_values(0) < pairs.index.get_level_values(1)]
top_pairs = pairs.sort_values(ascending=False).head(6)
top_pairs

# %%
sample_df = df.sample(n=3000, random_state=42)

fig, axes = plt.subplots(2, 2, figsize=(15, 11))

pairs = [
    ("Pre_Semester_GPA", "Post_Semester_GPA", "Pre_Semester_GPA vs Post_Semester_GPA (r = 0.93)"),
    ("Weekly_GenAI_Hours", "Traditional_Study_Hours", "Weekly_GenAI_Hours vs Traditional_Study_Hours (r = -0.16)"),
    ("Weekly_GenAI_Hours", "Skill_Retention_Score", "Weekly_GenAI_Hours vs Skill_Retention_Score (r = -0.12)"),
    ("Traditional_Study_Hours", "Skill_Retention_Score", "Traditional_Study_Hours vs Skill_Retention_Score (r = 0.15)")
]

for ax, (x_col, y_col, title) in zip(axes.flat, pairs):
    sns.regplot(
        data=sample_df, x=x_col, y=y_col, ax=ax,
        scatter_kws={"alpha": 0.25, "s": 15, "color": "#1f77b4"},
        line_kws={"color": "red", "linewidth": 2}
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("scatter_plots.png", dpi=300)
plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Boxplot 1 : Dépendance perçue à l'IA vs Heures hebdomadaires d'IA
sns.boxplot(
    data=df, x="Perceived_AI_Dependency", y="Weekly_GenAI_Hours",
    palette="Blues", ax=axes[0], showfliers=False
)
axes[0].set_title("Weekly_GenAI_Hours par niveau de Perceived_AI_Dependency", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Perceived_AI_Dependency (1 à 10)")
axes[0].set_ylabel("Weekly_GenAI_Hours")
axes[0].grid(True, axis="y", alpha=0.3)

# Boxplot 2 : Niveau d'anxiété aux examens vs Heures hebdomadaires d'IA
sns.boxplot(
    data=df, x="Anxiety_Level_During_Exams", y="Weekly_GenAI_Hours",
    palette="Oranges", ax=axes[1], showfliers=False
)
axes[1].set_title("Weekly_GenAI_Hours par niveau de Anxiety_Level_During_Exams", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Anxiety_Level_During_Exams (1 à 10)")
axes[1].set_ylabel("Weekly_GenAI_Hours")
axes[1].grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("boxplots.png", dpi=300)
plt.show()

# %%
df.assign(Burnout_Code=df["Burnout_Risk_Level"].map({"Low": 0, "Medium": 1, "High": 2}))[
    numeric_features + ["Burnout_Code"]].corr(method="spearman")["Burnout_Code"].drop("Burnout_Code").round(3).sort_values()

#bonjour
