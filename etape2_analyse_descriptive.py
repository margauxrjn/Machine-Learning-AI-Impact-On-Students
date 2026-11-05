# %%
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path("ai_student_impact_dataset.csv")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH).drop(columns=["Student_ID"])

continuous_features = ["Pre_Semester_GPA", "Post_Semester_GPA", "Weekly_GenAI_Hours",
                       "Traditional_Study_Hours", "Skill_Retention_Score"]
ordinal_features = ["Tool_Diversity", "Perceived_AI_Dependency", "Anxiety_Level_During_Exams"]
categorical_features = ["Major_Category", "Year_of_Study", "Primary_Use_Case",
                        "Prompt_Engineering_Skill", "Paid_Subscription",
                        "Institutional_Policy", "Burnout_Risk_Level"]

# %%
stats = df[continuous_features + ordinal_features].agg(["min", "mean", "max", "median", "std"]).T.round(3)
stats.columns = ["Min", "Mean", "Max", "Median", "Std"]
stats

# %%
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for ax, col in zip(axes.ravel(), continuous_features):
    ax.hist(df[col], bins=40, color="steelblue", edgecolor="white")
    ax.axvline(df[col].mean(), color="red", linestyle="--", label=f"mean = {df[col].mean():.2f}")
    ax.axvline(df[col].median(), color="orange", linestyle="-", label=f"median = {df[col].median():.2f}")
    ax.set_title(col)
    ax.set_ylabel("Students")
    ax.legend()
axes.ravel()[-1].axis("off")
plt.tight_layout()
plt.savefig(FIG_DIR / "hist_continuous.png", dpi=200)
plt.show()

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ordinal_features):
    df[col].value_counts().sort_index().plot.bar(ax=ax, color="seagreen")
    ax.set_title(col)
    ax.set_ylabel("Students")
plt.tight_layout()
plt.savefig(FIG_DIR / "bar_ordinal.png", dpi=200)
plt.show()

# %%
for col in categorical_features:
    counts = df[col].value_counts()
    table = pd.DataFrame({"Count": counts, "Percent": (counts / len(df) * 100).round(1)})
    print(f"\n{col}")
    print(table)

# %%
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
for ax, col in zip(axes.ravel(), categorical_features):
    df[col].value_counts().plot.bar(ax=ax, color="indianred")
    ax.set_title(col)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
axes.ravel()[-1].axis("off")
plt.tight_layout()
plt.savefig(FIG_DIR / "bar_categorical.png", dpi=200)
plt.show()
