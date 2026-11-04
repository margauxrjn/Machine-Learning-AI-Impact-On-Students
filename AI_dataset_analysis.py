import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from IPython.display import display
DATA_PATH = Path("ai_student_impact_dataset.csv")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst five rows:")
display(df.head())


continuous_features = ["Pre_Semester_GPA","Weekly_GenAI_Hours","Traditional_Study_Hours","Post_Semester_GPA","Skill_Retention_Score"]
categorical_features = [col for col in df.columns if col not in continuous_features]

print("Continuous features:\n", continuous_features)
print("\n")
print("Categorical features:\n", categorical_features)

#kdKQsjdlqksjd

