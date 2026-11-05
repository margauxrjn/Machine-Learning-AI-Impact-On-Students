Prompt 1 : 

Ok Claude, ton Rôle : 
Tu es relecteur technique et statistique. Ta mission est UNIQUEMENT de VÉRIFIER (double check) du code et des conclusions déjà écrits. Tu ne génères rien de nouveau.
Règles strictes

* Interdit : écrire un nouveau script, ajouter une analyse, un graphique, un modèle ou une étape qui n'existe pas déjà.
* Interdit : modifier, réécrire ou "améliorer" les fichiers du projet. Lecture seule. Tu peux exécuter le code existant et lancer des calculs de contrôle jetables (dans un répertoire temporaire, jamais dans le repo) uniquement pour confirmer ou infirmer un chiffre.
* Interdit : toucher à git (pas de commit, pas de push) et au Google Doc.
* Interdit : traiter les étapes suivantes (K-means, AHC, régression, KNN). Elles ne font pas partie de cette vérification.
* Autorisé : signaler un problème, citer la ligne fautive, expliquer pourquoi, et proposer en une phrase le correctif attendu SANS l'appliquer.
* Les fichiers ne contiennent volontairement aucun commentaire de code. Ce n'est pas un défaut à signaler.

Contexte du projet
Projet de Machine Learning en groupe, avec notebook, PPT et soutenance orale. Jeu de données Kaggle "AI impact on students" (`laveshjadon/ai-impact-on-students`) : 50 000 étudiants, 16 colonnes, fichier `ai_student_impact_dataset.csv`. Colonnes : Student_ID (identifiant, retiré), Major_Category, Year_of_Study, Pre_Semester_GPA, Post_Semester_GPA, Weekly_GenAI_Hours, Traditional_Study_Hours, Skill_Retention_Score, Tool_Diversity, Primary_Use_Case, Prompt_Engineering_Skill, Perceived_AI_Dependency, Paid_Subscription, Institutional_Policy, Anxiety_Level_During_Exams, Burnout_Risk_Level.
Les modèles prévus plus tard : K-means et classification hiérarchique (sur les variables continues), régression linéaire simple et multiple (OLS, MSE, R²), KNN pour classifier Burnout_Risk_Level. Les étapes 2 et 3 doivent donc préparer ces choix (variables continues, cible catégorielle, colinéarités, valeurs plafonnées).
Consigne officielle du PDF (Projet_ML_2026_EN.pdf), section "Data Analysis"

1. Dataset Description : start by describing the dataset: number of rows and columns; types of variables (continuous, categorical).
2. Data Cleaning and Exploration :
   1. Detect and handle missing or outlier values.
   2. Perform a descriptive analysis of continuous and categorical variables (mean, median, standard deviation, histograms).
   3. Visualize correlations between variables.

Autres consignes générales : tout le projet se fait en groupe, toutes les questions doivent être traitées, les sujets hors cours donnent des points bonus. Livrables : un notebook (nom/prénom, code Python, analyses descriptives et graphiques, résultats et interprétations, métriques et conclusion) et un PPT.
Le périmètre de CETTE vérification est uniquement les points 2.2 (analyse descriptive) et 2.3 (corrélations).
Fichiers à vérifier (à la racine du repo)

* `etape2_analyse_descriptive.py` : étape 2.2 (statistiques, histogrammes, barres, tableaux d'effectifs).
* `etape3_correlations.py` : étape 2.3 (heatmaps Pearson et Spearman, nuages de points, boxplots par niveau de burnout).
* `figures/` : les 6 images générées par ces scripts.
* Le texte rédigé dans le Google Doc "AI on students" (onglet "Outline + Code"), sections "Descriptive analysis" et "Correlations", est recopié ci-dessous pour que tu le contrôles contre les données.

Typage retenu des variables (à valider)

* Continues : Pre_Semester_GPA, Post_Semester_GPA, Weekly_GenAI_Hours, Traditional_Study_Hours, Skill_Retention_Score.
* Ordinales (échelles entières) : Tool_Diversity (1 à 5), Perceived_AI_Dependency (1 à 10), Anxiety_Level_During_Exams (1 à 10).
* Catégorielles : Major_Category, Year_of_Study, Primary_Use_Case, Prompt_Engineering_Skill, Paid_Subscription, Institutional_Policy, Burnout_Risk_Level.
* Décision prise : les valeurs aux bornes de l'échelle (GPA 4.0, rétention 100, GenAI 40 h, étude 1 h) sont des valeurs plafonnées, pas des erreurs, donc conservées. Limite annoncée : elles peuvent biaiser la régression linéaire et le K-means.

Ce que tu dois vérifier
A. Exactitude du code

1. Les deux scripts s'exécutent-ils de bout en bout sans erreur, depuis un dossier contenant uniquement le CSV ? Note toute erreur, tout warning, toute dépendance non listée.
2. Les chemins sont-ils corrects (CSV à la racine, figures enregistrées dans `figures/`) ? Le dossier `figures/` est-il créé s'il manque ?
3. `Student_ID` est-il bien retiré avant toute statistique, matrice de corrélation ou graphique ?
4. Les listes de variables (continues, ordinales, catégorielles) couvrent-elles exactement les 15 colonnes utiles, sans doublon ni oubli ? Les noms de colonnes correspondent-ils au CSV (ex. `Strict_Ban`, pas `Strictly_Ban`) ?
5. Étape 2.2 : le tableau (min, moyenne, max, médiane, écart-type) est-il calculé sur les bonnes colonnes ? Les histogrammes affichent-ils la bonne variable, la bonne moyenne et la bonne médiane ? Le dernier sous-graphique vide est-il bien masqué ?
6. Étape 2.3 : les matrices Pearson et Spearman utilisent-elles les 8 variables numériques voulues ? Le tri des paires par corrélation absolue est-il correct, sans doublons ni diagonale ? Le filtre `get_level_values(0) < get_level_values(1)` garde-t-il bien chaque paire une seule fois ?
7. Le codage du burnout (Low=0, Medium=1, High=2) est-il cohérent avec l'ordre naturel, et la corrélation de Spearman avec ce code est-elle bien calculée ?
8. L'échantillon de 5 000 lignes pour les nuages de points est-il reproductible (`random_state`) et ne fausse-t-il pas les droites de régression affichées ?
9. Les titres, axes, légendes et ordres de catégories des graphiques sont-ils justes et lisibles ?
10. Repère tout bug subtil : mauvais axe, ordre des classes, variable ordinale traitée à tort comme continue, graphique qui affiche autre chose que son titre.

B. Conformité à la consigne du PDF

11. Le point 2.2 est-il entièrement couvert : moyenne, médiane, écart-type, histogrammes pour les continues ET analyse descriptive des catégorielles ?
12. Le point 2.3 est-il couvert : visualisation des corrélations entre variables (heatmap suffisante, ou manque-t-il quelque chose d'attendu) ?
13. Reste-t-il un point de la consigne non traité dans ces deux étapes ? Si oui, précise lequel, sans le produire.
14. Le point 2.1 (valeurs manquantes et aberrantes) est hors de ces deux scripts, mais signale toute incohérence entre ce que dit le Doc ("No missing value", "No absurd value") et ce que les données montrent (valeurs plafonnées).

C. Exactitude des chiffres et des interprétations
Recalcule chaque chiffre ci-dessous à partir du CSV et indique pour chacun : confirmé, ou écart (valeur réelle).
Étape 2.2 :

* Weekly_GenAI_Hours : moyenne 8.43, médiane 5.80, très asymétrique à droite.
* Pre_Semester_GPA moyenne 3.15, Post_Semester_GPA moyenne 3.35 ; les deux sont asymétriques à gauche.
* Traditional_Study_Hours (moyenne 11.21, médiane 11.18) et Skill_Retention_Score (moyenne 75.80, médiane 76.00) : à peu près symétriques.
* Pics aux bornes : GPA 4.0, rétention 100, GenAI 40 h, étude 1 h.
* Tool_Diversity : moyenne 2.80, mode 3. Perceived_AI_Dependency : moyenne 3.51. Anxiety_Level_During_Exams : moyenne 4.27.
* Major_Category : STEM 30 %, Business 25 %, Humanities 20 %, Medical 13 %, Arts 12 %.
* Paid_Subscription : 42 % payants. Institutional_Policy : Allowed_With_Citation 50 %, Actively_Encouraged 30 %, Strict_Ban 20 %.
* Burnout_Risk_Level : Medium 42 %, Low 33 %, High 25 %.

Étape 2.3 (Pearson sauf mention) :

* Pre_Semester_GPA vs Post_Semester_GPA : r = 0.93.
* Weekly_GenAI_Hours vs Perceived_AI_Dependency : r = 0.67 (Spearman 0.56) ; vs Anxiety_Level_During_Exams : r = 0.27 ; anxiété vs dépendance : r = 0.31.
* Weekly_GenAI_Hours vs Post_Semester_GPA : r = -0.02.
* Post_Semester_GPA vs Skill_Retention_Score : 0.17 ; vs Traditional_Study_Hours : 0.14. Tool_Diversity vs Skill_Retention_Score : 0.20.
* Moyenne de Weekly_GenAI_Hours par niveau de burnout : Low 4.6 h, Medium 7.4 h, High 15.2 h.
* Spearman avec le code burnout : GenAI hours 0.44, dépendance 0.34, anxiété 0.15, Traditional_Study_Hours -0.13, Tool_Diversity 0.01.

Texte du Doc à contrôler contre les chiffres ci-dessus :

* "Weekly_GenAI_Hours is strongly right-skewed (mean 8.43 > median 5.80): most students use AI a few hours a week, a minority uses it heavily."
* "Capped values ... they are values capped to the scale limits, not errors, so they are kept. Limit: they may bias linear regression and K-means."
* "Pre_Semester_GPA and Post_Semester_GPA are almost collinear (r = 0.93), a point to handle in the regression."
* "Weekly_GenAI_Hours is not correlated with Post_Semester_GPA (r = -0.02): GenAI usage alone does not explain the grade."
* "These are the most useful variables for the KNN; Tool_Diversity (0.01) is not."

Pour chaque phrase d'interprétation, dis si elle est justifiée par les chiffres, exagérée, ou fausse. Vérifie en particulier :

* les adjectifs ("strongly", "moderately", "almost collinear") sont-ils proportionnés aux valeurs ?
* une corrélation est-elle présentée comme une causalité ?
* le choix de "valeurs plafonnées conservées" est-il défendable face à un jury, et la limite annoncée est-elle correcte ?
* Pearson est-il approprié pour des variables ordinales à peu de modalités, ou Spearman suffit-il ?

D. Risques pour la suite (signaler seulement)

15. Les résultats des étapes 2 et 3 créent-ils un risque pour K-means, la régression ou le KNN qui n'est pas mentionné (colinéarité GPA, échelles différentes, classes déséquilibrées, plafonnement) ?

Format de la réponse attendue

1. Verdict global en 2 lignes : le code est-il correct et conforme à la consigne ? oui, non, ou oui avec réserves.
2. Tableau des chiffres : valeur annoncée, valeur recalculée, statut (OK ou écart).
3. Liste numérotée des problèmes trouvés, classés par gravité (bloquant, important, mineur), chacun avec : fichier et ligne, ce qui ne va pas, pourquoi, et le correctif attendu en une phrase (non appliqué).
4. Liste des points vérifiés et corrects.
5. Ce que tu n'as pas pu vérifier, et pourquoi.

Ne produis rien d'autre : pas de nouveau code, pas de réécriture, pas d'étapes suivantes.


Prompt 2 :
