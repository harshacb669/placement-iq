"""
Run this script ONCE to generate the model files on your machine.
It replaces the pre-built .pkl files with ones compatible with
your local scikit-learn version.

Usage:
    python train_models.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import joblib
from pathlib import Path

BASE = Path(__file__).parent

print("Loading dataset...")
df = pd.read_csv(BASE / "student_placement_updated.csv")

# ── Encode categoricals ────────────────────────────────────────────
le_branch = LabelEncoder()
le_tier   = LabelEncoder()
df['branch_enc'] = le_branch.fit_transform(df['branch'])
df['tier_enc']   = le_tier.fit_transform(df['college_tier'])

FEATURES = [
    'branch_enc', 'tier_enc', 'cgpa', 'backlogs', 'coding_skills',
    'dsa_score', 'aptitude_score', 'communication_skills', 'ml_knowledge',
    'system_design', 'internships', 'projects_count', 'certifications',
    'hackathons', 'open_source_contributions', 'extracurriculars'
]

X  = df[FEATURES]
y  = df['placement_status']

# ── Train placement classifier ─────────────────────────────────────
print("Training placement classifier...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
clf.fit(X_train, y_train)

from sklearn.metrics import roc_auc_score, accuracy_score
acc = accuracy_score(y_test, clf.predict(X_test))
auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
print(f"  Accuracy : {acc:.3f}")
print(f"  ROC-AUC  : {auc:.3f}")

# ── Train salary regressor ─────────────────────────────────────────
print("Training salary regressor...")
placed = df[df['placement_status'] == 1].dropna(subset=['salary_package_lpa'])
Xs = placed[FEATURES]
ys = placed['salary_package_lpa']
Xs_train, Xs_test, ys_train, ys_test = train_test_split(
    Xs, ys, test_size=0.2, random_state=42
)
reg = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
reg.fit(Xs_train, ys_train)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(ys_test, reg.predict(Xs_test))
print(f"  Salary MAE: ₹{mae:.2f} LPA")

# ── Save everything ────────────────────────────────────────────────
print("Saving model files...")
joblib.dump(clf,       BASE / "clf_model.pkl")
joblib.dump(reg,       BASE / "reg_model.pkl")
joblib.dump(le_branch, BASE / "le_branch.pkl")
joblib.dump(le_tier,   BASE / "le_tier.pkl")
joblib.dump(FEATURES,  BASE / "features.pkl")

print("\nDone! All model files saved:")
for f in ["clf_model.pkl", "reg_model.pkl", "le_branch.pkl", "le_tier.pkl", "features.pkl"]:
    size = (BASE / f).stat().st_size // 1024
    print(f"  {f}  ({size} KB)")

print("\nNow run:  streamlit run app.py")
