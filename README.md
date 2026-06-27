
# PlacementIQ — Student Placement Predictor

A Streamlit web app that predicts campus placement outcomes for engineering students using a Gradient Boosting model trained on 1,00,000 students.

## Features
- **Dashboard** — EDA charts (placement by branch, tier, CGPA, salary distribution)
- **Predictor** — real-time probability gauge + salary estimate + radar comparison
- **Model Insights** — feature importance, confusion matrix, model comparison

## Quick start

```bash
# 1. Clone / download this folder
# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app opens at http://localhost:8501

## File structure
```
placement_app/
├── app.py                        # Main Streamlit app
├── clf_model.pkl                 # Trained GradientBoostingClassifier
├── reg_model.pkl                 # Trained GradientBoostingRegressor (salary)
├── le_branch.pkl                 # LabelEncoder for branch
├── le_tier.pkl                   # LabelEncoder for college tier
├── features.pkl                  # Feature column names list
├── student_placement_updated.csv # Dataset (for dashboard charts)
└── requirements.txt
```


## Model performance
| Metric | Score |
|---|---|
| Accuracy | 71.0% |
| ROC-AUC | 77.8% |
| F1 Score | 55.4% |
| Salary MAE | ₹0.98L |

## Dataset
Synthetic dataset of 1,00,000 Indian engineering students with features across academic performance, technical skills, soft skills, and extracurricular activities.
<img width="1920" height="1020" alt="Screenshot 2026-06-27 172911" src="https://github.com/user-attachments/assets/60756a69-6284-46fb-9a4f-bbdf8a98cbc0" />
<img width="1920" height="1020" alt="Screenshot 2026-06-27 174125" src="https://github.com/user-attachments/assets/7e2d6af3-477c-47e2-93ec-497f2f6d780b" />
