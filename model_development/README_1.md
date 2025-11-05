# DEPI_DS - Employee Attrition Model (Final)



## Files included
- best_final_stacking_model.pkl : Saved StackingClassifier (includes base pipelines)
- train_columns.pkl : list of training feature names (for get_dummies + reindex)
- predictions_from_saved_model.csv : predictions on HR-Employee.csv
- model_developmentNEW.ipynb : Notebook with preprocessing, training, evaluation, SHAP
- README.md : this file
- requirements.txt : (pandas
numpy
matplotlib
seaborn
xgboost
scikit-learn
imblearn
joblib
shap
)
 

## How to test locally
1. Install requirements:
   pip install -r requirements.txt

2. Example Python to load model and predict (run in same folder):
```python
import joblib, pandas as pd
model = joblib.load("best_final_stacking_model.pkl")
df = pd.read_csv("HR-Employee.csv")   # or other CSV
y_true = None
if 'Attrition' in df.columns:
    y_true = df['Attrition'].map({'Yes':1,'No':0})
    X = df.drop(columns=['Attrition'])
else:
    X = df.copy()



# if model fails due to feature names, do this:
train_cols = joblib.load("train_columns.pkl")
X_enc = pd.get_dummies(X, drop_first=True)
X_enc = X_enc.reindex(columns=train_cols, fill_value=0)

y_pred = model.predict(X_enc)
y_proba = None
try:
    y_proba = model.predict_proba(X_enc)[:,1]
except:
    pass

print("Sample preds:", y_pred[:10])
