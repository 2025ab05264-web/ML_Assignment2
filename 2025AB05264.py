import joblib
import pandas as pd
import numpy as np
from pandas.core.common import random_state
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import  DecisionTreeClassifier
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.naive_bayes import  GaussianNB
from    sklearn.ensemble import RandomForestClassifier
from streamlit import columns
from xgboost import XGBClassifier

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef)




filePath = "~/PycharmProjects/ML_Assignment2/Data/adult.csv"
df = pd.read_csv(filePath)
print("Details of input dataset are  :")
print(df.info)

df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

le=LabelEncoder()
for col in df.select_dtypes(include=['object']).columns :
    df[col]=le.fit_transform(df[col])

X= df.drop('income', axis=1)
y= df['income']

X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.2, random_state=42)
scaler= StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression" : LogisticRegression(),
    "Decision Tree " : DecisionTreeClassifier(random_state=42),
    "kNN" : KNeighborsClassifier(),
    "Naive Bayes" : GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(random_state=42),
    "XGBoost (Ensemble)" : XGBClassifier(eval_metric='logloss', random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba =model.predict_proba(X_test_scaled) [:,1] if hasattr(model, "predict_proba") else y_pred
    results.append({
        "ML Model Name" : name,
        "Accuracy" : round(accuracy_score(y_test, y_pred), 4),
        "AUC" : round(roc_auc_score(y_test, y_proba), 4),
        "Precision" : round(precision_score(y_test, y_pred), 4),
        "Recall" : round(recall_score(y_test, y_pred), 4),
        "F1" : round(f1_score(y_test, y_pred),4),
        "MCC" : round(matthews_corrcoef(y_test, y_pred),4)
    })
    clean_name = name.replace(" ", "_").lower().replace("(","").replace(")","")
    joblib.dump(model, f'model/{clean_name}.pkl')
    print(f"Saved {name} to model /{name}.pkl")



comparison_df = pd.DataFrame(results)
print("\n---FINAL COMPARISION TABLE---")
print(comparison_df.to_string(index=False))







