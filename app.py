import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix)

# --- App Title and Student Info ---
st.title("💰 Adult Income Prediction - ML Assignment 2")
st.write("M.Tech (AIML/DSE) - BITS Pilani")

# --- Step 5(a): Dataset Upload Option [cite: 91] ---
st.sidebar.header("1. Upload Test Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

def preprocess_data(df):
    # Data Cleaning: Remove missing values
    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # Feature Engineering: Label Encoding
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])
    return df

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = preprocess_data(df)
    
    st.write("### Dataset Preview (Cleaned)")
    st.dataframe(df.head())

    # Features and Target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # 80/20 Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Step 5(b): Model Selection Dropdown [cite: 92] ---
    st.sidebar.header("2. Choose ML Model")
    model_option = st.sidebar.selectbox(
        "Select one of the 6 required models",
        ("Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest", "XGBoost")
    )

    # Define Models [cite: 34-39]
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "kNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(eval_metric='logloss')
    }

    # Train and Predict
    model = models[model_option]
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # --- Step 5(c): Display Evaluation Metrics [cite: 93] ---
    st.write(f"## Evaluation Metrics for {model_option}")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
    
    # AUC calculation
    y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred
    col2.metric("AUC Score", f"{roc_auc_score(y_test, y_proba):.4f}")
    
    col3.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
    
    col4, col5, col6 = st.columns(3)
    col4.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
    col5.metric("F1 Score", f"{f1_score(y_test, y_pred):.4f}")
    col6.metric("MCC Score", f"{matthews_corrcoef(y_test, y_pred):.4f}")

    # --- Step 5(d): Confusion Matrix [cite: 94] ---
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)

else:
    st.info("Please upload a CSV file from the sidebar to begin.")
