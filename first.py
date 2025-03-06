import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load datasets
mental_health_data = pd.read_csv("mental_health_diagnosis_treatment_.csv")
schizophrenia_data = pd.read_csv("schizophrenia_dataset.csv")

# Data Preprocessing
encoder = LabelEncoder()
mental_health_data['Diagnosis'] = encoder.fit_transform(mental_health_data['Diagnosis'])
schizophrenia_data['Tanı'] = encoder.fit_transform(schizophrenia_data['Tanı'])

# Selecting features and target
X_mental = mental_health_data.drop(columns=['Patient ID', 'Diagnosis'])
y_mental = mental_health_data['Diagnosis']

X_schizo = schizophrenia_data.drop(columns=['Hasta_ID', 'Tanı'])
y_schizo = schizophrenia_data['Tanı']

# Scaling data
scaler = StandardScaler()
X_mental = scaler.fit_transform(X_mental)
X_schizo = scaler.fit_transform(X_schizo)

# Splitting datasets
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_mental, y_mental, test_size=0.2, random_state=42)
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_schizo, y_schizo, test_size=0.2, random_state=42)

# Model Selection
models = {
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression()
}

# Train models and evaluate
results = {}
for name, model in models.items():
    model.fit(X_train_m, y_train_m)
    predictions = model.predict(X_test_m)
    accuracy = accuracy_score(y_test_m, predictions)
    results[name] = accuracy

# Streamlit Frontend
st.title("AI-Powered Mental Illness Diagnosis System")
st.write("Select a machine learning model and input patient data for diagnosis.")

selected_model = st.selectbox("Choose a Model", list(models.keys()))

# User input fields
user_input = []
for col in mental_health_data.columns[1:-1]:
    user_input.append(st.number_input(f"{col}", value=0))

# Predict button
if st.button("Diagnose"):
    model = models[selected_model]
    user_input = np.array(user_input).reshape(1, -1)
    user_input = scaler.transform(user_input)
    diagnosis = model.predict(user_input)
    diagnosis_label = encoder.inverse_transform(diagnosis)
    st.success(f"Predicted Diagnosis: {diagnosis_label[0]}")

st.write("Model Accuracy:")
st.json(results)


