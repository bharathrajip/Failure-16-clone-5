
import streamlit as st
import joblib
import pandas as pd

# Load model using joblib
model = joblib.load('disease_prediction_model.pkl')

# Load encoders
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = joblib.load(f)

st.title("Disease Prediction from Symptoms")

symptoms = ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'Symptom_5',
            'Symptom_6', 'Symptom_7', 'Symptom_8', 'Symptom_9', 'Symptom_10',
            'Symptom_11', 'Symptom_12', 'Symptom_13', 'Symptom_14', 'Symptom_15',
            'Symptom_16', 'Symptom_17']

user_input = {}
for symptom in symptoms:
    user_input[symptom] = st.selectbox(f"{symptom.replace('_', ' ')}:", ['None'] + list(label_encoders[symptom].classes_))

input_df = pd.DataFrame([user_input])

# Encode input
for col in input_df.columns:
    try:
        if col in label_encoders:
            input_df[col] = label_encoders[col].transform(input_df[col].astype(str))
    except Exception as e:
        st.error(f"Encoding error in column {col}: {e}")

# Predict
if st.button("Predict Disease"):
    try:
        prediction = model.predict(input_df)
        st.success(f"Predicted Disease: {prediction[0]}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
