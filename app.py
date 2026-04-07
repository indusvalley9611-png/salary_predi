# main.py
import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Load trained model & encoder
# ----------------------------
model = joblib.load("model.pkl")
le_industry = joblib.load("encoder_industry.pkl")

# ----------------------------
# App title
# ----------------------------
st.title("💰 Salary Prediction App")

# ----------------------------
# User input
# ----------------------------
industry_input = st.text_input("Industry (e.g., IT, Finance, Healthcare)", "IT")
experience_input = st.number_input("Years of Experience", min_value=0, max_value=50, value=2)
skills_input = st.text_input("Skills (space separated, e.g., Python SQL)", "Python")
education_input = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"])

# ----------------------------
# Encode user input
# ----------------------------
try:
    industry_encoded = le_industry.transform([industry_input])[0]
except:
    st.warning("Industry not found in encoder. Using default 'IT'.")
    industry_encoded = le_industry.transform(["IT"])[0]

education_dict = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
education_encoded = education_dict.get(education_input, 0)

skill_count = len(skills_input.split())

# ----------------------------
# Prepare input DataFrame
# ----------------------------
X_input = pd.DataFrame([[industry_encoded, experience_input, skill_count, education_encoded]],
                       columns=["industry_encoded", "experience", "skill_count", "education_encoded"])

# ----------------------------
# Predict salary
# ----------------------------
if st.button("Predict Salary"):
    predicted_salary = model.predict(X_input)[0]
    st.success(f"💵 Predicted Salary: ${predicted_salary:.2f}")
