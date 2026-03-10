import os
import joblib
import streamlit as st

# Absolute path to the model
model_path = r"C:\Users\Yashashwini\Desktop\student_performance\model\student_model.pkl"

# Load trained model
model = joblib.load(model_path)

# Streamlit UI
st.title("🎓 Student Performance Prediction")
st.write("Enter student details to predict percentage")

study_hours = st.number_input("Study Hours", min_value=0.0, max_value=24.0)
attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0)
assignments = st.number_input("Assignments Completed", min_value=0.0, max_value=20.0)
previous_marks = st.number_input("Previous Marks", min_value=0.0, max_value=100.0)

if st.button("Predict Percentage"):
    prediction = model.predict([[study_hours, attendance, assignments, previous_marks]])
    st.success(f"Predicted Percentage: {round(prediction[0],2)}%")