import streamlit as st
import pickle
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Disease Prediction", page_icon="🩺", layout="wide")

def add_custom_css():
    st.markdown(
        """
        <style>
            body {
                font-family: 'Arial', sans-serif;
                background-color: #f4f4f4;
            }
            .main {
                background: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
            }
            h1 {
                text-align: center;
                color: #333;
            }
            .stButton>button {
                background-color: #007BFF;
                color: white;
                padding: 10px;
                border-radius: 10px;
            }
            .stButton>button:hover {
                background-color: #0056b3;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

add_custom_css()

st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

models = {
    'diabetes': pickle.load(open('Models/diabetes_model.sav', 'rb')),
    'heart_disease': pickle.load(open('Models/heart_disease_model.sav', 'rb')),
    'parkinsons': pickle.load(open('Models/parkinsons_model.sav', 'rb')),
    'lung_cancer': pickle.load(open('Models/lungs_disease_model.sav', 'rb'))
}

selected = option_menu(
    menu_title="Disease Prediction",
    options=[
        "Diabetes Prediction",
        "Heart Disease Prediction",
        "Parkinson's Prediction",
        "Lung Cancer Prediction"
    ],
    icons=["clipboard", "heart", "activity", "lungs"],
    menu_icon="stethoscope",
    default_index=0,
    orientation="horizontal"
)

def display_input(label, key, type="number"):
    return st.number_input(label, key=key, step=1) if type == "number" else st.text_input(label, key=key)

st.markdown("<div class='main'>", unsafe_allow_html=True)

if selected == "Diabetes Prediction":
    st.title("🩸 Diabetes Prediction")
    inputs = [display_input(label, label) for label in [
        'Number of Pregnancies', 'Glucose Level', 'Blood Pressure', 'Skin Thickness', 
        'Insulin Level', 'BMI', 'Diabetes Pedigree Function', 'Age'
    ]]
    
    if st.button("Predict Diabetes"):
        result = models['diabetes'].predict([inputs])
        st.success("Diabetic" if result[0] == 1 else "Not Diabetic")

elif selected == "Heart Disease Prediction":
    st.title("❤️ Heart Disease Prediction")
    inputs = [display_input(label, label) for label in [
        'Age', 'Sex (1 = Male, 0 = Female)', 'Chest Pain Type (0-3)', 'Resting Blood Pressure',
        'Cholesterol', 'Fasting Blood Sugar > 120 (1 = Yes, 0 = No)', 'Resting ECG (0-2)',
        'Max Heart Rate', 'Exercise Induced Angina (1 = Yes, 0 = No)', 'ST Depression',
        'Slope of ST (0-2)', 'Number of Major Vessels (0-3)', 'Thal (0=Normal, 1=Fixed Defect, 2=Reversible)'
    ]]
    
    if st.button("Predict Heart Disease"):
        result = models['heart_disease'].predict([inputs])
        st.success("Has Heart Disease" if result[0] == 1 else "No Heart Disease")

elif selected == "Parkinson's Prediction":
    st.title("🧠 Parkinson's Disease Prediction")
    inputs = [display_input(label, label) for label in [
        'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 
        'MDVP:RAP', 'MDVP:APQ', 'HNR', 'RPDE', 'DFA', 'Spread1', 'D2', 'PPE'
    ]]

    if st.button("Predict Parkinson's"):
        result = models['parkinsons'].predict([inputs])
        st.success("Has Parkinson's" if result[0] == 1 else "No Parkinson's")

elif selected == "Lung Cancer Prediction":
    st.title("🫁 Lung Cancer Prediction")
    inputs = [display_input(label, label) for label in [
        'Gender (1=Male, 0=Female)', 'Age', 'Smoking', 'Yellow Fingers', 'Anxiety', 
        'Peer Pressure', 'Chronic Disease', 'Fatigue', 'Allergy', 'Wheezing', 
        'Alcohol Consuming', 'Coughing', 'Shortness of Breath', 'Swallowing Difficulty', 'Chest Pain'
    ]]

    if st.button("Lung Cancer Test Result"):
        result = models['lung_cancer'].predict([inputs])
        st.success("The person has lung cancer" if result[0] == 1 else "No lung cancer")

st.markdown("</div>", unsafe_allow_html=True)
