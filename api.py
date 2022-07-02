import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Loading up the trained model
Mmodel = pickle.load(open('Mclassifier.pkl', 'rb'))
Gmodel = pickle.load(open('Gclassifier.pkl','rb'))

st.title("Welcome to MMSE and GDS-15 Prediction")
st.write("""## We need some information to predict the MMSE and GDS-15""")

st.markdown("""
        <style>
        .css-15zrgzn {display: none}
        .css-eczf16 {display: none}
        .css-jn99sy {display: none}
        </style>
        """, unsafe_allow_html=True)

#demographic features

st.write("""### Demographic status form""")

age = st.number_input("Age",1,99)

gender = {1:"Female", 2:"Male"}
gender_opt = st.selectbox("Select gender:", gender.keys(), format_func=lambda x:gender[ x ])

state = {1:"Johor", 2:"Perak", 3: "Kelantan", 4:"Selangor", 5:"Wilayah Persekutuan"}
state_opt = st.selectbox("Select state:", state.keys(), format_func=lambda x:state[ x ])

tm_income = st.number_input("Total monthly income (main + household + side)")

job = {1:"Manager", 2:"Professional", 3: "Technicians", 4:"Clerical", 5:"Sales and Service", 6:"Jobs Skilled", 7:"Craft and Trades", 8:"Plant and Machine Operators", 9:"Basic Jobs", 10:"Military"}
job_opt = st.selectbox("Select job:", job.keys(), format_func=lambda x:job[ x ])

psector = {1:"Public Sector", 2:"NGO", 3: "Private Sector", 4:"Self"}
psector_opt = st.selectbox("Select previous job sector:", psector.keys(), format_func=lambda x:psector[ x ])

marital = {1:"Bujang", 2:"Berkahwin", 3: "Bercerai", 4:"Balu/Duda"}
marital_opt = st.selectbox("Select marital status:", marital.keys(), format_func=lambda x:marital[ x ])

living = {1:"Sendirian", 2:"Bersama Orang Lain"}
living_opt = st.selectbox("Select living status:", living.keys(), format_func=lambda x:living[ x ])

#psychology features
st.write("""### Psychology status form""")

qol = st.slider("Scale of Quality of Life (1: Very satisfied  to   4: Not satisfied at all)", 1,4)

swls = st.slider("Scale of Satisfaction with Life (0: Disagree to   2: Agree)", 0,2)

epq = {1:"Yes", 0:"No"}
epq_opt = st.selectbox("Eysenck Personality Questionnaire EPQ :", epq.keys(), format_func=lambda x:epq[ x ])

loneliness = st.slider("Scale of Loneliness (1: Hardly ever  to   3: Often)", 1,3)

submit = st.button("Predict")

if submit:
    X = np.array([[age,gender_opt,state_opt,tm_income,job_opt,psector_opt,marital_opt,living_opt,qol,swls,epq_opt,loneliness]])
    MMSE = Mmodel.predict(X).tolist()[0]
    GDS = Gmodel.predict(X).tolist()[0]

    MMSEprediction = MMSE
    if MMSEprediction == 0: st.error("MMSE result is Severe")
    elif MMSEprediction == 1: st.warning("MMSE result is Mild")
    elif MMSEprediction == 2: st.success("MMSE result is Normal")
    else: st.write("Impossible") 
    
    GDSprediction = GDS
    if GDSprediction == 2: st.error("GDS result is Depression")
    else: st.success("GDS result is Normal") 
  
