import streamlit as st
# import preprocessor,helper
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import xgboost as xg
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

model = pickle.load(open('model.pkl', 'rb'))
encoder = pickle.load(open('target_encoder.pkl', 'rb'))
transformer = pickle.load(open('transformer.pkl', 'rb'))

st.title("Insurance Premium Prediction")

sex = st.selectbox("Please select your gender", ('Male', 'Female'))

age = st.text_input("Please select your age",23)
age = int(age)

bmi = st.text_input("Enter your BMI", 20)
bmi = float(bmi)

children = st.selectbox("Please select number of children", (0,1,2,3,4,5))
children = int (children)

smoker = st.selectbox("Do you smoke",("Yes", "No") )

region = st.selectbox("Please select region",("southwest","northwest", "southeast", "northeast" ))

l = {}
l['age'] = age
l['sex'] = sex
l['bmi'] = bmi
l['children'] = children
l['smoker'] = smoker
l['region'] = region

df = pd.DataFrame(l, index=[0])

df['region'] = encoder.transform(df['region'])
df['sex'] = df['sex'].map({"male": 1, "female": 0})
df['smoker'] = df['smoker'].map({"Yes": 1, "No": 0})

df = transformer.transform(df)
y_pred = model.predict(df)

if st.button("Submit"):
    value = y_pred[0]
    round_value = round(value, 2)
    st.header(f"{round_value} INR")
    # predicted = y_pred[0]
    # value = round(predicted,2)
    # st.header(f"{value} INR")
print(value)
print(round_value)
    