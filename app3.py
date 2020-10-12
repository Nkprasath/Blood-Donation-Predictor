import pandas as pd
import numpy  as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import streamlit as st


df = pd.read_csv("transfusion.data")
Y = df["whether he/she donated blood in March 2007"]
X = df.drop(["whether he/she donated blood in March 2007"], axis = 1)

st.write("""
# Blood Donation Prediction
This app predicts if a person is going to donate blood Every 3 months
""")

st.sidebar.header('User Input Parameters')

uploaded_file = st.sidebar.file_uploader("Upload your file here:", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input():
        Recent = st.sidebar.slider('Recency(months)',0,74,0)
        Frq = st.sidebar.slider('Frequency (times)',1 ,50 ,1)
        Blood = st.sidebar.slider('Monetary (c.c. blood)', 250, 12500, 250)
        Times = st.sidebar.slider('Time (months)', 2, 98, 5)
        data = {
        'Recency(months)' : Recent,
        'Frequency (times)' : Frq,
        'Monetary (c.c. blood)' : Blood,
        'Time (months)' : Times,
        }
        features = pd.DataFrame(data, index=[0])
        return features               
    df_user = user_input()


X_train, X_test, y_train, y_test = \
train_test_split(X, Y)



if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below)')
    st.write(df_user)

rf = RandomForestClassifier\
(n_estimators=100, max_depth=3)
rf.fit(X_train, y_train)

pred = rf.predict(df_user)
pred_proba =rf.predict_proba(df_user)

st.write("Prediction")
if pred == 1:
    st.write("This person will donate blood")
else:
    st.write("This person will not donate blood")


