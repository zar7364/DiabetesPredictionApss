import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Prediksi Diabetes App
oleh : Nezar Abdilah Prakasa

This Apps using Diabetes Dataset Open Data
""")

st.sidebar.header('Masukan datamu!')


# Collects user input features into dataframe
def user_input_features():
    Glucose = st.sidebar.slider('Kadar Glukosa', 0.00,199.00,121.18)
    BloodPressure = st.sidebar.slider('Tekanan Darah', 0.00,122.00,69.14)
    SkinThickness = st.sidebar.slider('Ketebalan Kulit', 0.00,110.00,20.93)
    Insulin = st.sidebar.slider('Kadar Insulin', 0.00,744.00,80.25)
    BMI=st.sidebar.slider('BMI', 0.00,80.60,32.19)
    Age=st.sidebar.slider('Usia',21.00,81.00,33.09)
    data = {'Kadar Glukosa': Glucose,
            'Tekanan Darah': BloodPressure,
            'Ketebalan Kulit': SkinThickness,
            'Kadar Insulin': Insulin,
            'BMI':BMI,
            'Usia':Age}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Displays the user input features
st.subheader('Datamu')
st.write(df)


# Reads in saved classification model
load_clf = pickle.load(open('diabetes_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

st.subheader('Prediksi')
Kondisi = np.array(['Sehat','Diabetes'])
st.write(Kondisi[prediction])

st.subheader('Kemungkinan Prediksi')
st.write(prediction_proba)


st.write("""
#
Keterangan : Kolom 0 Menunjukan % Kemungkinan Sehat dan Kolom 1 menunjukan % 
kemungkinan  Diabetes.
""")
