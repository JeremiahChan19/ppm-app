import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn import metrics

st.write('''
# PPM Prediction App

This app predicts if the person has Mitral PPM.
''')

st.sidebar.header('User Input Parameters')

def user_input_features():
    age = st.sidebar.slider('Age', 20, 100, 55)
    gender = st.sidebar.slider('Gender',1,0,1)
    height = st.sidebar.slider('Height', 130, 200, 160)
    weight = st.sidebar.slider('Weight', 40, 120, 60)
    bmi = st.sidebar.slider('BMI', 10.0, 35.0, 20.0, step=0.1)
    bsa = st.sidebar.slider('BSA', 0.7, 3.0, 1.5, step=0.1)
    smoking_history = st.sidebar.slider('Smoking History',1,0,1)
    hypertension = st.sidebar.slider('Hypertension',1,0,1)
    coronary_heart_disease = st.sidebar.slider('Coronary Heart Disease',1,0,1)
    mitral_stenosis = st.sidebar.slider('Mitral Stenosis',1,0,1)
    lvef = st.sidebar.slider('LVEF', 0, 100, 50)
    
    data = {
        'Age': age,
        'Gender': gender,
        'Height': height,
        'Weight': weight,
        'BMI': bmi,
        'BSA': bsa,
        'Smoking History': smoking_history,
        'Hypertension': hypertension,
        'Coronary Heart Disease': coronary_heart_disease,
        'Mitral Stenosis': mitral_stenosis,
        'LVEF': lvef
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

ppm = pd.read_excel(r"C:\Users\surv535\Desktop\Python files Jeremiah\ppm_data.xlsx")
y = ppm['PPM']
x = ppm.drop('PPM', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

clf = xgb.XGBClassifier()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
st.write(accuracy)

# Now you can use the same df for prediction
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Prediction')
st.write(ppm['PPM'][prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
