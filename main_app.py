import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components


st.set_page_config(layout="wide")

st.markdown("<h1 style='text-align: center; '>SME Loan Application </h1>", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; '> </h1>", unsafe_allow_html=True)

cb = pickle.load(open('jomhack(1.0).sav', 'rb'))

col1, col2, col3, col4, col5  = st.beta_columns(5)

def get_user_input():
    sector = col1.selectbox('Industrial Sector:', ['Accommodation and food services',
       'Administrative and support and waste management and remediation services',
       'Agriculture, forestry, fishing and hunting',
       'Arts, entertainment, and recreation', 'Construction',
       'Educational services', 'Finance and insurance',
       'Health care and social assistance', 'Information',
       'Management of companies and enterprises', 'Manufacturing',
       'Mining, quarrying, and oil and gas extraction', 'Other services',
       'Professional, scientific, and technical services',
       'Public administration', 'Real estate and rental and leasing',
       'Retail trade', 'Transportation and warehousing', 'Utilities',
       'Wholesale trade'])
    term = col2.number_input('Loan Term (Months):', 1, 600, 105)
    business = col3.selectbox('Existing/New Business:', ['Existing Business', 'New Business'])
    credit = col4.selectbox('Revolving line of credit:', ['Yes', 'No'])
    amount = col5.number_input('Loan Amount ($):', 1000, 5000000, 150000)
   
    user_data = {'Industrial Sector': sector, 'Loan Term': term, 'Existing Business or New Business': business,
    			'Revolving line of credit': credit, 'Loan Amount': amount}

    features = pd.DataFrame(user_data, index=[0])

    return features

user_input = get_user_input()
ex1 = shap.TreeExplainer(cb)
shap_values1 = ex1.shap_values(user_input)
shap.initjs()

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

st.markdown("<h1 style='text-align: center; '> </h1>", unsafe_allow_html=True)    

if cb.predict(user_input)[0] == 1:
    st.success("Loan application is approved with {}% risk of default".format(((1-cb.predict_proba(user_input)[:,1][0])*100).round(2)))
else: 
    st.warning("Loan application is not approved with {}% risk of default".format(((1-cb.predict_proba(user_input)[:,1][0])*100).round(2)))

  
col1, col2, col3, col4, col5  = st.beta_columns(5)


if col3.button("Here's why:"):
    st_shap(shap.force_plot(ex1.expected_value, shap_values1[0, :], user_input))


uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    df1=pd.read_csv(uploaded_file) 
    for i in range(len(cb.predict(df1))):
        if cb.predict(df1)[i] == 1:
            st.sidebar.success('Applicants {}: Approved'.format(i+1)) 
        else:
            st.sidebar.warning('Applicants {}: Not Approved'.format(i+1))










