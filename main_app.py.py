import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components


st.set_page_config(layout="wide")

st.markdown("<h1 style='text-align: center; color: white;'>SME Loan Application </h1>", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; color: #051723;'>  </h1>", unsafe_allow_html=True)

cb = pickle.load(open('jomhack.sav', 'rb'))

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
    term = col2.number_input('Loan Term (Months):', 0, 527, 105)
    business = col3.selectbox('Existing Business or New Business:', ['Existing Business', 'New Business'])
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

      

if cb.predict(user_input)[0] == 1:
    st.markdown("<h2 style='text-align: center; color: blue;'>Loan application is approved </h1>", unsafe_allow_html=True)
else: 
    st.markdown("<h2 style='text-align: center; color: red;'>Loan application is not approved</h1>", unsafe_allow_html=True)
  
col1, col2, col3, col4, col5  = st.beta_columns(5)

if col3.button('Why?'):
    st_shap(shap.force_plot(ex1.expected_value, shap_values1[0, :], user_input))









