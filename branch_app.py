import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import seaborn as sns
import os


st.set_page_config(layout="wide")


uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
	df1=pd.read_csv(uploaded_file)


cb = pickle.load(open('jomhack(1.0).sav', 'rb'))

st.write(cb.predict(df1)[0])

for i in range(len(cb.predict(df1))):
	if cb.predict(df1)[i] == 1:
		st.sidebar.success('Applicants {}: Approved'.format(i+1)) 
	else: 
		st.sidebar.warning('Applicants {}: Not Approved'.format(i+1)) 