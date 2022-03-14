# Main code here
import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn import datasets

# My modules
# from src.data_fn.data_process import load_sample_iris
from src.models_fn.imputer_models import MySimpleImputer

st.set_page_config(
    page_title="AutoImputer",
    layout='wide',
    initial_sidebar_state='auto',
)


# ----------------------------------------------------------------------------------------
# LAYOUT
t1, t2 = st.columns(2)
with t1:
    st.title('AutoImputer')

with t2:
    st.write("")
    st.write("")
    st.write("""
    **By Johannes Mäkinen** | [johmakinen.github.io](https://johmakinen.github.io)
    """)

st.write("")
st.markdown("""Imputing missing values automatically...""")

uploaded_file = st.file_uploader("Upload CSV", type=".csv")

use_example_file = st.checkbox(
    "Use example file", False, help="Use in-built example file to demo the app"
)

if use_example_file:
    uploaded_file = "data\processed\iris_nans.csv"
    
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    # validate uploaded_file: dtypes, size, has nans, is not empty, ...
    st.markdown("### Data preview")
    st.dataframe(df[pd.isnull(df).any(axis=1)].head())

# Select data according to the input
st.sidebar.title('Select target column(s) to impute')
if uploaded_file and (uploaded_file != "data\processed\iris_nans.csv"):
    target_col = st.sidebar.multiselect(
        '', set(np.append(df.columns.values,'all')), default=df.columns[0])
elif use_example_file:
    target_col = st.sidebar.multiselect(
        'Only sepal length (cm) available for sample data', set(['sepal length (cm)']), default='sepal length (cm)')

GOT_DATA = 1 if uploaded_file or use_example_file else 0


if GOT_DATA:
    res = df.copy()
    st.markdown("### Imputer selection and settings")
    # Select imputing method
    method = st.selectbox('Choose the imputing algorithm', ['SimpleImputer', 'Something_else'])
    if method == 'SimpleImputer':
        with st.expander("Settings:"):
            strategy = st.radio('Select imputing method',
                    ('mean','median','most_frequent'))
            opts = {'strategy':strategy}
    elif method == 'Something_else':
            opts = {'strategy':'mean'}

    with st.form(key='my_form'):
        submit_btn = st.form_submit_button(label="Impute!")

    if submit_btn:
        imputer = MySimpleImputer(target_col=target_col,**opts)
        res = imputer.impute(df)

    c1,_,c2 = st.columns((3,0.2,3))
    with c1:
        st.subheader('Resulting data')
        st.write(res)
    with c2:
        st.subheader('Original data')
        st.write(df)
        
        # Show validation error (RMSE if numerical, something else if categorical)
