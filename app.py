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
from src.data_fn.data_process import test_input_data


st.set_page_config(
    page_title="AutoImputer",
    layout='wide',
    initial_sidebar_state='auto',
)

# ----------------------------------------------------------------------------------------
# FUNCTIONS:
@st.experimental_memo
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')
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
st.markdown("""Imputing missing values automatically...   
                _Only numerical data for now_""")

uploaded_file = st.file_uploader("Upload CSV", type=".csv")

use_example_file = st.checkbox(
    "Use example file", False, help="Use in-built example file to demo the app"
)

GOT_DATA = 0
if use_example_file:
    uploaded_file = "data\processed\iris_nans.csv"

# Read data, error handling
if uploaded_file:
    FILE_OK = 1
    df = pd.read_csv(uploaded_file)
    val_df = test_input_data(df)
    if val_df['is_empty']:
        st.error('Error: Empty data')
        FILE_OK = 0
    if np.dtype('object') in val_df['dtypes']:
        st.error('Error: Non-numeric data not supported yet')
        FILE_OK = 0
    if val_df['prop_missing'] > 0.75:
        st.error('Error: Proportion of missing values too high (' + str(int(val_df['prop_missing']*100)) +'%)')
        FILE_OK = 0
    if FILE_OK:
        st.markdown("### Data preview")
        st.dataframe(df[pd.isnull(df).any(axis=1)].head())
        GOT_DATA = 1


# Select data according to the input
st.sidebar.title('Select target column(s) to impute')

if uploaded_file:
    nan_cols = np.append(np.array(['ALL']),df.columns[df.isna().any()].values)
    target_col = st.sidebar.selectbox(
     '',
     nan_cols)


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

    # If selection ready, press "Submit" to impute.
    with st.form(key='my_form'):
        submit_btn = st.form_submit_button(label="Impute!")
    if submit_btn:
        imputer = MySimpleImputer(target_col=target_col,**opts)
        res = imputer.impute(df)

    # Show resulting table and the original data
    c1,_,c2 = st.columns((3,0.2,3))
    with c1:
        st.subheader('Resulting data')
        st.write(res)
    with c2:
        st.subheader('Original data')
        st.write(df)

    # Give ability to download resulting data.
    csv = convert_df(res)
    st.download_button(
        label="Download result data as CSV",
        data=csv,
        file_name='output.csv',
        mime='text/csv',
    )
        
        # Show validation error (RMSE if numerical, something else if categorical)

