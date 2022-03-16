# Main code here
import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn import datasets

# My modules
# from src.data_fn.data_process import load_sample_iris
from src.models_fn.imputer_models import MySimpleImputer,XGBImputer,measure_val_error
from src.data_fn.data_process import test_input_data,format_dtypes


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

# call back function -> runs BEFORE the rest of the app
def reset_button():
    st.session_state["p"] = False
    st.session_state['cat_cols'] = []
    return

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
st.sidebar.title('Settings')



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

if GOT_DATA:
    # Get user input for column types
    cols = df.columns
    cat_cols = st.multiselect(
                        'Please input categorical columns here.',
                        cols,
                        default=None,
                        key='cat_cols')
    dtypes = ['numeric' if col not in cat_cols else 'categorical' for col in cols]
    c1,c2,_ = st.columns((2,2,5))
    with c1:
        submit_cols = st.checkbox(label="Submit column types", key='p')
    with c2:
        #button to control reset
        reset=st.button('Set columns again', on_click=reset_button)


FORMATTED_DATA = 0
if GOT_DATA and submit_cols:
    df,dtype_list = format_dtypes(df,dtypes,cols)
    FORMATTED_DATA = 1


if GOT_DATA and FORMATTED_DATA:
    res = df.copy()
    st.markdown("### Imputer selection and settings")

    # Default options:
    strategy = 'mean'
    cv_opt = 3


    # Select imputing method
    method = st.selectbox('Choose the imputing algorithm', ['SimpleImputer', 'XGBoost'])
    if method == 'SimpleImputer':
        with st.expander("Settings:"):
            strategy = st.radio('Select imputing method',
                    ('mean','median','most_frequent'))
            st.write('Do note that SimpleImputer treats all columns as numeric.')
    elif method == 'XGBoost':
        with st.expander("Settings:"):
            cv_opt = st.slider('Number of folds to use in cross-validation when learning the best parameters.',
                            1,5,3)

    # If selection ready, press "Submit" to impute.
    with st.form(key='my_form'):
        submit_btn = st.form_submit_button(label="Impute!")
        imputer_dict = {'SimpleImputer': MySimpleImputer(strategy=strategy),
                        'XGBoost': XGBImputer(dtype_list=dtype_list,random_seed=42,verbose=0,cv=cv_opt)}
    if submit_btn:
        # imputer = MySimpleImputer(strategy=strategy)
        imputer = imputer_dict[method]
        res = imputer.impute(df)

        # Measure validation error
        st.markdown("### Validation error")
        n_folds = 5
        error = measure_val_error(df,imputer=imputer,n_folds=n_folds)
        st.write(f"""Using {n_folds}-fold cross_validation      
            Root Mean Squared Error (RMSE): {round(error[0],2)} $\pm$ {round(error[1],2)}""")
            
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

