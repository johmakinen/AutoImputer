# Main code here
import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn import datasets

# My modules
from src.data_fn.data_process import load_sample_iris
from src.models_fn.imputer_models import impute_values, measure_error

st.set_page_config(
    page_title="AutoImputer",
    layout='wide',
    initial_sidebar_state='auto',
)


# ----------------------------------------------------------------------------------------
# FUNCTIONS

@st.experimental_memo
def load_sample_data():
    iris = load_sample_iris()
    return iris


@st.experimental_memo
def impute_values_(df):
    return impute_values(df)

# LAYOUT


data = load_sample_data()
idx = data[pd.isnull(data).any(axis=1)].index
imputed_data = impute_values_(data)


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

c1, _, c2 = st.columns((3, 0.2, 3))
with c1:

    st.header("Current data")
    st.write(data.iloc[idx])

with c2:

    st.header("Imputed values")
    st.write(imputed_data.iloc[idx])

# st.header('Complete imputed data')
# res_data =
# st.write(res_data)
