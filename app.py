import streamlit as st
import pandas as pd
import numpy as np
from timeit import default_timer as timer
from pathlib import Path
import base64
import os

# My modules
from src.models_fn.imputer_models import MySimpleImputer, XGBImputer, measure_val_error
from src.data_fn.data_process import test_input_data
from src.visualization_fn.visuals import plot_na_prop


st.set_page_config(
    page_title="AutoImputer", layout="wide",
)

# ----------------------------------------------------------------------------------------
# FUNCTIONS:
@st.experimental_memo
def convert_df(df):
    """Encode dataframe for csv writing.
        (Streamlit needs this)

    Parameters
    ----------
    df : pd.DataFrame
        Input data

    Returns
    -------
    Encoded csv object
        
    """
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")


@st.experimental_memo
def plot_nas(df):
    """Plots the missing value proportions for each column.
        Streamlit's cached version.

    Parameters
    ----------
    df : pd.DataFrame
        Input data  

    Returns
    -------
    Plotly figure object
    """
    return plot_na_prop(df)


# call back function -> runs BEFORE the rest of the app
def reset_button(infer_cat_cols):
    """Resets session states for some variables.
    """
    st.session_state["p"] = False
    st.session_state["cat_cols"] = infer_cat_cols
    return


# ----------------------------------------------------------------------------------------
# LAYOUT
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


header_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
    img_to_bytes("figures/Logo_figma.png")
)


st.markdown(
    header_html, unsafe_allow_html=True,
)


t1, t2 = st.columns((9, 3))

with t2:
    st.write("")
    st.write("")
    st.write(
        """
    **By Johannes Mäkinen** | [johmakinen.github.io](https://johmakinen.github.io)
    """
    )

st.write("")
with st.expander("What is this app?", expanded=False):
    st.write(
        """
    Imputing can be seen as assigning a value to something by inference. With this app, you are able to impute the missing values of your dataset easily and accurately.   
    All you have to do is to upload your data as a .csv file to the app, and follow the instructions. Afterwards, you have the chance to download the imputed dataset.    
            """
    )
st.write("")

uploaded_file = st.file_uploader("Upload CSV", type=".csv")

use_example_file = st.checkbox(
    "Use example file", False, help="Use in-built example file to demo the app"
)

GOT_DATA = 0
if use_example_file:
    cwd = os.getcwd()
    uploaded_file = cwd + "/data/processed/iris_nans.csv"

# Read data, error handling
if uploaded_file:
    FILE_OK = 1
    df = pd.read_csv(uploaded_file)
    val_df = test_input_data(df)
    if val_df["is_empty"]:
        st.error("Error: Empty data")
        FILE_OK = 0

    if val_df["prop_missing"] > 0.75:
        st.warning(
            "Warning: Proportion of missing values too high for accurate imputation ("
            + str(int(val_df["prop_missing"] * 100))
            + "%)"
        )
    if val_df["bool_all_rows_have_nans"]:
        st.warning(
            "Warning: All rows contain at least one missing value. Can't compute validation error."
        )

    if val_df["n_full_nan_rows"]:
        st.warning(
            "Warning: Rows with all missing values found in the input data. Removing these rows.."
        )
        df.dropna(how="all", inplace=True)

    if val_df["n_full_nan_cols"]:
        st.warning(
            "Warning: Columns with all missing values found in the input data. Removing these columns.."
        )
        df.dropna(axis=1, how="all", inplace=True)

    if FILE_OK:
        st.markdown("### Data preview")
        st.dataframe(df[pd.isnull(df).any(axis=1)].head())
        with st.expander("Missing values plotted"):
            st.plotly_chart(plot_nas(df), use_container_width=True)
        GOT_DATA = 1

if GOT_DATA:
    # Get user input for column types
    cols = df.columns

    # Infer coltypes:
    test_df = df.dropna()
    text_cols = [
        x
        for x in cols
        if (test_df[x].dtype == object) and (isinstance(test_df.iloc[0][x], str))
    ]
    low_cardinality_cols = [
        col for col in test_df.columns if len(np.unique(test_df[col])) < 5
    ]
    infer_cat_cols = set(text_cols + low_cardinality_cols)

    # This gave a warning at some point, then it didn't.
    # The widget with key "cat_cols" was created with a default value but also had its value set via the Session State API.

    cat_cols = st.multiselect(
        "Please input categorical columns here. We have inferred some of them already.",
        cols,
        default=infer_cat_cols,
        key="cat_cols",
    )

    dtypes = ["numeric" if col not in cat_cols else "categorical" for col in cols]

    c1, c2, _ = st.columns((2, 2, 5))
    with c1:
        submit_cols = st.checkbox(label="Submit column types", key="p")
    with c2:
        # button to control reset
        reset = st.button(
            "Set columns again", on_click=reset_button, args=(infer_cat_cols,)
        )


GOT_DTYPE_LIST = 0
if GOT_DATA and submit_cols:
    dtype_list = dict(zip(cols, dtypes))
    obv_wrong_dtype = [
        col
        for col in test_df.columns
        if (dtype_list[col] == "numeric") and (col in text_cols)
    ]
    if obv_wrong_dtype:
        st.warning(
            "Warning: A column with string features was selected as 'numeric'. Please check your column selection for columns: "
            + str(obv_wrong_dtype)
        )
    else:
        GOT_DTYPE_LIST = 1


if GOT_DATA and GOT_DTYPE_LIST:
    res = df.copy()
    st.markdown("### Imputer selection and settings")

    # Default options:
    strategy = "mean"
    cv_opt = 2

    # Select imputing method
    method = st.selectbox("Choose the imputing algorithm", ["SimpleImputer", "XGBoost"])
    if method == "SimpleImputer":
        with st.expander("Settings:"):
            strategy = st.radio(
                "Select imputing method", ("mean", "median", "most_frequent")
            )
            st.write(
                "Do note that SimpleImputer will use 'most_frequent' strategy for all categorical features."
            )
    elif method == "XGBoost":
        with st.expander("Settings:"):
            cv_opt = st.slider(
                "Number of folds to use in cross-validation when learning the best parameters. Due to Streamlit's limited resources, it is suggested to not use too large values.",
                2,
                4,
                2,
            )
            st.markdown(
                """Extreme Gradient Boosting works by predicting each column\'s missing values using the other columns as features.   
                    This is done for each column that has missing values. For large dataset, or datasets with high cardinal categorical features, the runtime of the model can
                    be long."""
            )

    # If selection ready, press "Submit" to impute.
    with st.form(key="my_form"):
        submit_btn = st.form_submit_button(label="Impute!")
        imputer_dict = {
            "SimpleImputer": MySimpleImputer(dtype_list=dtype_list, strategy=strategy),
            "XGBoost": XGBImputer(
                dtype_list=dtype_list, random_seed=42, verbose=0, cv=cv_opt
            ),
        }
    # Impute! -button is pressed -> impute, measure elapsed time, compute validation error.
    if submit_btn:
        start_time = timer()
        imputer = imputer_dict[method]
        with st.spinner("Imputing missing values..."):
            res = imputer.impute(df)

        elapsed_time = timer() - start_time
        st.write(f"Imputation took {round(elapsed_time,2)} seconds.")

        # Show resulting table and the original data
        c1, _, c2 = st.columns((3, 0.2, 3))
        with c1:
            st.subheader("Resulting data")
            st.write(res)
        with c2:
            st.subheader("Original data")
            st.write(df)

        # Measure validation error
        with st.expander("Validation metrics"):
            st.write(
                r"""
                    #### For numeric features the validation error is measured with Root Mean Squared Error (RMSE).
                    _(Lower is better)_   
                    $$ 
                    RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_{\text{pred}}-y_{\text{real}})^2}
                    $$ 
                    """
            )
            st.write(
                r"""
                    #### For categorical features the measure is $F_1$ score.
                    _(higher is better)_   
                    For binary features
                    $$ 
                    F_1 = \frac{\text{true positives}}   {\text{true positives} + \frac{1}{2}(\text{false positives}+\text{false negatives})}
                    $$ 
                    For multiclass features micro -averaging is used:
                    _Micro averaging computes a global average F1 score by counting the sums of the True Positives (TP),
                     False Negatives (FN), and False Positives (FP) over all classes. These are then plugged in the above $F_1$ equation._
                    """
            )
            n_folds = 3
            with st.spinner("Validating..."):
                error = measure_val_error(df, imputer=imputer, n_folds=n_folds)
            st.subheader(f"Metrics with {n_folds} validation folds.")
            st.write(error)

    # Give ability to download resulting data.
    csv = convert_df(res)
    st.download_button(
        label="Download result data as CSV",
        data=csv,
        file_name="output.csv",
        mime="text/csv",
    )

