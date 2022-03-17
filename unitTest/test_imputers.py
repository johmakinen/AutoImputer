import pytest
import numpy as np
import pandas as pd
from src.models_fn.imputer_models import MySimpleImputer, XGBImputer
from src.data_fn.data_process import simulate_missing_values, format_dtypes
import random


@pytest.fixture(scope="session")
def get_test_data():
    """Creates test set for imputers

    Returns
    -------
    list[(pd.DataFrame,(int,int),Boolean)]

    list[(data,dtypes)]
    """
    test_set = []

    for size in range(5, 10, 3):
        col_names = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")[:size]
        dtypes = ["numeric"] * len(col_names)
        curr_df = pd.DataFrame(
            np.random.random_sample(size=(size * 20, size)), columns=col_names
        )

        curr_df = simulate_missing_values(
            curr_df, output_name=None, prop=np.random.random()
        )

        # Assign categorical columns randomly:
        n_cat_cols = np.random.randint(low=0, high=len(col_names))
        cat_cols = curr_df.sample(n=n_cat_cols, axis="columns").columns
        for col in cat_cols:
            n_classes = np.random.randint(low=1, high=5)
            curr_df[col] = pd.Series(
                np.random.choice(a=[*range(0, n_classes)], size=curr_df.shape[0]),
                index=curr_df.index,
            )
            dtypes[curr_df.columns.tolist().index(col)] = "categorical"

        test_set.append((curr_df, dtypes))
    return test_set


def test_Imputers(get_test_data):
    """Tests that each imputer works as intended

    | Current tests:    
        | 1. Imputer does not mutate the data shape 
        | 2. If input is empty, output should be empty  
        | 3. All missing values are filled  
    """
    for i in ["MySimpleImputer", "XGBImputer"]:
        for curr in get_test_data:
            df = curr[0]
            dtypes = curr[1]
            res, dtypes_list = format_dtypes(df, dtypes=dtypes, cols=df.columns)
            if i == "MySimpleImputer":
                imp = MySimpleImputer(strategy="mean")
            if i == "XGBImputer":
                imp = XGBImputer(dtype_list=dtypes_list, cv=1)
            res = imp.impute(res)

            assert df.shape == res.shape  # Shape is not changed
            assert df.empty == res.empty  # If input is empty, output is empty
            assert res.isnull().sum().sum() == 0  # All missing values filled


# def test_input()...


# Instructions:
# Run testsuite from the main directory.
# python -m pytest -r unitTest\test_imputers.py #from main dir
