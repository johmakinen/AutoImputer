import pytest
import numpy as np
import pandas as pd
from src.models_fn.imputer_models import MySimpleImputer, XGBImputer, measure_val_error
from src.data_fn.data_process import simulate_missing_values
import random
import string


def get_test_data():
    """Creates test set for imputers

    Returns
    -------

    list[(data,dtypes)]
    """
    max_size = 10
    test_set = []
    letters = string.ascii_lowercase

    for size in range(2, max_size, 2):
        # Random column names
        col_names = [
            "".join(random.choice(letters) for i in range(10)) for x in range(size)
        ]
        # Initialise all columns as numeric
        dtypes = ["numeric"] * len(col_names)
        curr_df = pd.DataFrame(
            np.random.random_sample(
                size=(size * max(25, np.random.randint(low=5, high=20)), size)
            ),
            columns=col_names,
        )

        # Assign categorical columns randomly:
        n_cat_cols = np.random.randint(low=0, high=len(col_names) - 1)
        cat_cols = curr_df.sample(n=n_cat_cols, axis="columns").columns
        for col in cat_cols:
            n_classes = np.random.randint(low=2, high=5)
            classes_strs = np.random.choice(
                a=[
                    "".join(random.choice(letters) for i in range(5))
                    for x in range(n_classes)
                ],
                size=curr_df.shape[0] - n_classes,
            )
            curr_df[col] = pd.Series(
                # Need to make sure the column has at least two classes
                np.append(classes_strs, np.unique(classes_strs)),
                index=curr_df.index,
            )
            dtypes[curr_df.columns.tolist().index(col)] = "categorical"

        # Simulate missing values
        curr_df = simulate_missing_values(
            curr_df, output_name=None, prop=np.random.uniform(low=0.05, high=0.7)
        )

        # Add a few np.inf values to random dataframes,
        # We want the models to be robust
        if np.random.random() < 0.8:
            inf_col = random.choice(
                [col for col in curr_df.columns if col not in cat_cols]
            )
            curr_df.loc[
                np.random.randint(low=0, high=curr_df.shape[0]), inf_col
            ] = np.inf
            curr_df.loc[
                np.random.randint(low=0, high=curr_df.shape[0]), inf_col
            ] = -np.inf
        test_set.append((curr_df, dtypes))

    return test_set


@pytest.mark.parametrize(
    "imputer, test_data",
    [
        (MySimpleImputer(dtype_list=None, strategy="mean"), get_test_data()),
        (XGBImputer(dtype_list=None, cv=2), get_test_data()),
    ],
    ids=["SimpleImputer", "XGBImputer"],
)
def test_imputer(imputer, test_data):
    """Tests that an imputer works as intended

    | Current tests:    
        | 1. Imputer does not mutate the data shape 
        | 2. If input is empty, output should be empty  
        | 3. All missing values are filled  
    """

    for curr in test_data:
        df = curr[0]
        dtypes = curr[1]
        imputer.dtype_list = dict(zip(df.columns, dtypes))
        res = imputer.impute(df)

        assert df.shape == res.shape  # Shape is not changed
        assert df.empty == res.empty  # If input is empty, output is empty
        assert res.isnull().sum().sum() == 0  # All missing values filled


@pytest.mark.parametrize(
    "imputer, test_data",
    [(MySimpleImputer(dtype_list=None, strategy="mean"), get_test_data()),],
    ids=["SimpleImputer"],
)
def test_validation_error(imputer, test_data):
    """Test the computation of validation error
        Simple test that each column gets an error/accuracy score
    """
    for curr in test_data:
        df = curr[0]
        dtypes = curr[1]
        imputer.dtype_list = dict(zip(df.columns, dtypes))
        errors = measure_val_error(df, imputer=imputer, n_folds=2)

        assert len(errors) == len(df.columns)  # All features have validation errors


# Instructions:
# Run testsuite from the main directory.
# python -m pytest -r pyTest\test_suite.py #from main dir
# python -m pytest -p no:warnings -r pyTest\test_suite.py
