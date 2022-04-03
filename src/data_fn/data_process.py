from pathlib import Path

print("Running" if __name__ == "__main__" else "Importing", Path(__file__).resolve())

from sklearn import datasets
import pandas as pd
import numpy as np
import os


def create_iris_sample():
    """Creates a sample dataset using the iris data.
        Writes the data into csv for faster usage.

    Returns
    -------
    pd.DataFrame
        Iris data
    """
    path_project = os.path.abspath(os.path.join(__file__, "../../.."))
    path_processed_data = path_project + "/data/processed/"
    iris = datasets.load_iris()
    df = pd.DataFrame(
        data=np.c_[iris["data"], iris["target"]],
        columns=iris["feature_names"] + ["target"],
    )
    df.to_csv(path_processed_data + "iris_original.csv", index=False)
    return df


def simulate_missing_values(df, output_name=None, prop=0.4):
    """Adds missing values to a dataframe,
            saves the df into a csv if output_name given.
            Used mostly for sample data purposes.

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    output_name : string
        name for the output file (csv)
    prop : float, optional
        Approximate proportion missing values
    
    Returns
    -------
    pd.DataFrame
        Data with missing values
    """
    nans = df.mask(np.random.random(df.shape) < prop)
    if output_name:
        path_project = os.path.abspath(os.path.join(__file__, "../../.."))
        path_processed_data = path_project + "/data/processed/"
        nans.to_csv(path_processed_data + output_name + ".csv", index=False)
    return nans


def test_input_data(df):
    """Tests the given input data for:
        1. Is it empty
        2. What dtypes it has
        3. Proportion of nan values

    Parameters
    ----------
    df : pd.DataFrame
        Input data

    Returns
    -------
    dict
        {
        "is_empty": Boolean,
        "dtypes": set,
        "prop_missing": float,
    }
    """
    res = {
        "is_empty": df.empty,
        "dtypes": set(df.dtypes.tolist()),
        "prop_missing": df.isnull().sum().sum() / (df.shape[0] * df.shape[1]),
        "n_full_nan_rows": len(df.index[df.isnull().all(1)]),
        "n_full_nan_cols": sum(df.isnull().values.all(axis=0) * 1),
        "bool_all_rows_have_nans":sum((df.isnull()).sum(axis=1)>0) == df.shape[0]
    }

    return res


def replace_infs(df):
    """Replaces infinities with min/max of the column

    Parameters
    ----------
    df : pd.DataFrame
        Input data

    Returns
    -------
    pd.DataFrame
    """

    # m1 = df.eq(np.inf)
    # m2 = df.eq(-np.inf)
    # df = df.mask(m1, df[~m1].max(), axis=1).mask(m2, df[~m2].min(), axis=1)
    # For performance resons infinities become nan.
    # Could use max/min of column values.
    res = df.replace([np.inf, -np.inf], np.nan)
    return res


def infer_cols(df):
    test_df = df.dropna()
    text_cols = [
        x
        for x in test_df.columns
        if (test_df[x].dtype == object) and (isinstance(test_df.iloc[0][x], str))
    ]
    low_cardinality_cols = [
        col for col in test_df.columns if len(np.unique(test_df[col])) < 5
    ]
    infer_cat_cols = set(text_cols + low_cardinality_cols)

    return text_cols,infer_cat_cols


if __name__ == "__main__":
    df = create_iris_sample()
    simulate_missing_values(df, output_name="sample_data_with_errors", prop=0.9)
    simulate_missing_values(df, output_name="iris_nans", prop=0.2)
