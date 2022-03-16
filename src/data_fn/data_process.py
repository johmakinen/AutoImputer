from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import os
import random


def create_iris_sample():
    """Load and create iris dataset,
        save it to csv"""
    path_project = os.path.abspath(os.path.join(__file__, "../../.."))
    path_processed_data = path_project + "/data/processed/"
    # path_raw_data = path_project+'/data/raw/'
    # import some data to play with
    iris = datasets.load_iris()
    df = pd.DataFrame(
        data=np.c_[iris["data"], iris["target"]],
        columns=iris["feature_names"] + ["target"],
    )
    df.to_csv(path_processed_data + "iris_original.csv", index=False)
    return df


def simulate_missing_values(df, output_name, prop=0.4, n_cols=1):
    """Adds missing values to a dataframe,
            saves the df into a csv.
            Used mostly for sample data purposes."""
    path_project = os.path.abspath(os.path.join(__file__, "../../.."))
    path_processed_data = path_project + "/data/processed/"

    nans = df.copy()
    nans.loc[
        df.sample(int(df.shape[0] * prop), random_state=32).index,
        random.sample(df.columns.values.tolist(), k=n_cols),
    ] = np.nan
    nans.to_csv(path_processed_data + output_name + ".csv", index=False)


def test_input_data(df):
    """Tets the given input data for:
        1. Is it empty
        2. What dtypes it has
        3. Proportion of nan values, impossible to impute if over 80%(?)"""
    res = {
        "is_empty": df.empty,
        "dtypes": set(df.dtypes.tolist()),
        "prop_missing": df.isnull().sum().sum() / (df.shape[0] * df.shape[1]),
    }

    return res


def format_dtypes(df, dtypes, cols):
    dtype_list = dict(zip(cols, dtypes))
    categorical_columns = [col for col in cols if dtype_list[col] == "categorical"]
    le = LabelEncoder()
    df[categorical_columns] = df[categorical_columns].apply(le.fit_transform)
    return df, dtype_list


if __name__ == "__main__":
    df = create_iris_sample()
    simulate_missing_values(
        df, output_name="sample_data_with_errors", prop=0.9, n_cols=5
    )
