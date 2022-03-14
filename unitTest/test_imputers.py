import pytest
import numpy as np
import pandas as pd
from src.models_fn.imputer_models import MySimpleImputer
import random

@pytest.fixture
def get_test_data_single_col():
    """Creates test dataframes where single column need imputing"""
    test_set = []
    for size in range(20,100,10):
        col_names = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')[:10]
        rand_col = random.choice(col_names)
        curr_df = pd.DataFrame(np.random.randint(-50,9000,size=(size,10)), columns=col_names)
        curr_df.loc[curr_df.sample(np.random.randint(5,int(size*0.4)), random_state=32).index,rand_col] = np.nan
        test_set.append((curr_df,rand_col,curr_df[rand_col].mean(skipna=True)))
    return test_set

def test_SimpleImputer_single_col(imputer,get_test_data_single_col):
    """Tests a Simple mean imputer for a single column.
        Other imputers will not get a test as the missing values can
        be filled with different accuracy by each model."""
    for curr in get_test_data_single_col:
        df = curr[0]
        col = curr[1]
        expected = curr[2]
        imp = MySimpleImputer(target_col=col,**{'strategy':'mean'})
        res = imp.impute(df)
        assert  np.round(res[col].mean(),2) == np.round(expected,2)


#def test_input()...


# Instructions:
# Run testsuite from the main directory.
# python -m pytest -r unitTest\test_imputers.py #from main dir