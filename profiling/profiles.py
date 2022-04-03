import sys
import os
from src.models_fn.imputer_models import measure_val_error,XGBImputer,MySimpleImputer
from src.data_fn.data_process import simulate_missing_values,replace_infs
from pyTest.test_suite import get_test_data
from line_profiler import LineProfiler
from sklearn.preprocessing import LabelEncoder
import random
import pandas as pd
import numpy as np




if __name__ == '__main__':
    # Instructions:
    # Run python -m profiling.profiles from main folder.

    
    test_data = get_test_data()
    # for curr in test_data:
    df = test_data[0][0]
    dtypes = test_data[0][1]
    dtype_list = dict(zip(df.columns, dtypes))

    # Profile imputer impute method ------
    # imp = MySimpleImputer(dtype_list=dtype_list,strategy='mean')
    imp = XGBImputer(dtype_list=dtype_list)

    # lp = LineProfiler()
    # lp_wrapper = lp(imp.impute)
    # lp_wrapper(df)
    # lp.print_stats()


    

    # Profile XGBImputer train method ------
    lp = LineProfiler()
    le = LabelEncoder()
    curr_col = df.columns[0]
    df_train = replace_infs(df.copy().dropna(subset=[curr_col]))
    df_train.loc[:, curr_col] = le.fit_transform(df_train[curr_col])

    lp_wrapper = lp(imp.train)
    lp_wrapper(*[df_train,curr_col])
    lp.print_stats()
    # 99.8% of time is spent on gridsearch.

    # Profile measure_val_error ------
    # lp = LineProfiler()
    # lp_wrapper = lp(measure_val_error)

    # lp_wrapper(*[df,imp,5])
    # lp.print_stats()
    # 99.5% of time is spent on imputing -> of which as seen previously 99.8% is spent on gridsearch.

    #--> All in all, grid search is the bottleneck for our performance, as could be expected.


