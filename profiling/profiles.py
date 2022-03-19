import sys
import os
from src.models_fn.imputer_models import measure_val_error,XGBImputer,MySimpleImputer
from src.data_fn.data_process import simulate_missing_values,replace_infs
from pyTest.test_suite import get_test_data
from line_profiler import LineProfiler
import random
import pandas as pd
import numpy as np




if __name__ == '__main__':
    # Instructions:
    # Run python -m profiling.profiles from main folder.

    # def do_stuff(numbers):
    #     s = sum(numbers)
    #     l = [numbers[i]/43 for i in range(len(numbers))]
    #     m = ['hello'+str(numbers[i]) for i in range(len(numbers))]

    # numbers = [random.randint(1,100) for i in range(1000)]
    test_data = get_test_data()
    # for curr in test_data:
    df = test_data[0][0]
    dtypes = test_data[0][1]
    dtype_list = dict(zip(df.columns, dtypes))
    imp = MySimpleImputer(dtype_list=dtype_list,strategy='mean')

    lp = LineProfiler()
    lp_wrapper = lp(imp.impute)
    lp_wrapper(df)
    lp.print_stats()