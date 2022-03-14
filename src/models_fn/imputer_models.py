from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np


def impute_values(df):
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    idf = pd.DataFrame(imp.fit_transform(df))
    idf.columns = df.columns
    idf.index = df.index
    return idf


def measure_error(y_real, y_pred, type='RMSE'):
    if type == 'RMSE':
        return mean_squared_error(y_real, y_pred)
