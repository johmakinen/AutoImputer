from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

class MySimpleImputer:
    def __init__(self,target_col,**kwargs):
        self.target_col = target_col
        self.strategy = kwargs.get('strategy','mean')

    def impute(self,df):
        imp = SimpleImputer(missing_values=np.nan, strategy=self.strategy)
        if self.target_col == 'all':
            idf = pd.DataFrame(imp.fit_transform(df))
            idf.columns = df.columns
            idf.index = df.index
        else:
            idf = df.copy()
            imp.fit(idf[self.target_col].values.reshape(-1, 1))
            idf[self.target_col] = imp.transform(idf[self.target_col].values.reshape(-1, 1))
        return idf

# x = MySimpleImputer(target_col='sepal length (cm)',strategy='mean')
# res = x.impute(df)

# Haluan että voin kutsua app.py:ssä:
# x = MyClass('simpleimputer')
# res = x.impute(**kwargs)
# val_errors = x.run_validation()



# def impute_values(df,target_col,method,**kwargs):
#     if method == 'SimpleImputer':
#         imp = SimpleImputer(missing_values=np.nan, strategy=kwargs.get('strategy', 'mean'))
#         if target_col == 'all':
#             idf = pd.DataFrame(imp.fit_transform(df))
#             idf.columns = df.columns
#             idf.index = df.index
#         else:
#             idf = df.copy()
#             imp.fit(idf[target_col])
#             idf[target_col] = imp.transform(idf[target_col])
#         return idf
#     else:
#         return df


def measure_error(y_real, y_pred, type='RMSE'):
    if type == 'RMSE':
        return mean_squared_error(y_real, y_pred)
