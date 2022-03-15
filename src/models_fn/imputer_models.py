from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import pandas as pd
import numpy as np
import random


class MySimpleImputer():
    def __init__(self,target_col,**kwargs):
        self.target_col = target_col
        self.strategy = kwargs.get('strategy','mean')

    def impute(self,df):
        imp = SimpleImputer(missing_values=np.nan, strategy=self.strategy)
        if self.target_col == 'ALL':
            idf = pd.DataFrame(imp.fit_transform(df))
            idf.columns = df.columns
            idf.index = df.index
        else:
            idf = df.copy()
            imp.fit(idf[self.target_col].values.reshape(-1, 1))
            idf[self.target_col] = imp.transform(idf[self.target_col].values.reshape(-1, 1))
        return idf




def measure_val_error(df,imputer,n_folds=20):
    """This function takes a dataframe and computes
        the possible validation error."""
    curr_df = df.dropna(axis=0,how='any').copy()
    errors = np.empty(n_folds)
    # 10-fold cross validation
    for i in range(n_folds):
        if imputer.target_col != 'ALL':
            nans = curr_df.mask((np.random.random(curr_df.shape)<.4) & (curr_df.columns == imputer.target_col))
            target = curr_df[nans[imputer.target_col].isna()][imputer.target_col].values
            res = imputer.impute(nans)
            res_values = res[nans[imputer.target_col].isna()][imputer.target_col].values

        else:
            target = np.array(curr_df.to_numpy()).ravel()
            nans = curr_df.mask(np.random.random(curr_df.shape)<0.4)
            res = imputer.impute(nans)
            res_values = np.array(res.to_numpy()).ravel()
        
        errors[i] = mean_squared_error(target,res_values,squared=True)

    return errors.mean(),errors.std()


