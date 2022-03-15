from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import pandas as pd
import numpy as np
import os

import xgboost as xgb


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

class XGBImputer():
    def __init__(self,dtype_list=None,random_seed=42,verbose=0,cv=1):
        self.dtype_list = dtype_list
        self.random_seed = random_seed
        self.verbose = verbose
        self.cv = cv

    def impute(self,df):
        result = df.copy()
        nan_cols = df.columns[df.isna().any()].tolist()

        for curr_col in nan_cols:
            best_model = self.train(df,curr_col)
            X_to_fill = df[df[curr_col].isnull()]

            preds = pd.Series(
                best_model.predict(X_to_fill.drop(columns=curr_col)),
                index=pd.Index(X_to_fill.index)
            )
            # preds.index = 
            result[curr_col] = result[curr_col].fillna(preds)

        return result

    def train(self,df,curr_col):
        # Take all expect one column,
        # Train xgboost with cv or not
        # return best params
        # curr_df = df.drop(curr_col,axis=1)
        
        train_set = df.dropna(subset=[curr_col])

        X_train, y_train = train_set.drop(curr_col,axis=1), train_set[curr_col]

        # Choose type of model:

        if self.dtype_list[curr_col] == 'regression':
            model = xgb.XGBRegressor()
            scoring = 'neg_mean_squared_error'
            param_grid = {
            'booster':['gbtree'],
            'max_depth': [4,6,8],
            'alpha': [1,5,10],
            'lambda':[0,3,5],
            'learning_rate': [0.5]
                }  

        elif self.dtype_list[curr_col] == 'classification':
            model = xgb.XGBClassifier()
            classes = pd.unique(data.dropna(subset=[curr_col])[curr_col])
            n_classes = len(classes)
            param_grid = {
                'learning_rate': [0.25,0.3,0.4],
                'max_depth': [4,6,8],
                'subsample': [0.1,0.5,1],
                'colsample_bytree': [0.1,0.5,1],
                'n_estimators': [10,50,100]
                }

            if n_classes == 2:
                model.set_params(objective='binary:logistic')
                scoring = 'roc_auc'

            else:
                model.set_params(objective='multi:softmax', n_classes=n_classes)
                scoring = 'accuracy'

        grid_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            scoring=scoring,
            n_jobs=2,
            cv=5,
            refit=True,
            random_state=self.random_seed
        )

        grid_search.fit(X_train, y_train)

        return grid_search.best_estimator_




if __name__ == '__main__':

    path_project = os.path.abspath(
        os.path.join(__file__, "../../.."))
    path_processed_data = path_project+'/data/processed/'

    imp1 = MySimpleImputer(target_col='ALL')
    data = pd.read_csv(path_processed_data+'iris_nans.csv')
    dtype_list = dict(zip(data.columns, ['regression','regression','regression','regression','classification']))
    print(dtype_list)
    imp2 = XGBImputer(dtype_list=dtype_list,random_seed=42,verbose=0,cv=1)
    res = imp2.impute(data)
    print(res.head())
    # res = imp1.impute(data)
    print(measure_val_error(data,imp1))
    print(measure_val_error(data,imp2))