from pathlib import Path
print('Running' if __name__ == '__main__' else 'Importing', Path(__file__).resolve())

from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
# from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
# from src.data_fn.data_process import format_dtypes,replace_infs
from ..data_fn.data_process import format_dtypes,replace_infs
import pandas as pd
import numpy as np
import os
import warnings



warnings.filterwarnings("ignore")
import xgboost as xgb


class MySimpleImputer:
    """ A Class for the simple imputer.
    """

    def __init__(self, strategy="mean"):
        """Initialise imputer

        Parameters
        ----------
        strategy : str, optional
            What strategy to use, by default 'mean'
            Other options: mean, median or most_frequent.
        """
        self.strategy = strategy

    def impute(self, df):
        """ Fills the missing values with the given strategy.

        Parameters
        ----------
        df : pd.DataFrame
            Full DataFrame that is to be imputed

        Returns
        -------
        pd.DataFrame
            The resulting data where missing values have been filled using the given strategy.
        """
        if df.empty:
            return df
        
        imp = SimpleImputer(missing_values=np.nan, strategy=self.strategy)
        idf = pd.DataFrame(imp.fit_transform(replace_infs(df)))
        idf.columns = df.columns
        idf.index = df.index
        return idf


class XGBImputer:
    """Imputing class that uses XGBoost for each column that has missing values.
    """

    def __init__(self, dtype_list, random_seed=42, verbose=0, cv=1):
        """Initialise imputer

        Parameters
        ----------
        dtype_list : dict
            A dict that shows what type of data each column has:
            {column_name_1:'numerical',column_name_2:'categorical',...}
        random_seed : int, optional
            , by default 42
        verbose : int, optional
            , by default 0
        cv : int, optional
            Number of cross-validation folds used in hyperparameter optimization, by default 1
        """
        self.dtype_list = dtype_list
        self.random_seed = random_seed
        self.verbose = verbose
        self.cv = cv

    def impute(self, df):
        """Impute with XGBoost

        Parameters
        ----------
        df : pd.DataFrame
            Data with the missing values

        Returns
        -------
        pd.DataFrame
            Data with missing values imputed
        """
        if df.empty:
            return df

        # We want to keep separate the results and the current data.
        # This is because we dont want our imputations of one column
        # to affect the imputations of other columns.
        # This can be changed later if we want such behaviour.
        
        curr_df = replace_infs(df.copy())
        result = curr_df.copy()

        nan_cols = df.columns[df.isna().any()].tolist()
        cat_cols = [col for col in curr_df.columns if self.dtype_list[col] == "categorical"]

        for curr_col in nan_cols:
            df_target_col = curr_df.dropna(subset=[curr_col]) # Target has no nans
            df_feature_col = curr_df[curr_df[curr_col].isnull()] # prediction observations are the ones where target is nan

            # If current column to be imputed is categorical -> encode label
            if curr_col in cat_cols:
                # Encode label
                le = LabelEncoder()
                df_target_col[curr_col] = le.fit_transform(df_target_col[curr_col])
                df_feature_col[curr_col] = le.fit_transform(df_feature_col[curr_col])

            # Dummify rest categorical cols
            for cat_col in cat_cols: # For all catgorical cols
                if cat_col != curr_col: # But not current col
                    # Drop cat col in question, and concat the dummified version of that column.
                    df_target_col = pd.concat([df_target_col.drop(cat_col,axis=1), pd.get_dummies(df_target_col[cat_col],prefix=cat_col)], axis=1)
                    df_feature_col = pd.concat([df_feature_col.drop(cat_col,axis=1), pd.get_dummies(df_feature_col[cat_col],prefix=cat_col)], axis=1)

            #Get best model
            best_model = self.train(df_target_col, curr_col)

            # These are the features to use for prediction (where target has nan)
            # X_to_fill = df_target_col[df_target_col[curr_col].isnull()]
            # print(X_to_fill.drop(columns=curr_col))

            preds = pd.Series(
                best_model.predict(df_feature_col.drop(columns=curr_col)),
                index=pd.Index(df_feature_col.index),
            )

            result[curr_col] = result[curr_col].fillna(preds)

        return result

    def train(self, df, curr_col):
        """Train XGBoost model

        Parameters
        ----------
        df : pd.DataFrame
            Data to be imputed
        curr_col : _type_
            What column is currently being imputed

        Returns
        -------
        xgboost.XGBRegressor() or xgboost.Classifier()
            Best model found for current column data
        """
        # Already done in prev phase...
        train_set = df.dropna(subset=[curr_col])

        X_train, y_train = train_set.drop(curr_col, axis=1), train_set[curr_col]

        # Choose type of model:

        if self.dtype_list[curr_col] == "numeric":
            model = xgb.XGBRegressor()
            scoring = "neg_mean_squared_error"
            param_grid = {
                "booster": ["gbtree"],
                "max_depth": [4, 6, 8],
                "alpha": [1, 5, 10],
                "lambda": [0, 3, 5],
                "learning_rate": [0.5],
            }

        elif self.dtype_list[curr_col] == "categorical":
            model = xgb.XGBClassifier(use_label_encoder=False)
            classes = pd.unique(y_train)
            n_classes = len(classes)
            param_grid = {
                "learning_rate": [0.25, 0.3, 0.4],
                "max_depth": [4, 6, 8],
                "subsample": [0.1, 0.5, 1],
                "colsample_bytree": [0.1, 0.5, 1],
                "n_estimators": [10, 50, 100],
            }

            if n_classes == 2:
                model.set_params(objective="binary:logistic")
                scoring = "roc_auc"

            else:
                model.set_params(objective="multi:softmax", num_class=n_classes)
                scoring = "f1_micro"

        grid_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            scoring=scoring,
            n_jobs=-1,
            cv=5,
            refit=True,
            random_state=self.random_seed,
        )

        grid_search.fit(X_train, y_train)

        return grid_search.best_estimator_


def measure_val_error(df, imputer, n_folds=5):
    """ Computes the possible error of the imputation.
        Uses N-fold subsampling and averages the errors
        over the folds. At the moment only RMSE for all data
        columns is available.


    Parameters
    ----------
    df : pd.DataFrame
        Input data with missing values
    imputer : Custom imputer
        Must have an "impute" method, which returns a dataframe. E.g. SimpleImputer()
    n_folds : int, optional
        Number of N-folds to perform, by default 5

    Returns
    -------
    dict
        Mean RMSE error for each column as a dict

    """

    curr_df = df.dropna(axis=0, how="any").copy()
    errors = pd.DataFrame(((curr_df-curr_df)**2).mean(axis=0)**(1/2)).T
    # n-fold cross validation
    for i in range(n_folds):
        nans = curr_df.mask(np.random.random(curr_df.shape) < 0.4)
        res = imputer.impute(nans)
        curr_errors = pd.DataFrame(((res-curr_df)**2).mean(axis=0)**(1/2)).T
        errors = pd.concat([errors,curr_errors],axis=0)

    
    return pd.DataFrame(errors.iloc[1:,:].mean(axis=0)).round(2).to_dict()[0]



if __name__ == "__main__":

    # Load data
    path_project = os.path.abspath(os.path.join(__file__, "../../.."))
    path_processed_data = path_project + "/data/processed/"
    data = pd.read_csv(path_processed_data + "iris_nans.csv")

    # nää pitää tehdä inputille
    dtypes = ["numeric", "numeric", "numeric", "numeric", "categorical"]
    cols = data.columns
    data, dtype_list = format_dtypes(data, dtypes=dtypes, cols=cols)

    # Implementoi tämä -> app.py
    # imp = MySimpleImputer()
    # res = imp.impute(data)

    imp = XGBImputer(dtype_list=dtype_list, random_seed=42, verbose=0, cv=1)
    res = imp.impute(data)
    print(data.head(10))
    print(res.head(10))
    # error_ = measure_val_error(data,imp,n_folds=1)
    # print(error_)
