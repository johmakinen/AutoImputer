from pathlib import Path

print("Running" if __name__ == "__main__" else "Importing", Path(__file__).resolve())

from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from ..data_fn.data_process import replace_infs
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
        self.le_name_mapping = None

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
        # df.dropna(axis=0,how='all',inplace=True)

        # We want to keep separate the results and the current data.
        # This is because we dont want our imputations of one column
        # to affect the imputations of other columns.
        # This can be changed later if we want such behaviour.

        curr_df = replace_infs(df.copy())
        result = curr_df.copy()

        nan_cols = df.columns[df.isna().any()].tolist()
        cat_cols = [
            col for col in curr_df.columns if self.dtype_list[col] == "categorical"
        ]

        for curr_col in nan_cols:

            cat_cols_no_curr_col = list(filter(lambda x: x != curr_col, cat_cols))
            df_with_dummies = curr_df.copy()

            if cat_cols_no_curr_col:

                # Dummify categorical columns
                oe = OneHotEncoder(sparse=False, handle_unknown="ignore", dtype="int")

                # Exclude curr_col
                transformed_data = oe.fit_transform(
                    df_with_dummies[cat_cols_no_curr_col]
                )

                # the above transformed_data is an array so convert it to dataframe
                encoded_data = pd.DataFrame(
                    transformed_data, index=df_with_dummies.index
                )

                # now concatenate the original data and the encoded data using pandas
                # Drop the non-onehotencoded categorical columns excluding curr_col.
                df_with_dummies = pd.concat(
                    [df_with_dummies, encoded_data], axis=1
                ).drop(columns=cat_cols_no_curr_col)

            # curr_df_with_dummies = curr_df.copy()
            df_train = df_with_dummies.dropna(
                subset=[curr_col]
            ).copy()  # Target has no nans
            df_test = df_with_dummies[df_with_dummies[curr_col].isnull()].copy()

            # If current column to be imputed is categorical -> encode label
            if curr_col in cat_cols:
                # Encode label
                le = LabelEncoder()
                df_train.loc[:, curr_col] = le.fit_transform(df_train[curr_col])
                self.le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

            # Get best model
            best_model = self.train(df_train, curr_col)

            # Predict values
            preds = pd.Series(
                best_model.predict(df_test.drop(columns=curr_col)),
                index=pd.Index(df_test.index),
            )

            # Filling missing values with predicted values
            result[curr_col] = result[curr_col].fillna(preds)

            # # Map categorical targets back to original labels
            if curr_col in cat_cols:
                result[curr_col] = result[curr_col].map(self.le_name_mapping)

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
            model = xgb.XGBRegressor(eval_metric="rmse")
            scoring = "neg_root_mean_squared_error"
            param_grid = {
                "booster": ["gbtree"],
                "max_depth": [4, 6, 8],
                "alpha": [0, 3],
                "lambda": [1, 3],
                "learning_rate": [0.5],
            }

        elif self.dtype_list[curr_col] == "categorical":
            model = xgb.XGBClassifier(use_label_encoder=False)
            classes = pd.unique(y_train)
            n_classes = len(classes)
            param_grid = {
                "learning_rate": [0.5],
                "max_depth": [4, 6, 8],
                "subsample": [0.5, 1],
                "colsample_bytree": [0.5, 1],
            }

            if n_classes == 2:
                model.set_params(objective="binary:logistic", eval_metric="logloss")
                scoring = "neg_log_loss"

            else:
                model.set_params(
                    objective="multi:softmax",
                    num_class=n_classes,
                    eval_metric="mlogloss",
                )
                scoring = "neg_log_loss"

        grid_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            scoring=scoring,
            n_jobs=2,
            cv=max(2, self.cv),
            refit=True,
            random_state=self.random_seed,
            error_score="raise",
        )

        grid_search.fit(X_train, y_train)

        return grid_search.best_estimator_


def measure_val_error(df, imputer,dtype_list, n_folds=5):
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
    dtype_list : dict
        Dict which shows whether a feature is numeric or categorical.
    n_folds : int, optional
        Number of N-folds to perform, by default 5

    Returns
    -------
    dict
        Mean RMSE error for each column as a dict
        For categorical values use 

    """

    curr_df = df.dropna(axis=0, how="any").copy()
    errors = pd.DataFrame(((curr_df - curr_df) ** 2).mean(axis=0) ** (1 / 2)).T
    # n-fold cross validation
    for i in range(n_folds):
        nans = curr_df.mask(np.random.random(curr_df.shape) < 0.4)
        res = imputer.impute(nans)
        curr_errors = pd.DataFrame(((res - curr_df) ** 2).mean(axis=0) ** (1 / 2)).T
        errors = pd.concat([errors, curr_errors], axis=0)

    return pd.DataFrame(errors.iloc[1:, :].mean(axis=0)).round(2).to_dict()[0]


if __name__ == "__main__":
    pass
    # # Load data
    # path_project = os.path.abspath(os.path.join(__file__, "../../.."))
    # path_processed_data = path_project + "/data/processed/"
    # data = pd.read_csv(path_processed_data + "iris_nans.csv")

    # # nää pitää tehdä inputille
    # dtypes = ["numeric", "numeric", "numeric", "numeric", "categorical"]
    # cols = data.columns
    # dtype_list = dict(zip(cols, dtypes))

    # # Implementoi tämä -> app.py
    # # imp = MySimpleImputer()
    # # res = imp.impute(data)

    # imp = XGBImputer(dtype_list=dtype_list, random_seed=42, verbose=0, cv=1)
    # res = imp.impute(data)
    # print(data.head(10))
    # print(res.head(10))
    # error_ = measure_val_error(data,imp,n_folds=1)
    # print(error_)
    # print(res['target'].value_counts())

    # python -m src.models_fn.imputer_models
