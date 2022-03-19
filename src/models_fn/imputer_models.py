from pathlib import Path

print("Running" if __name__ == "__main__" else "Importing", Path(__file__).resolve())

from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from ..data_fn.data_process import replace_infs
import pandas as pd
import numpy as np
import warnings


warnings.filterwarnings("ignore")
import xgboost as xgb


class MySimpleImputer:
    """ A Class for the simple imputer.
    """

    def __init__(
        self, dtype_list, strategy="mean",
    ):
        """Initialise imputer

        Parameters
        ----------
        dtype_list: dict
            A dict that shows what type of data each column has:
            {column_name_1:'numeric',column_name_2:'categorical',...}
        strategy : str, optional
            What strategy to use, by default 'mean'
            Other options: mean, median or most_frequent.
        """
        self.strategy = strategy
        self.dtype_list = dtype_list

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
            Categorical features will always be imputed with "most_frequent"
        """
        if df.empty:
            return df

        idf = replace_infs(df.copy())
        num_cols = [k for k in self.dtype_list if self.dtype_list[k] == "numeric"]
        cat_cols = [col for col in idf.columns if col not in num_cols]

        if num_cols:
            imp_num = SimpleImputer(missing_values=np.nan, strategy=self.strategy)
            idf.loc[:, num_cols] = pd.DataFrame(
                imp_num.fit_transform(replace_infs(idf[num_cols])), index=idf.index,
            ).values

        if cat_cols:
            imp_cat = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
            idf.loc[:, cat_cols] = pd.DataFrame(
                imp_cat.fit_transform(replace_infs(idf[cat_cols])), index=idf.index,
            ).values

        idf.columns = df.columns
        idf.index = df.index
        return idf


class XGBImputer:
    """Imputing class that uses XGBoost for each column that has missing values.
    """

    def __init__(self, dtype_list, random_seed=42, verbose=0, cv=2):
        """Initialise imputer

        Parameters
        ----------
        dtype_list : dict
            A dict that shows what type of data each column has:
            {column_name_1:'numeric',column_name_2:'categorical',...}
        random_seed : int, optional
            , by default 42
        verbose : int, optional
            , by default 0
        cv : int, optional
            Number of cross-validation folds used in hyperparameter optimization, by default 2
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
        For categorical values use ...

    """
    curr_df = df.copy()
    curr_df = replace_infs(curr_df)
    curr_df = curr_df.dropna(axis=0,how='any')


    if curr_df.empty:
        return dict(zip(curr_df.columns, [0] * len(curr_df.columns)))

    errors = pd.DataFrame(columns=curr_df.columns)
    # n-fold cross validation
    for i in range(n_folds):
        fold_error = pd.DataFrame(0, columns=curr_df.columns, index=range(1))

        nans = curr_df.mask(np.random.random(curr_df.shape) < 0.4)
        res = imputer.impute(nans)

        for curr_col in curr_df.columns:
            if imputer.dtype_list[curr_col] == "numeric":
                fold_error.loc[0, curr_col] = mean_squared_error(
                    y_true=curr_df[curr_col].values,
                    y_pred=res[curr_col].values,
                    squared=False,
                )
            elif imputer.dtype_list[curr_col] == "categorical":
                # F1 = 2 * (precision * recall) / (precision + recall)
                if len(pd.unique(curr_df[curr_col])) > 2:
                    average = "micro"
                else:
                    average = "binary"
                fold_error.loc[0, curr_col] = f1_score(
                    y_true=curr_df[curr_col].values,
                    y_pred=res[curr_col].values,
                    average=average,
                )

        errors = pd.concat([errors, fold_error], axis=0)

    return pd.DataFrame(errors.mean(axis=0)).round(2).to_dict()[0]


if __name__ == "__main__":
    pass

