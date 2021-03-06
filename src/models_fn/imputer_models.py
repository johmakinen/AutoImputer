from multiprocessing.sharedctypes import Value
from pathlib import Path

print("Running" if __name__ == "__main__" else "Importing", Path(__file__).resolve())

from sklearn.impute import SimpleImputer

# from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
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

        # Replace infinities with nan
        idf = replace_infs(df.copy())

        num_cols = [k for k in self.dtype_list if self.dtype_list[k] == "numeric"]
        cat_cols = [col for col in idf.columns if col not in num_cols]

        column_trans = ColumnTransformer(
            [
                (
                    "imp_col1",
                    SimpleImputer(strategy=self.strategy, verbose=100),
                    num_cols,
                ),
                (
                    "imp_col2",
                    SimpleImputer(strategy="most_frequent", verbose=100),
                    cat_cols,
                ),
            ],
            remainder="passthrough",
        )

        # Columnstransformer doesn't keep the original column order...
        extracted_cols = (
            column_trans.transformers[0][2] + column_trans.transformers[1][2]
        )
        transformed_data = column_trans.fit_transform(idf)

        res = pd.DataFrame(transformed_data, index=idf.index, columns=extracted_cols,)

        return res.reindex(
            columns=idf.columns
        )  # Reindex to match the original column order


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

        # We want to keep separate the results and the current data.
        # This is because we dont want our imputations of one column
        # to affect the imputations of other columns.
        # This can be changed later if we want such behaviour.

        curr_df = replace_infs(df.copy())
        result = curr_df.copy()

        nan_cols = curr_df.columns[curr_df.isna().any()].tolist()
        cat_cols = [
            col for col in curr_df.columns if self.dtype_list[col] == "categorical"
        ]

        for curr_col in nan_cols:

            cat_cols_no_curr_col = list(filter(lambda x: x != curr_col, cat_cols))
            df_with_dummies = curr_df.copy()

            if cat_cols_no_curr_col:

                # Dummify categorical columns
                # Here we do not care that much about data leaks, as train set is separated afterwards for GridSearch.
                oe = OneHotEncoder(sparse=False, handle_unknown="ignore", dtype="int")

                # Exclude curr_col
                transformed_data = oe.fit_transform(
                    df_with_dummies[cat_cols_no_curr_col]
                )

                # the above transformed_data is an array so convert it to dataframe
                encoded_data = pd.DataFrame(
                    transformed_data, index=df_with_dummies.index
                ).add_prefix("dummy_col_")

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

                # If there are rare (<5%) classes, dont use them for now...
                # This messes with Kfold splits even if using stratifiedKfold
                # val_counts = df_train[curr_col].value_counts(normalize=True) < 0.01
                # keep_classes = [
                #     col for col in val_counts.index.tolist() if not val_counts[col]
                # ]

                # df_train = df_train[df_train[curr_col].isin(keep_classes)]

                # Encode label
                le = LabelEncoder()
                df_train.loc[:, curr_col] = le.fit_transform(df_train[curr_col])
                self.le_name_mapping = dict(zip(le.transform(le.classes_), le.classes_))

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
                result[curr_col] = result[curr_col].replace(self.le_name_mapping)
        return result

    def train(self, df, curr_col):
        """Train XGBoost model.

        Parameters
        ----------
        df : pd.DataFrame
            Data to be imputed (training data)
        curr_col : _type_
            What column is currently being imputed

        Returns
        -------
        xgboost.XGBRegressor() or xgboost.Classifier()
            Best model found for current column data
        """
        # For now, we do not use hyperparamter tuning as Streamlit's resources do not handle massive computation
        # On personal laptop hyperparameter tuning works well.

        X_train, y_train = df.drop(curr_col, axis=1), df[curr_col]

        # Choose type of model:
        if self.dtype_list[curr_col] == "numeric":
            model = xgb.XGBRegressor(eval_metric="rmse", n_estimators=100)
            # scoring = "neg_root_mean_squared_error"
            # param_grid = {
            # "learning_rate": [1],
            # "booster": ["gbtree"],
            # "max_depth": [4, 6],
            # "alpha": [1],
            # "lambda": [1],
            # }

        elif self.dtype_list[curr_col] == "categorical":

            model = xgb.XGBClassifier(use_label_encoder=False, n_estimators=100)
            classes = pd.unique(y_train)
            n_classes = len(classes)
            # param_grid = {
            # "learning_rate": [1],
            # "max_depth": [4, 6],
            # "subsample": [0.5, 1],
            # "colsample_bytree": [0.5, 1],
            # }

            # I dont like that objectives logloss and mlogloss are used in xgb but f1's are used in grid search
            # Change this in the future, for now it is ok.
            if n_classes == 2:
                model.set_params(objective="binary:logistic", eval_metric="logloss")
                # scoring = "f1"

            else:
                model.set_params(
                    objective="multi:softmax",
                    num_class=n_classes,
                    eval_metric="mlogloss",
                )
                # scoring = "f1_micro"

        # grid_search = GridSearchCV(
        #     estimator=model,
        #     param_grid=param_grid,
        #     scoring=scoring,
        #     n_jobs=1,  # <- Streamlit's resources are very limited...
        #     cv=max(2, self.cv),
        #     refit=True,
        #     error_score="raise",
        # )

        # grid_search.fit(X_train, y_train)
        # return grid_search.best_estimator_
        model.fit(X_train, y_train, verbose=False)

        # model.fit(X_train, y_train,
        #     eval_set=[(X_train, y_train), (X_val, y_val)],
        #     early_stopping_rounds =10,
        #     verbose=False)

        # evals_result = model.best_score

        return model


def measure_val_error(df, imputer, n_folds=3):
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
        Number of N-folds to perform, by default 3

    Returns
    -------
    dict
        Mean RMSE error for each column as a dict
        For categorical values use ...

    """

    curr_df = df.copy()
    curr_df = replace_infs(curr_df)
    curr_df = curr_df.dropna(axis=0, how="any")

    # Not enough data for validation
    if curr_df.empty or (curr_df.shape[0] < 10):
        return dict(zip(curr_df.columns, [0] * len(curr_df.columns)))

    else:
        errors = pd.DataFrame(columns=curr_df.columns)
        # n-fold cross validation
        for i in range(n_folds):
            fold_error = pd.DataFrame(0, columns=curr_df.columns, index=range(1))

            nans = curr_df.mask(np.random.random(curr_df.shape) < 0.4)
            nan_cols = nans.columns[nans.isna().any()].tolist()
            res = imputer.impute(nans)

            for curr_col in nan_cols:
                if imputer.dtype_list[curr_col] == "numeric":
                    fold_error.loc[0, curr_col] = mean_squared_error(
                        y_true=curr_df[curr_col].values,
                        y_pred=res[curr_col].values,
                        squared=False,
                    )
                elif imputer.dtype_list[curr_col] == "categorical":
                    labels = pd.unique(curr_df[curr_col])

                    if len(labels) > 2:
                        average = "micro"
                        try:
                            fold_error.loc[0, curr_col] = f1_score(
                                y_true=curr_df[curr_col].values,
                                y_pred=res[curr_col].values,
                                average=average,
                            )
                        except ValueError:
                            raise ValueError(
                                f'Probably had categorical dtype for numeric data. Please check your column type selection for column "{curr_col}".'
                            )
                    else:
                        average = "binary"
                        try:
                            fold_error.loc[0, curr_col] = f1_score(
                                y_true=curr_df[curr_col].values,
                                y_pred=res[curr_col].values,
                                average=average,
                                pos_label=labels[
                                    0
                                ],  # <- binary classification needs pos label or 0/1 data... --> WETWET that we cant really get rid off.
                            )
                        except ValueError:
                            raise ValueError(
                                f'Probably had categorical dtype for numeric data. Please check your column type selection for column "{curr_col}".'
                            )
            errors = pd.concat([errors, fold_error], axis=0)

        return pd.DataFrame(errors.mean(axis=0)).round(2).to_dict()[0]


if __name__ == "__main__":
    pass

