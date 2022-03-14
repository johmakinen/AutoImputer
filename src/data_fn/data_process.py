from sklearn import datasets
import pandas as pd
import numpy as np
import os


def create_iris_sample(n_nans=30):
    """Load and create iris dataset with nans.
        Save as csv both the original and the nan version."""

    path_project = os.path.abspath(
        os.path.join(__file__, "../../.."))

    path_raw_data = path_project+'/data/raw/'
    path_processed_data = path_project+'/data/processed/'

    # import some data to play with
    iris = datasets.load_iris()
    df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                      columns=iris['feature_names'] + ['target'])
    nans = df.copy()
    nans.loc[df.sample(n_nans, random_state=32).index,
             'sepal length (cm)'] = np.nan

    df.to_csv(path_processed_data+'iris_original.csv', index=False)
    nans.to_csv(path_processed_data+'iris_nans.csv', index=False)


if __name__ == '__main__':
    create_iris_sample()
