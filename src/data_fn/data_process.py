from sklearn import datasets
import pandas as pd
import numpy as np


def load_sample_iris():
    # import some data to play with
    iris = datasets.load_iris()
    df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                      columns=iris['feature_names'] + ['target'])
    df.loc[df.sample(10, random_state=32).index,
           'sepal length (cm)'] = np.nan
    return df
