import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureConcatTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, transformer_list):
        self.transformer_list = transformer_list

    def fit(self, X, y=None):
        for transformer in self.transformer_list:
            transformer.fit(X, y)
        return self

    def transform(self, X, y=None):
        cols = [transformer.transform(X) for transformer in self.transformer_list]
        res = X
        for c in cols:
            res = np.c_[res, c]
        return res

    def get_feature_names(self):
        return X.columns.tolist() + [transformer.get_feature_names(X) for transformer in self.transformer_list]


class DiffTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, base_value=0):
        self.base_value = base_value

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        return np.c_[X.apply(lambda x: self.base_value - x)]

    def get_feature_names(self):
        return str(self.base_value) + '-' + X.columns.tolist()[0]


class TimeDiffTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, base_date=pd.Timestamp.now(), unit='M'):
        self.unit = unit
        self.base_date = base_date

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        return np.c_[
            X.apply(lambda x: pd.Timestamp.now() - pd.Timestamp(x)).astype('timedelta64[{}]'.format(self.unit))]

    def get_feature_names(self):
        return str(self.base_date) + '-' + X.columns.tolist()[0]


class CategoryDropTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cutoff_frequency=.05, other_name='Other'):
        self.cutoff_frequency = cutoff_frequency
        self.other_name = other_name

    def fit(self, X, y=None):
        self.categories, self.counts = np.unique(X, return_counts=True)
        self.good_categories = self.categories[self.counts / X.shape[0] > self.cutoff_frequency]
        return self  # nothing else to do

    def transform(self, X, y=None):
        return np.c_[np.where(~np.isin(X, self.good_categories), 'NU', X)]

    def get_feature_names(self):
        return super.get_feature_names()


class ToLowerCaseTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        return np.c_[np.char.lower(X.astype('|S30'))]

    def get_feature_names(self):
        return X.columns.tolist()

