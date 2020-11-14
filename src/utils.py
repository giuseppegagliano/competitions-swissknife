from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from datetime import date, datetime
from matplotlib import pyplot as plt


class BiColsAggTransformer(BaseEstimator, TransformerMixin):
    '''
    Applies the op aggregation function between the columns col1_name and col2_name
    '''
    def __init__(self, col1_name='col1', col2_name='col2', colres_name='', op='sum'):
        self.col1_name = col1_name
        self.col2_name = col2_name
        self.colres_name = colres_name
        self.op = op
        
    def fit(self, X, y=None):
        if self.colres_name == '':
            self.colres_name = self.col1_name + '_' + self.col2_name + '_' + self.op
        self.columns = list(X.columns)
        self.columns.append(self.colres_name)
        return self
    
    def transform(self, X, y=None):
        if self.op == 'sum':
            Y = X.copy()
            Y[self.colres_name] = Y[self.col1_name] + Y[self.col2_name]
            return Y
        return self


class SeparateStringTransformer(BaseEstimator, TransformerMixin):
    '''
    Separate a string column named col_name, generating a prefix column with the first str_len1 characters
    and a suffix columns taking the last str_len2 characters.
    colres_name1 and colres_name2 are the respective names
    '''
    def __init__(self, col_name, str_len1, str_len2, colres_name1='', colres_name2=''):
        self.col_name = col_name
        self.str_len1 = str_len1
        self.str_len2 = str_len2
        self.colres_name1 = colres_name1
        self.colres_name2 = colres_name2
        
    def fit(self, X, y=None):
        if self.colres_name1 == '':
            self.colres_name1 = self.col_name + '_1'
        if self.colres_name2 == '':
            self.colres_name2 = self.col_name + '_2'
        return self
    
    def transform(self, X, y=None):
        Y = X.copy()
        Y[self.colres_name1] = Y[self.col_name].apply(lambda x: x[:self.str_len1]).astype('category')
        Y[self.colres_name2] = (Y[self.col_name]
                                .apply(lambda x: x[self.str_len1:self.str_len1+self.str_len2]).astype('category'))
        return Y
    
    
class CategoryDropTransformer(BaseEstimator, TransformerMixin):
    '''
    Merges the categories with frequency lower than cutoff_frequency in a categorical column with
    more than 2 categories. The merged categories take the name from other_name
    '''
    def __init__(self, cutoff_frequency=.01, other_name='other'):
        self.cutoff_frequency = cutoff_frequency
        self.other_name = other_name
        
    def fit(self, X, y=None):
        self.categories, self.counts = np.unique(X, return_counts=True)
        if self.categories.shape[0] > 2:
            self.good_categories = self.categories[self.counts/X.shape[0] > self.cutoff_frequency]
        else:
            self.good_categories = self.categories
        return self
    
    def transform(self, X, y=None):
        Y = X.copy()
        return np.c_[np.where(~np.isin(Y, self.good_categories), self.other_name, Y)]
    
    def get_feature_names(self):
        return super.get_feature_names()
    

class PreprocessDatesTransformer(BaseEstimator, TransformerMixin):
    '''
    This transformer takes a dataframe and a list of date formatted columns as %d/%B,
    and infers the year for the columns
    '''
    def __init__(self, date_cols):
        self.date_cols = date_cols
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        Y = X.copy()
        for c in self.date_cols:
            Y[c] = (Y[c]
                        .apply(lambda x: x.replace('-', '/'))
                        .apply(lambda x: datetime.strptime(x, '%d/%B')))
            
        n_cols = len(self.date_cols)
        for i in range(n_cols-1):
            for j in range(i+1,n_cols):
                Y[self.date_cols[j]] = Y.apply(lambda x:
                                               x[self.date_cols[j]].replace(year=x[self.date_cols[i]].year+1)
                                               if x[self.date_cols[i]] > x[self.date_cols[j]]
                                               else x[self.date_cols[j]], axis=1)
        return Y
    
    
class PeakDaysTransformer(BaseEstimator, TransformerMixin):
    '''
    This transformer takes a time series and retrieves a boolean with True if the count of ids 
    for the specific day is greater than the average of 3 days before and 4 after
    '''
    def __init__(self, id_col, target_col):
        self.id_col = id_col
        self.target_col = target_col
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        count_col = 'COUNT'
        mavg_col = 'MAVG'
        date_col = 'DATE'
        res_col = 'PEAK_DAY'
        Y = X.copy()
        tmp = Y[[self.id_col, self.target_col]].copy().melt(id_vars=self.id_col, var_name=count_col, value_name=date_col)
        tmp = (pd.get_dummies(tmp, columns=[count_col], prefix=None)
             .groupby(date_col)
             .sum()
             .reset_index())
        if self.id_col in tmp.columns:
            tmp = tmp.drop(self.id_col, axis=1)

        tmp[mavg_col + '_' + self.target_col] = (tmp
                                          .rolling(7)
                                          .apply(np.mean, raw=True)
                                          .shift(-3)
                                          .fillna(method='backfill')
                                          .fillna(method='ffill')
                                        )
        tmp[res_col + '_' + self.target_col] = (tmp[mavg_col + '_' + self.target_col] <
                                              tmp[count_col + '_' + self.target_col] )
        tmp = tmp.drop([mavg_col + '_' + self.target_col, count_col + '_' + self.target_col], axis=1)
        return Y.join(tmp.set_index(date_col), on=self.target_col)


class DaysDiffTransformer(BaseEstimator, TransformerMixin):
    '''
    Computes the difference in terms of days between 2 date columns
    '''
    def __init__(self, date_column_1, date_column_2, colres_name=''):
        self.date_column_1 = date_column_1
        self.date_column_2 = date_column_2
        self.colres_name = colres_name
        
    def fit(self, X, y=None):
        if self.colres_name == '':
            self.colres_name = self.date_column_1 + '_' + self.date_column_2 + '_DIFF'
        return self
    
    def transform(self, X, y=None):
        Y = X.copy()
        Y[self.colres_name] = Y[self.date_column_2].sub(Y[self.date_column_1], axis=0).dt.days
        return Y
    
    
class CategoryToLowerCaseTransformer(BaseEstimator, TransformerMixin):
    '''
    Transforms a string column to lower case
    '''
    def __init__(self):
        self
        
    def fit(self, X, y=None):
        return self  # nothing else to do
    
    def transform(self, X, y=None):
        Y = X.copy()
        return np.c_[np.char.lower(Y.astype('|S30'))]
    
    
class ManualFeatureSelector(TransformerMixin):
    """
    Transformer for manual selection of features using sklearn style transform method. 
    (https://stackoverflow.com/questions/28296670/remove-a-specific-feature-in-scikit-learn)
    """
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.features]


def plot_grid_search(gs_dict, params, logscale=False):
    res_df = pd.DataFrame(gs_dict.cv_results_)
    for p in params:
        adf = res_df.sort_values('param_' + p)
        plt.plot(adf['param_' + p], adf['mean_test_score'])
        plt.grid()
        plt.xticks(adf['param_' + p].values.tolist(), rotation=90)
        if logscale:
            plt.xscale('log')
        plt.title(p)
        plt.show()