import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# ---------- DATES TRANSFORMERS ----------

class CustomPreprocDatesTransformer(BaseEstimator, TransformerMixin):
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

class TimeDiffFromBaseTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, base_date=pd.Timestamp.now(), unit='M'):
        self.unit = unit
        self.base_date = base_date

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        return np.c_[
            X.apply(lambda x: pd.Timestamp.now() - pd.Timestamp(x)).astype('timedelta64[{}]'.format(self.unit))]

    def get_feature_names(self):
        return [str(self.base_date) + '-' + self.cols_]

class DatesDiffTransformer(BaseEstimator, TransformerMixin):
    '''
    Computes the difference between 2 date columns
    '''
    def __init__(self, date_column_1, date_column_2, colres_name='', granularity='days'):
        self.date_column_1 = date_column_1
        self.date_column_2 = date_column_2
        self.colres_name = colres_name
        self.granularity = granularity
        
    def fit(self, X, y=None):
        if self.colres_name == '':
            self.colres_name = self.date_column_1 + '_' + self.date_column_2 + '_DIFF'
        return self
    
    def transform(self, X, y=None):
        X_ = X.copy()
        if self.granularity == 'days':
            X_[self.colres_name] = X_[self.date_column_2].sub(X_[self.date_column_1], axis=0).dt.days
        elif self.granularity == 'hours':
            X_[self.colres_name] = X_[self.date_column_2].sub(X_[self.date_column_1], axis=0).dt.hours
        elif self.granularity == 'minutes':
            X_[self.colres_name] = X_[self.date_column_2].sub(X_[self.date_column_1], axis=0).dt.minutes
        elif self.granularity == 'seconds':
            X_[self.colres_name] = X_[self.date_column_2].sub(X_[self.date_column_1], axis=0).dt.seconds
        else:
            raise 'Granularity not found'
        return X_
    

# ---------- GENERAL UTILITY TRANSFORMERS ----------
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

class FeatureConcatTransformer(BaseEstimator, TransformerMixin):
    '''
    Takes a list of transformers and concatenates the result of each to the initial dataset

    Parameters
    ----------
    transformer_list : list of sklearn transformer as described in https://scikit-learn.org/stable/data_transforms.html

    '''

    def __init__(self, transformer_list):
        self.transformer_list = transformer_list

    def fit(self, X, y=None):
        """
        Fit Transformers
        Applies the fit function to each transformer in `transformer_list`
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input dataframe
        Returns
        -------
        self
            Fitted transformer.
        """
        for transformer in self.transformer_list:
            transformer.fit(X, y)
        return self

    def transform(self, X, y=None):
        """
        Transformation phase
        Applies the transformations to the dataframe
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input dataframe
        Returns
        -------
        T : array-like of shape (n_samples, n_features)
            Returns the original dataframe with the additional columns for the trasformations.
        """
        X_ = X.copy()
        cols = np.array([transformer.transform(X_) for transformer in self.transformer_list]).squeeze()
        return np.c_[X_, cols]

    def get_feature_names(self, X):
        """
        Names of features
        Returns the list of names for the columns in the output dataframe
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input dataframe
        Returns
        -------
        columns : array-like of shape (n_columns)
            List of column names for the output dataframe
        """
        return X.columns.tolist() + [transformer.get_feature_names(X) for transformer in self.transformer_list]

class DiffFromBaseTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, base_value=0):
        self.base_value = base_value

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        self.cols_ = X.columns
        X_ = X.copy()
        return np.c_[X_.apply(lambda x: self.base_value - x)]

    def get_feature_names(self):
        return [str(self.base_value) + '-' + self.cols_]

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

# ---------- STRING TRANSFORMERS ----------
class ToLowerCaseTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        X_ = X.copy()
        self.feature_names_ = X.columns
        return np.c_[np.char.lower(X_.astype('|S30'))]

    def get_feature_names(self):
        return self.feature_names_

class SplitStringToDummiesTransformer(BaseEstimator, TransformerMixin):
    
    '''
    Split a column containing a list, into multiple boolean columns
    
    example: 
    SplitStringToDummiesTransformer(col='test', sep=', ').fit_transform(pd.DataFrame({'test': ['a, b, c', 'd, e, f', 'a, ']}))

    '''
    def __init__(self, col, sep=', ', cols_to_skip=['']):
        self.col = col
        self.sep = sep
        self.cols_to_skip = cols_to_skip
        
    def fit(self, X, y=None):
        self.dummies_ = list(set(np.unique(
            X[self.col]
            .str.split(self.sep, expand=True)
            .dropna()
            .values
            .ravel('K')
        )) - set(self.cols_to_skip))
        return self
    
    def transform(self, X, y=None):
        X_ = X.copy()
        X_[self.dummies_] = False
        col_list = set(
            np.unique(
                X_[self.col]
                .str
                .split(self.sep, expand=True)
                .dropna()
                .values
                .ravel('k'))
                      ) - set(self.cols_to_skip)
        
        for c in col_list:
            X_[c] = X_[self.col].str.split(self.sep).apply(lambda x: c in x)
        X_ = X_.drop([self.col], axis=1)
        return X_

class SplitStringByLengthTransformer(BaseEstimator, TransformerMixin):
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
   

# ---------- TIME SERIES ------------

class LastNValuesAverage(BaseEstimator, TransformerMixin):
    '''
    This transformer computes the average of the last n values for the value_col grouped by group_cols
    and ordered by order_col. It ignores the null values in the calculation
    '''
    def __init__(self, n, order_col, value_col, group_cols):
        self.n = n
        self.order_col = order_col
        self.value_col = value_col
        self.group_cols = group_cols
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_ = X.copy()

        for i in range(self.n):
            X_['AMOUNT_GBP_LAG_' + str(i)] = (X_.sort_values(by=self.order_col, ascending=True)
                                  .groupby(self.group_cols)[self.value_col]
                                  .shift(i)
                                 )
        X_['AMOUNT_GBP_LAST_' + str(self.n) + '_AVG'] = X_[['AMOUNT_GBP_LAG_' +
                                                          str(i) for i in range(self.n)]].mean(axis=1)
        try:
            X_ = X_.drop(['AMOUNT_GBP_LAG_' + str(i) for i in range(self.n)], axis=1)
        except:
            print('cannot remove temporary cols')
        return X_

class AverageInterEventDatesTransformer(BaseEstimator, TransformerMixin):
    '''
    Computes the average time difference in seconds among the datetime events specified in value_col
    for each group specified by group_cols
    '''
    def __init__(self, value_col, group_cols):
        self.value_col = value_col
        self.group_cols = group_cols
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_ = X.copy()
        X_['LAST_' + self.value_col] = (X_.sort_values(by=self.value_col, ascending=True)
                                        .groupby(self.group_cols)[self.value_col]
                                        .shift(1)
                                       )
        X_[self.value_col + '_DIFF'] = (X_['LAST_' + self.value_col] - X[self.value_col]).dt.seconds
        res_series = (X_
                      .groupby(self.group_cols)[self.value_col + '_DIFF']
                      .mean()
                      .rename(self.value_col + '_AVG_DIFF')
                     )
        X_ = X_.drop(['LAST_' + self.value_col, self.value_col + '_DIFF'], axis=1)
        return pd.merge(X_, res_series, on=self.group_cols, how='left')

class PivotPercentageTransformer(BaseEstimator, TransformerMixin):
    '''
    Computes the for each subpopulation and a categorical column the percentage of rows for each category
    e.g. for each user the percentage of transactions of a certain type
    '''
    def __init__(self, value_col, group_cols, category_col):
        self.value_col = value_col
        self.group_cols = group_cols
        self.category_col = category_col
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_ = X.copy()
        pivoted = X_.pivot_table(values=self.value_col, index=self.group_cols, columns=self.category_col,
                      aggfunc=lambda x: len(x.unique()), margins=True).fillna(.0)
        pivoted = pivoted.div(pivoted.All, axis=0).reset_index()[:-1]
        pivoted = pivoted.drop('All', axis=1)
        pivoted.columns = [c + '_PERC' if c not in self.group_cols else c for c in pivoted.columns]
        return pd.merge(X_, pivoted, on=self.group_cols, how='left')





    