from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from datetime import date, datetime
from matplotlib import pyplot as plt



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

