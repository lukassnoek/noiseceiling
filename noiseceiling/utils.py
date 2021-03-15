import pandas as pd


def get_percentiles(x, n_percentiles=5):
    return pd.qcut(x, q=n_percentiles, retbins=False, labels=False)
