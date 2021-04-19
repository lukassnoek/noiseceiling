import numpy as np
import pandas as pd
from tqdm import tqdm
from .core import compute_nc_classification, compute_nc_regression
from .utils import _find_repeats


def run_bootstraps_nc(X, y, n_bootstraps=10, classification=True, kwargs=None):
    """ Runs a number of bootstraps to estimate the variability of the noise ceiling.

    Parameters
    ----------
    X : DataFrame
        A pandas DataFrame with the intended predictors in columns and
        observations in rows
    y : Series
        A pandas Series with the dependent variable (may contain strings
        or integers, as long as it's categorical in nature)
    n_bootstraps : int
        Number of bootstrap iterations to run
    classification : bool
        Whether its a classification model or not
    kwargs : dict
        Arguments to feed to the noise ceiling estimator function
    
    Returns
    -------
    nc : DataFrame
        A Pandas DataFrame with bootstrap results in rows
    """
    if kwargs is None:
        kwargs = {}
    
    if 'progress_bar' in kwargs.keys():
        pb = kwargs['progress_bar']
    else:
        pb = False

    rep_idx, _ = _find_repeats(X, progress_bar=pb)
    n_uniq = np.unique(rep_idx).size
        
    ncs = []
    for _ in tqdm(range(n_bootstraps)):

        samp_idx = np.random.choice(np.unique(rep_idx), size=n_uniq, replace=True)
        X_, y_ = [], []
        for s in samp_idx:
            X_.append(X.loc[rep_idx == s, :])
            y_.append(y.loc[rep_idx == s])

        X_ = pd.concat(X_, axis=0)
        y_ = pd.concat(y_)

        if classification:
            nc = compute_nc_classification(X_, y_, **kwargs)
        else:
            nc = compute_nc_regression(X_, y_, **kwargs)
        ncs.append(nc)
    
    nc = pd.concat(ncs, axis=0)
    return nc
    