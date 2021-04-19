import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from .utils import _find_repeats, _check_Xy, _use_index, _y2opt


def compute_nc_classification(X, y, use_repeats_only=True, soft=True, per_class=True,
                              use_index=False, score_func=roc_auc_score, progress_bar=False):
    """ Computes a noise ceiling for classification models (i.e., when
    the dependent variable is categorical).

    Parameters
    ----------
    X : DataFrame
        A pandas DataFrame with the intended predictors in columns and
        observations in rows
    y : Series
        A pandas Series with the dependent variable (may contain strings
        or integers, as long as it's categorical in nature)
    use_repeats_only : bool
        Whether to compute the noise ceiling on the repeated trials *only*,
        or whether to use all trials
    soft : bool
        Whether to use soft labels (probabilities) or hard (discrete) labels
    per_class : bool
        Whether to compute the noise ceiling for each class of the dependent
        variable separately
    use_index : bool
        In determining which trials are repeats, use the index instead of the
        actual rows (not useful for 99% of the usecases)
    score_func : function
        Scikit-learn style function that computes the performance of the 
        optimal model, i.e., the noise ceiling
    progress_bar : bool
        Whether to show a progress bar when finding repeats

    Returns
    -------
    nc : DataFrame
        A pandas DataFrame with the noise ceiling (if per_class is True, then
        the columns represent the different classes).
    """
    X_, y = _check_Xy(X, y, categorical=True, use_index=use_index)
    if use_index:
        X_ = _use_index(X)
    
    classes = sorted(y.unique())

    if use_repeats_only:
        # Remove all trials that have not been 
        rep_idx = X_.duplicated(keep=False).to_numpy()
        X_, X, y = X_.loc[rep_idx, :], X.loc[rep_idx, :], y.loc[rep_idx]

    # Find repeated trials (keep all!)
    rep_id, _ = _find_repeats(X_, progress_bar=progress_bar)

    # Add repetition ID to y so we can use groupby
    y = y.to_frame().assign(rep_id=rep_id)
    opt = _y2opt(y)

    # Remove index name (otherwise duplicate with column name)
    opt.index.name = None

    # Repeat the optimal predictions for all reps
    opt_rep = opt.loc[y['rep_id'], :].set_index(y.index)
    
    # Do not need rep_id anymore
    y = y.drop('rep_id', axis=1)
    y = pd.get_dummies(y)
    
    if not soft:
        raise NotImplementedError
        #y = y.idxmax(axis=1)
        #opt_rep = opt_rep.idxmax(axis=1)
    
    if per_class:
        nc = score_func(y.values, opt_rep.values, average=None)
        nc = pd.DataFrame([nc], columns=classes)
    else:
        nc = score_func(y.values, opt_rep.values, average='macro')
        nc = pd.DataFrame([nc], columns=['average'])
    
    return nc


def compute_nc_regression(X, y, use_repeats_only=True, use_index=False,
                          progress_bar=False):
    """ Computes a noise ceiling for regression models (i.e., when
    the dependent variable is continuous). Note that the noise ceiling
    is always expressed as an R2 score (i.e., explained variance).

    Parameters
    ----------
    X : DataFrame
        A pandas DataFrame with the intended predictors in columns and
        observations in rows
    y : Series
        A pandas Series with the dependent variable (must be continuous)
    use_repeats_only : bool
        Whether to compute the noise ceiling on the repeated trials *only*,
        or whether to use all trials
    use_index : bool
        In determining which trials are repeats, use the index instead of the
        actual rows (not useful for 99% of the usecases)
    progress_bar : bool
        Whether to show a progress bar when finding repeats

    Returns
    -------
    nc : float
        The maximum possible R2 score given the variance in the dependent
        variable
    """

    X_, y = _check_Xy(X, y, categorical=False, use_index=use_index)
    if use_index:
        X_ = _use_index(X)
    
    if use_repeats_only:
        # Remove all trials that have not been 
        rep_idx = X_.duplicated(keep=False).to_numpy()
        X_, X, y = X_.loc[rep_idx, :], X.loc[rep_idx, :], y.loc[rep_idx]

    rep_id, _ = _find_repeats(X_, progress_bar=progress_bar)
    y = y.to_frame().assign(rep_id=rep_id)
    
    # Compute total sums of squares (TSS) and residual sums of squares (RSS)
    TSS = ((y['rating'] - y['rating'].mean()) ** 2).sum()
    RSS = y.groupby('rep_id').agg(lambda x: np.sum((x - x.mean()) ** 2)).sum()
    nc = 1 - (RSS / TSS)
    return nc
