import numpy as np
import pandas as pd


def get_percentiles(x, n_percentiles=5):
    return pd.qcut(x, q=n_percentiles, retbins=False, labels=False)
    

def reduce_repeats(X, y, categorical=False, use_index=False):
    """ Removes the repeated trials from the target (y) and
    design matrix (X) by taking the mean (if continuous) or
    argmax (if categorical) across trial repetitions. """

    X_, y = _check_Xy(X, y, categorical=categorical, use_index=use_index)
    if use_index:
        X_ = _use_index(X)
    
    # Find repeated trials (keep all!)
    rep_id, X_reduced = _find_repeats(X_)

    # Add repetition ID to y so we can use groupby
    y = y.to_frame().assign(rep_id=rep_id)
    
    ### FROM HERE ON, IT IS SPECIFIC TO THIS FUNCTION
    if categorical:
        opt = _y2opt(y)
        # opt is a DataFrame with classes in columns, repetition IDs
        # in rows and the fraction of class labels as values
        # To get the reduced labels, simply take the argmax
        y_reduced = opt.idxmax(axis=1)
    else:        
        # If y is continuous, simply average the values
        # across trials with the same repetition ID
        y_reduced = y.groupby('rep_id').mean()
        y_reduced = y_reduced.iloc[:, 0]  # series to index

    # Make sure indices match
    y_reduced.index = X_reduced.index
    if use_index:  # stupid but necessary
        # Want our original features (X) back, not the ones we used
        # to determine repeats (X_)
        uniq_idx = X_.duplicated(keep='first')
        X_reduced = X.loc[~uniq_idx.to_numpy(), :]

    # Check whether indices align & X does not contain repeats anymore
    assert(y_reduced.index.equals(X_reduced.index))
    assert((~X_reduced.duplicated()).any())
    
    return X_reduced, y_reduced


def _check_Xy(X, y, categorical=False, use_index=False):
    """ Some preemptive checks. """
    nx, ny = X.shape[0], y.shape[0]
    if nx != ny:
        raise ValueError(f"Number of samples in X ({nx}) does not match "
                         f"the number of samples in y ({ny})!")

    if not categorical:
        # Make sure y are floating point values
        # (crashes when "object")
        y = y.astype(float)
    else:
        y = y.rename("target")

    if not use_index:
        y.index = range(y.size)
        X.index = range(X.shape[0])

    return X, y


def _use_index(X):
    # Use index instead of values
    X_ = pd.get_dummies(X.reset_index()['index'])
    X_ = X_.set_index(X.index)
    return X_


def _y2opt(y_rep):
    # Compute the fraction of ratings for each repetition ID
    # for each class
    opt = (y_rep \
        # 1. Per unique index, compute the count per class
        .groupby(['rep_id', 'target']).size() \
        .groupby(level=0) \
        # 2. Divide counts by sum (per unique index)
        .apply(lambda x: x / x.sum()) \
        # 3. Unstack to move classes to columns (long -> wide)
        .unstack(level=1) \
        # 4. Fill NaNs (no ratings) with 0
        .fillna(0)
    )
    return opt


def _find_repeats(X):

    # Within all repeated trials, get all unique trial configs
    X_uniq = X.drop_duplicates(keep='first')
    rep_id = np.zeros(X.shape[0])  # store indices

    # Loop over unique rows to see which match other rows!
    for i in range(X_uniq.shape[0]):
        # ALL columns should match
        idx = (X_uniq.iloc[i, :] == X).all(axis=1).to_numpy()
        rep_id[idx] = i + 1  # each repeated trial gets a "repetition ID"

    # After the loop, there should be no trials with ID 0 ...
    if np.sum(rep_id == 0) != 0:
        raise ValueError("Something went wrong in determining repeats ...")

    return rep_id, X_uniq


if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('noiseceiling/data/sub-01_task-expressive_ratings.tsv', sep='\t', index_col=0)
    df = df.query("rating_type == 'emotion'").query("rating != 'Geen van allen'")
    y = df['rating']
    #y = df.query("rating_type == 'arousal'")['rating'].astype(float)
    
    X = pd.read_csv('noiseceiling/data/featurespace-AU.tsv', sep='\t', index_col=0)
    X = X.loc[y.index, :]

    reduce_repeats(X, y, categorical=True, use_index=True)
    #reduce_repeats(X, y, categorical=False, use_index=False)