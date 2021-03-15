import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


def compute_noise_ceiling_classification(X, y, use_repeats_only=True, soft=True, per_class=True,
                                         score_func=roc_auc_score, n_bootstraps=0):

    y = y.rename("target")
    classes = y.unique()

    if use_repeats_only:
        rep_idx = X.duplicated(keep=False)
        # Copy to avoid warning
        X = X.copy().loc[rep_idx, :]
        y = y.loc[rep_idx]

    # Make sure index is unique
    X.index = range(X.shape[0])
    y.index = range(X.shape[0])

    # Extract repeats
    X = _find_repeats(X)

    # This needs to happen for the operation below to work
    y = y.to_frame().assign(rep_id=X['rep_id']).set_index('rep_id')

    # Compute the "optimal" predictions:
    opt = (y \
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

    if not soft:
        opt = opt.idxmax(axis=1).to_frame()

    # Remove index name (otherwise duplicate with column name)
    opt.index.name = None

    # Repeat the optimal predictions for all reps
    opt_rep = opt.copy().loc[y.index, :].sort_index()
    y = y.sort_index()  # make sure index aligns

    # Compute noise ceiling
    to_loop = [None] if n_bootstraps == 0 else range(n_bootstraps)
    K = classes.size if per_class else 1
    nc = np.zeros((len(to_loop), K))
    for i, bs in enumerate(to_loop):

        if bs is None:
            y2 = y.copy()
            opt_rep2 = opt_rep.copy()
        else:
            opt_rep2 = opt_rep.copy()
            opt_rep2.index = range(opt_rep2.shape[0])

            # Resample with replacement
            opt_rep2 = opt_rep2.sample(frac=1, replace=True)
            y2 = y.iloc[opt_rep2.index]

        if soft:  # one-hot encode!
            y2 = pd.get_dummies(y2)
        
        if per_class:  # make sure average is set to None
            nc[i, :] = score_func(y2.values, opt_rep2.values, average=None)
        else:  # leave out average, defaults to 'macro'
            try:
                nc[i] = score_func(y2.values, opt_rep2.values)
            except ValueError:
                nc[i] = score_func(y2.values, opt_rep2.values, average='macro')

    if per_class:
        nc = pd.DataFrame(nc, columns=classes)
    else:
        nc = pd.DataFrame(nc, columns=['average'])

    return nc


def compute_noise_ceiling_regression(X, y, use_repeats_only=True, levels=None):

    y = y.rename("target")

    if use_repeats_only:
        rep_idx = X.duplicated(keep=False)
        # Copy to avoid warning
        X = X.copy().loc[rep_idx, :]
        y = y.loc[rep_idx]

    # Make sure index is unique
    X.index = range(X.shape[0])
    y.index = range(X.shape[0])

    # Extract repeats
    X = _find_repeats(X)

    y = y.to_frame().assign(rep_id=X['rep_id'], rep_nr=X['rep_nr'])
    y = y.pivot(index='rep_id', columns='rep_nr', values='rating')
    y = y.loc[:, (~y.isna()).sum(axis=0) > 30]
    corrs = y.corr().to_numpy()
    corrs = corrs[np.triu_indices_from(corrs, k=1)]
    nc_mean = corrs.mean()
    nc_std = corrs.std()


def _find_repeats(X):
    """ Finds repeated rows in a dataframe. """

    # Get indices of repeated trials    
    rep_indices = X.loc[X.duplicated(keep='first'), :].index

    # Save both the rep ID and, per ID, the number of reps
    rep_id = np.zeros(X.shape[0])
    rep_nr = np.zeros(X.shape[0])

    # Loop over rep indices
    for i, rep_idx in enumerate(tqdm(rep_indices)):
        # Ugly way to find repeated rows
        idx = (X.loc[rep_idx, :] == X).all(axis=1).to_numpy()
        rep_id[idx] = i + 1  # store unique index
        rep_nr[idx] = range(idx.sum())  # store number of reps

    # Store in X
    X = X.assign(rep_id=rep_id, rep_nr=rep_nr)

    # Print out some stuff
    nr_of_reps = X.groupby('rep_id').size()
    print(f"Found {np.unique(rep_id).size} repeated trials with an "
          f"average of {nr_of_reps.mean()} repetitions!")
 
    return X


if __name__ == '__main__':
    from sklearn.metrics import recall_score

    df = pd.read_csv('noiseceiling/data/sub-01_task-expressive_ratings.tsv', sep='\t', index_col=0)
    df = df.query("rating_type == 'emotion'").query("rating != 'Geen van allen'")
    y = df['rating']
    #y = df.query("rating_type == 'arousal'")['rating'].astype(float)
    
    X = pd.read_csv('noiseceiling/data/featurespace-AU.tsv', sep='\t', index_col=0)
    X = X.loc[y.index, :]
    nc = compute_noise_ceiling_classification(X, y, soft=False, per_class=False, score_func=recall_score, n_bootstraps=10)
    print(nc)
