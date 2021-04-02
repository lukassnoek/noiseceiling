import pytest
import numpy as np
import pandas as pd
import os.path as op
from sklearn.metrics import roc_auc_score, recall_score
from noiseceiling import compute_nc_classification, compute_nc_regression
from noiseceiling import reduce_repeats
from noiseceiling.bootstrap import run_bootstraps_nc


def _load_data(classification=True):

    f = op.join('noiseceiling', 'data', 'sub-xx_task-expressive_ratings.csv')
    y = pd.read_csv(f, index_col=0)
    f = op.join('noiseceiling', 'data', 'featurespace-AU.tsv')
    X = pd.read_csv(f, sep='\t', index_col=0)
    
    if classification:
        y = y.query("rating_type == 'emotion'").query("rating != 'Geen van allen'")
        y = y['rating']
    else:
        y = y.query("rating_type == 'arousal'")['rating']

    X = X.loc[y.index, :]
    return X, y


@pytest.mark.parametrize("classification", [False, True])
def test_no_repeats(classification):
    """ Should raise a ValueError. """
    X = pd.DataFrame(np.random.normal(0, 1, size=(100, 5)))
    if classification:
        y = pd.Series(np.random.choice(['a', 'b', 'c'], size=100))
        with pytest.raises(ValueError):
            compute_nc_classification(X, y)
    else:
        y = pd.Series(np.random.normal(0, 1, 100))
        with pytest.raises(ValueError):
            compute_nc_regression(X, y)


@pytest.mark.parametrize("classification", [False, True])
def test_bootstrap_nc(classification):
    """ Should raise a ValueError. """
    
    X, y = _load_data(classification=classification)
    run_bootstraps_nc(X, y, classification=classification)


@pytest.mark.parametrize("use_index", [False, True])
@pytest.mark.parametrize("use_repeats_only", [False, True])
@pytest.mark.parametrize("per_class", [False, True])
def test_nc_classification(use_index, use_repeats_only, per_class):

    X, y = _load_data(classification=True)
    
    compute_nc_classification(
        X, y, use_repeats_only=use_repeats_only, soft=True, per_class=per_class,
        use_index=use_index, score_func=roc_auc_score
    )

@pytest.mark.parametrize("y_type", ["integer", "string"])
@pytest.mark.parametrize("per_class", [False, True])
def test_nc_classification_ytype(y_type, per_class):
    X, y = _load_data(classification=True)
    if y_type == 'integer':
        # Convert strings to integers
        y = pd.Series(pd.get_dummies(y).to_numpy().argmax(axis=1), index=y.index)
    
    nc = compute_nc_classification(
        X, y, use_repeats_only=False, soft=True, per_class=per_class,
        use_index=False, score_func=roc_auc_score
    )


@pytest.mark.parametrize("use_index", [False, True])
@pytest.mark.parametrize("use_repeats_only", [False, True])
def test_nc_regression(use_index, use_repeats_only):

    X, y = _load_data(classification=False)
    compute_nc_regression(
        X, y, use_repeats_only=use_repeats_only,
        use_index=use_repeats_only
    )


@pytest.mark.parametrize("use_index", [False, True])
@pytest.mark.parametrize("classification", [False, True])
def test_reduce_repeats(use_index, classification):

    X, y = _load_data(classification=classification)
    reduce_repeats(X, y, categorical=classification, use_index=use_index)
