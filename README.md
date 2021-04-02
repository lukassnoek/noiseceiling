# noiseceiling
Noise ceiling estimation for machine learning models.

## What does it do?
This package contains functionality to estimate a "noise ceiling" for any model that predicts a particular target variable/dependent variable (`y`) from a set of predictors/independent variables (`X`). The noise ceiling is (an estimate of) the upper bound of model performance given the consistency (or, inversely, measurement noise) of the target variable (`y`).

The package uses repeated observations (determined from `X`) to estimate the "variance" of the dependent variable across those repetitions, which in turn forms the basis of the noise ceiling estimate. If your data (`X`) does not contain repeated observations, the noise ceiling will be 1 (which is probably not very informative). 

## API
The `noiseceiling` package contains two main functions, `compute_nc_classification` and `compute_nc_regression`, which (as their name suggests) computes the noise ceiling for classification models (dependent variable is categorical) and regression models (dependent variable is continuous) respectively.

Note that both functions expect two (mandatory) arguments:

* `X`: a *pandas* `DataFrame` with predictors/features in columns and observations in rows;
* `y`: a *pandas* `Series` with observations in rows

In addition, you can estimate the variability of the noise ceiling using the `run_bootstraps_nc` function, which accepts an addition keyword, `classification` (a bool), which indicates whether a categorical (`True`) or continuous (`False`) noise ceiling should be estimated.