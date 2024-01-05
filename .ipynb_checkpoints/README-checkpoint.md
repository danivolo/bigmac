# BIGMAC (BuildIng Gradually My ALErCE Clone)
## Introduction
This project aims to build a clone of the ALErCE broker lightcurve classifier described in [Sanchez–Saez (2020)](dx.doi.org/10.3847/1538-3881/abd5c1). 

The classifier uses 152 features (though 183 are listed in the dataset) computed from the lightcurves in the red and green bands to distinguish ZTF alert events into 15 classes.
The classification is performed in two steps: first, the signals are divided into transient (supernovae), stochastic (active galactic nuclei/cataclysmic variables)
and periodic (variable stars), then each subgroup is classified indipendently into 4-6 classes. There are thus 4 different classifiers.

The authors found the best performance using a (Balanced) Random Forest (BRF) classifier, implemented with the imbalanced-learn python package.
A Gradient Boosting Classifier and a Multilayer Perceptron Classifier were also tried, with a slightly worse outcome.

I will be using a portion of the complete dataset available [here](https://zenodo.org/records/4279623). 

## Notes on testing
- Basically the scores are the six above (frobenius score is nothing strange really, let's see if it's useful)
- Comparison of parameters are given by graphs, comparison of different strategies by rankings
- Every value has to be mean +- std: use ShuffleSplit, get scores (mainly macro-averaged on labels) and average on splits, then np.mean, np.std
- For the final version I want 50 splits for every data point
## Contents of package `utils` (update!):
### `testers.py`
- `ss()`
- `assess()`
- `test_datasets()`
### `preprocessers.py`
- `compute_save_rfecv()`
- `compute_save_ppca()`
- `myLabelEncoder()` (class)
### `reporters.py`
- `bs_rankings()`
- `bs_curves()`
- `plot_cm()`
- `format_array_stats()`
- `compare_cm()` (only useful to compare worst to best case)
### `helpers.py`
- `fro_score()`
- `diag()`
- `odiag()`
- `dfs_stats()`
- `mask50` (array): should just save the fitted RFECV object and use the transform() method
