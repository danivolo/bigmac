# BIGMAC (BuildIng Gradually My ALErCE Clone)
## Introduction
This project aims emulate a subset of the ALErCE broker lightcurve classifier described in [Sanchez–Saez et al. (2020)](dx.doi.org/10.3847/1538-3881/abd5c1). 

The classifier uses 152 features (though 183 are listed in the dataset) computed from the lightcurves in the red and green bands to distinguish ZTF alert events into 15 classes.
The classification is performed in two steps: first, the signals are divided into transient (supernovae), stochastic (active galactic nuclei/cataclysmic variables)
and periodic (variable stars), then each subgroup is classified indipendently into 4-6 classes. There are thus 4 independent classifiers.

The authors found the best performance using a (Balanced) Random Forest (BRF) classifier, implemented with the imbalanced-learn python package.
A Gradient Boosting Classifier and a Multilayer Perceptron Classifier were also tried, with a slightly worse outcome.

I will be using a portion of the complete dataset available [here](https://doi.org/10.5281/zenodo.4279623) of about 5k "transient", 30k "stochastic" and 90k "periodic" labeled samples and other 700k unlabeled samples (which I don't plan on using for the moment). 

## Goals
### Preprocessing
Main problems: too many features, too many missing values.
- Perform recursive feature elimination using sklearn RFECV(step=1) on three datasets relative to the second classification step (transient, stochastic, periodic) (spoiler: for periodic 50 features does wonders, still to check the other two).
- Discuss physically the different sets of features selected (still to do).

Fixing the "periodic" dataset (about 90k samples):
- Compare different strategies of filling missing values in the dataset: constant value imputing and [probabilistic PCA](https://github.com/allentran/pca-magic/) (spoiler: after RFE it's about the same, but good results even with 10 assumed underlying principal components).
- Compare the BRF performance on just-filled and PPCA-rotated data (spoiler: it's better not to rotate).
- Discuss the explained variance ratios obtained with different strategies (still to do).

### Model selection
Main problem: dataset is highly imbalanced.

Fixing a specific preprocessing (spoiler: -999 constant imputing as in paper, RFE down to 50 features):
- BRF: find suitable number of trees (spoiler: about 1k but no overfitting at least with n_estimators < n_samples).
- Try other classifiers that support predict_proba (ADABoost, Gradient Boost, KNN) and do the same with a small number of hyperparameters (still to do). There are no "balanced" versions so I'll have to use an external imblearn pipeline.

## Metrics
- Main scores for hyperparameter exploration are precision, recall, f1-score, area under RoC curve, Frobenius scores (introduced here, let's see if they're useful).
- Comparison of hyperparameters are given by graphs, comparison of different strategies by rankings.
- Every value has to be mean +- std: use ShuffleSplit, get macro-averaged scores for each split and average on splits, then np.mean, np.std
- For the final version I want 20-50 splits for every data point
- Confusion matrices for the best classifiers.
  
## Contents of package `utils` (outdated):
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

## References
- Sánchez-Sáez, P., et al. "Alert classification for the alerce broker system: The light curve classifier." The Astronomical Journal 161.3 (2021): 141.
- P. Sánchez-Sáez, I. Reyes, C. Valenzuela, Francisco Förster, S. Eyheramendy, F. Elorrieta, F. E. Bauer, G. Cabrera-Vives, P. A. Estévez, M. Catelan, G. Pignata, P. Huijse, D. De Cicco, P. Arévalo, R. Carrasco-Davis, J. Abril, R. Kurtev, J. Borissova, J. Arredondo, … E. Camacho-Iñiguez. (2020). The ALeRCE Light Curve Classifier: labeled set, features, and classifications [Data set]. Zenodo. 
- For machine learning I mainly sklearn and imblearn.
- PPCA implemented in https://github.com/allentran/pca-magic, according to one user it uses the MAPPCA algorithm in appendix E of “Ilin and Raiko, 2010, J Machine Learning Res.”
