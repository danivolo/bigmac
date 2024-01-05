# BIGMAC (BuildIng Gradually My ALErCE Clone)
## Introduction
This project aims to build a clone of the ALErCE broker lightcurve classifier described in [Sanchez–Saez (2020)](dx.doi.org/10.3847/1538-3881/abd5c1). 

The classifier uses 152 features (though 183 are listed in the dataset) computed from the lightcurves in the red and green bands to distinguish ZTF alert events into 15 classes.
The classification is performed in two steps: first, the signals are divided into transient (supernovae), stochastic (active galactic nuclei/cataclysmic variables)
and periodic (variable stars), then each subgroup is classified indipendently into 4-6 classes. There are thus 4 different classifiers.

The authors found the best performance using a (Balanced) Random Forest (BRF) classifier, implemented with the imbalanced-learn python package.
A Gradient Boosting Classifier and a Multilayer Perceptron Classifier were also tried, with a slightly worse outcome.

I will be using a portion of the complete dataset available [here](https://zenodo.org/records/4279623). 

## Data Preprocessing
### Feature reduction
- RFE(CV) with 50 features is the single best thing to help the classifier
- For assessment see the PCA section
- Apply just RFE to the other sets to see the list of features. A bit of science.

### PCA
- Data is full of NaNs: we have to interpolate using a constant -999 (paper) or other ideas (mean, median...)
- Probabilistic PCA iteratively tries to get a good set of principal components starting from a random one,
  interpolating the data meanwhile
- Just a better imputer or the rotation is useful? Data suggests it does not really help:

PPCA transformation vs simple interpolating (ppcas/):
- PPCA-rotating we get pretty distanced 0.92 and 0.94 roc_auc for full and rfe-reduced dataset largely independent of number components assumed during PPCA
- Just using the interpolated data but without caring about principal components the score goes up to 0.972 and 0.974 respectively, so better.
- Try different random_states, just in case, but I think this proves PPCA is just good enough to fill
- Rankings

Minimum to maximum number of assumed principal components (2->50) (ppcas2/):
- Focusing on the rfe-reduced dataset, it's interesting that even 10 components (out of 50 features) give a really good result
- Either way the best result is obtained assuming as many underlying components as features (50)
- Curve plot

Raw dataset vs rfe vs ppca imputing (ppcas3/):
- Proves that according to every metric but my frobenius_score the rfe-reduced set wins, but there isn't a clear winner between -999 filling and PPCA interpolation without rotation, so simplicity suggests just using constant value imputing
- Rankings

### Still to do
- Now focusing on -999 and PPCA-interpolated rfe-reduced datasets, does conventional PCA help?
- As in, there's no reason PPCA should give a different answer, but maybe algorithms are different, sklearn's function are more informative?
- I tried on the -999 earlier and it didn't seem useful, but maybe because -999 is a cringe value? PPCA prepares the data better?
- Concretely: compare explained variances of PPCA, PCA on -999 and PCA on PPCA (I know, stupid, but try), is there anything useful?
- More concretely: I want a plot of three curves of explained variance as a function of component rank.
## Model analysis
Here the testing is on rfe-reduced, -999 filled dataset.
### BRF
- Why is RF easy to balance?
- What are the hyperparameters other than n_estimators?
- I'm exploring n_estimators and it seems like roc_auc keeps going up until 10000? Hope it's not so
- Testing on rfe-reduced, -999 filled dataset with basic_scores()
- Concretely: a plot of a curve for prec, rec, f1, roc_auc, froD, froOD (everything is in (0,1)) for x logspaced from 100 to 10000.
### Overfitting
- Seems like I don't get overfitting using n_trees < n_samples... hell I cannot even train a 100k tree RF without everything breaking, not counting there's less than 90k samples in periodic...
- Idk I'll do a good scan of 50-10k max and get stds
### Other classifiers
- ADABoost, Gradient Boost, KNN (because I know them and they all support proba)
- For each one do a reasonable scan of complexity parameters and get the best
- Concretely: a plot like above for each classifier
- Then compare with the best BRF using basic_scores() rankings
## Stuff to include somehow
- A confusion matrix of the best case, with stds
## Notes on testing
- Basically the scores are the six above (frobenius score is nothing strange really, let's see if it's useful)
- Comparison of parameters are given by graphs, comparison of different strategies by rankings
- Every value has to be mean +- std: use ShuffleSplit, get scores (mainly macro-averaged on labels) and average on splits, then np.mean, np.std
- For the final version I want 50 splits for every data point
## Contents of package `utils`:
### `testers.py`
- `ss()`
- `assess()`
- `test_datasets()`
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
- `myLabelEncoder()` (class)
- `mask50` (array): should just save the fitted RFECV object and use the transform() method?
