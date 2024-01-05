import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
import pickle as pkl
import importlib

from sklearn.model_selection import train_test_split,cross_val_score, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.impute import SimpleImputer

import utils.helpers
importlib.reload(utils.helpers)
from utils.helpers import myLabelEncoder, dfs_stats

def ss(X, y, labels=None, random_state=48, test_size=0.2, fill_value=-999, n_estimators=400, sampling_strategy="auto", replacement=True, bootstrap=True, n_splits=1, scaling=False, progress=True):
    """
    Performs cross-validation of a BRF model using ShuffleSplit.
    - The outputs are numpy arrays where with n_splits as the first dimension encoding
    test labels, predicted labels, predicted probability labels, and the classifier for 
    each of the folds.
    - The label encoding is my custom function written to emulate sklearn syntax, I need
    it to change the order of the labels from the alphabetical default to match the data
    reports from the paper.
    - If NaNs are found they are automatically filled with the constant value -999 as the
    authors did in the paper. I'm expecting to interpolate missing data with PPCA (pca-magic)
    and in that case the feature won't activate.
    - A progress bar is implemented via tqmd
    """
    ## y formatting
    if labels is None:
        labels = np.unique(y)
    me = myLabelEncoder(labels)
    y = me.fit_transform(y)

    ## Check for missing values
    theres_nans = np.any(np.isnan(X))
    if theres_nans:
        si = SimpleImputer(strategy="constant",fill_value=fill_value)
        print("Missing values found. Imputing.")

    ## BRF estimator
    clf = BalancedRandomForestClassifier(n_estimators=n_estimators,
                                    random_state=random_state,
                                    sampling_strategy=sampling_strategy,
                                    replacement=replacement,
                                    bootstrap=bootstrap,
                                    n_jobs=-1)

    ## not k-fold, just split in random ways the set. lemme handle scoring myself.
    rs = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

    y_test_array = []
    y_pred_array = []
    y_pred_proba_array = []
    clf_array = []

    split = rs.split(X)
    if not progress:
        r = range(rs.get_n_splits())
    else:
        r = tqdm(range(rs.get_n_splits()))
    for i in r:
        train_index, test_index = next(split)
        X_train = X[train_index,:]
        X_test = X[test_index,:]
        y_train = y[train_index]
        y_test_array.append(y[test_index])
        ## Optional: scaling (unnecessary with ppca)
        if scaling:
             scaler = StandardScaler()
             X_train = scaler.fit_transform(X_train)
             X_test = scaler.transform(X_test)
        ## shouldn't happen with ppca
        if theres_nans:
            X_train = si.fit_transform(X_train)
            X_test = si.transform(X_test)
        ## Training
        clf.fit(X_train, y_train)
        ## Testing
        y_pred_array.append(clf.predict(X_test))
        y_pred_proba_array.append(clf.predict_proba(X_test))
        clf_array.append(clf)

    return (np.stack(y_test_array),
            np.stack(y_pred_array),
            np.stack(y_pred_proba_array),
            np.stack(clf_array))

def assess(y_test_array, y_pred_array, y_pred_proba_array=None, labels=None, average=None, split_average=False):
    """
    Takes the first 3 outputs from ss and provides:
    - precision, recall, fscore, support
    - confusion matrix
    - area under roc curve
    as python lists of dataframes, with dimension n_splits.
    If split_average=True then the outputs are mean and stds for each 
    of the dataframes, instead of lists
    average="macro" performs average on classes as in sklearn parameters.
    """
    if labels is None:
        labels = np.unique(y_test_array)
    av_labels = labels
    n_splits = y_test_array.shape[0]
    n_labels = len(labels)
    
    prfs = []
    cm = []
    rocauc = []
    for i in range(n_splits):
        if average is None:
            prfs_array = np.asarray(precision_recall_fscore_support(y_test_array[i,:], y_pred_array[i,:])).T
        else:
            prfs_array = np.asarray(precision_recall_fscore_support(y_test_array[i,:], y_pred_array[i,:], average=average)).reshape((1,-1))
            av_labels=[str(average) + " average"]
        prfs.append(pd.DataFrame(prfs_array, index=av_labels, columns=["precision", "recall", "fscore", "support"]))
        cm.append(pd.DataFrame(\
            confusion_matrix(y_test_array[i,:], y_pred_array[i,:],normalize="true"),
            index=labels,columns=labels))
        if y_pred_proba_array is not None:
            rocauc.append(\
                pd.DataFrame(np.asarray(roc_auc_score(y_test_array[i,:], 
                                       y_pred_proba_array[i,:], 
                                       multi_class='ovr', 
                                       average=average)).reshape((1,-1)),
                            index=["roc_auc"], columns=av_labels))
    if split_average:
        prfs_mean, prfs_std = dfs_stats(prfs)
        cm_mean, cm_std = dfs_stats(cm)
        rocauc_mean, rocauc_std = dfs_stats(rocauc)
        return (prfs_mean, prfs_std, cm_mean, cm_std, rocauc_mean, rocauc_std)
    else:
        return (prfs, cm, rocauc)