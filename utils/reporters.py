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
from utils.helpers import diag, odiag, dfs_stats, fro_score

def plot_cm(cm, title='Confusion Matrix', std=None, string=None, mult=1):
    """
    Annotated heatmap for a confusion matrix.
    - If std is included format_array_stats is used to create annotation, with 
    string= as the format
    - Use mult=100 to get percentages.
    """
    plt.figure(figsize=(8, 6)) 
    if std is None:
        annot = True
        fmt=".2f"
    else:
        annot = format_array_stats(cm,std,mult=mult)
        fmt = ""
    sns.heatmap(cm, annot=annot, fmt=fmt,cmap="Blues", cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.savefig("imgs/"+title+".png")
    plt.show()

def bs_curves(scores, scores_stds, tick_rot=0, xscale="linear", yscale="linear"):
    """
    Takes the 5 scores from basic_scores() and plots curves.
    """
    for key in scores:
        if key != "name":
            plt.figure()
            plt.errorbar(scores["name"], scores[key], scores_stds[key], fmt="*", label=key)
            plt.xlabel("n_estimators")
            plt.ylabel("Score")
            plt.xscale(xscale)
            plt.yscale(yscale)
            plt.legend()
            plt.xticks(rotation=tick_rot)
    plt.show()

def format_array_stats(mean,std,string="{:3.1f}±{:3.1f}",mult=1):
    """
    Creates a dataframe with elements mean+-std given dataframes
    mean and std and a format string.
    - Use mult=100 if mean is a confusion matrix and you want percentages.
    - It feeds to plot_cm
    """
    formatted = mean.copy()
    for i in range(mean.shape[0]):
        for j in range(mean.shape[1]):
            m = mean.iloc[i,j]
            s = std.iloc[i,j]
            f = string.format(mult*m, mult*s)
            formatted.iloc[i,j] = f
    return formatted

def bs_rankings(scores, winners=False):
    """
    Takes the 5 scores from basic_scores() and gives rankings.
    If winners=True only the highest are shown.
    """
    for key in scores:
        if key != "name":
            scores[key] = np.array(scores[key])
            order = -np.argsort(-scores[key])
            print(key)
            m = order.size-1
            for j, i in enumerate(order):
                if not winners or j==m:
                    print("Number {}: {}, with {}±{}".format(j+1, scores["name"][i], scores[key][i],scores_stds[key][i])) 

def basic_scores(file_pattern,numeric=False):
    """
    Returns a dictionary with precision, recall, f1-score, frobenius score, roc_auc and filename
    of a bunch of files containing outputs from assess().
    Every value in the dict is a python list of single values.
    Only input is the file patter to feed glob.glob
    """
    scores = {
    "prec": [],
    "rec": [],
    "f1": [],
    "ra": [],
    "froD": [],
    "froOD": [],
    "name": []
}
    scores_stds = {
    "prec": [],
    "rec": [],
    "f1": [],
    "ra": [],
    "froD": [],
    "froOD": []
}
    for filepath in glob(file_pattern):
        filename = os.path.basename(filepath)
        with open(filepath,'rb') as f:
            pm, ps, cmm, cms, rm, rs, notes = pkl.load(f)
        
        scores["ra"].append(rm.values[0,0])
        scores["prec"].append(pm.values[0,0])
        scores["rec"].append(pm.values[0,1])
        scores["f1"].append(pm.values[0,2])
        _, b, c = fro_score(cmm)
        scores["froD"].append(b)
        scores["froOD"].append(c)
        
        scores_stds["ra"].append(rs.values[0,0])
        scores_stds["prec"].append(ps.values[0,0])
        scores_stds["rec"].append(ps.values[0,1])
        scores_stds["f1"].append(ps.values[0,2])
        _, b_up, c_up = fro_score(cmm+cms)
        _, b_down, c_down = fro_score(cmm-cms)
        scores_stds["froD"].append(abs(b_up-b_down)/2)
        scores_stds["froOD"].append(abs(c_up-c_down)/2)
        if numeric:
            scores["name"].append(int(notes))
        else:
            scores["name"].append(filename)
    return scores, scores_stds

def compare_cm(cm1f,cm2f,std1f=None,std2f=None):
    """
    Compares two confusion matrices:
    - input is in dataframes
    - perc_change is the percent change element-wise,
    where no change is 0 and change from zero to non-zero 
    is np.nan (instead of np.inf)
    - then there is diagonal/off-diagonal analysis: 
    higher diagonal elements and lower off-diagonal elements are
    desired
    - if std confusion matrix dataframes are not provided,
    (off-)diag stds are computed from elements directly
    - the output is the z_score without the modulus,
    so the mean change of diag and non diag elements in units
    of respective stds.
    """
    cm1 = cm1f.values
    cm2 = cm2f.values
    n_labels = cm1.shape[0]
    perc_change_array = np.zeros((n_labels,n_labels))
    for i in range(n_labels):
        for j in range(n_labels):
            if cm1[i,j] != 0:
                perc_change_array[i,j] = (cm2[i,j]-cm1[i,j])/cm1[i,j]
            else:
                if cm2[i,j]==cm1[i,j]:
                    perc_change_array[i,j] = 0
                else:
                    perc_change_array[i,j] = np.nan
    perc_change_array = perc_change_array
    perc_change = pd.DataFrame(perc_change_array,index=cm1f.index,columns=cm1f.columns)
    diag1 = diag(cm1)
    off_diag1 = odiag(cm1)
    diag2 = diag(cm2)
    off_diag2 = odiag(cm2)
    
    diag1_mean = np.nanmean(diag1)
    off_diag1_mean = np.nanmean(off_diag1)
    diag2_mean = np.nanmean(diag2)
    off_diag2_mean = np.nanmean(off_diag2)

    if std1f is None:
        diag1_std = np.nanstd(diag1)#/np.sqrt(len(diag1))
        off_diag1_std = np.nanstd(off_diag1)#/np.sqrt(len(off_diag1))
    else:
        std1 = std1f.values
        diag1_std = np.sqrt(sum(diag(std1)**2))/np.sqrt(len(diag1))
        off_diag1_std = np.sqrt(sum(odiag(std1)**2))/np.sqrt(len(diag1))

    if std2f is None:
        diag2_std = np.nanstd(diag2)#/np.sqrt(len(diag2))
        off_diag2_std = np.nanstd(off_diag2)#/np.sqrt(len(off_diag2))
    else:
        std2 = std2f.values
        diag2_std = np.sqrt(sum(diag(std2)**2))/np.sqrt(len(diag2))
        off_diag2_std = np.sqrt(sum(odiag(std2)**2))/np.sqrt(len(diag2))

    diag_z = (diag2_mean-diag1_mean)/np.sqrt(diag1_std**2+diag2_std**2)
    off_diag_z = (off_diag2_mean-off_diag1_mean)/np.sqrt(off_diag1_std**2+off_diag2_std**2)
    
    return (perc_change, diag_z, off_diag_z)