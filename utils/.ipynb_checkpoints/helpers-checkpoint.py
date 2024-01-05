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
import requests

from sklearn.model_selection import train_test_split,cross_val_score, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.impute import SimpleImputer

def fro_score(cmf, reduced=True):
    """
    Computes squared Frobenius distance (Euclidean distance in dimension n^2) 
    of a (presumably) confusion matrix from the identity.
    Distance is normalized by worst case which would be 1 off diagonals,
    giving distance n^2.
    """
    cm = cmf.values
    n = cm.shape[0]
    d1 = np.sum((1-diag(cm))**2)
    d2 = np.sum(odiag(cm)**2)
    d = d1+d2
    if reduced:
        d = d/n**2
        d1 = d1/n
        d2 = d2/(n**2-n)
    return d, d1, d2

def diag(a):
    """
    Diagonal elements of a matrix.
    """
    return np.array([a[i,i] for i in range(a.shape[0])])

def odiag(a):
    """
    Off-diagonal elements of a matrix
    """
    return np.array([a[i,j] for i in range(a.shape[0]) for j in range(i)])

def dfs_stats(dfs, over_sqrt_n=True):
    """
    Mean and std of purely numerical (no dtype=object shit) dataframes.
    """
    index = dfs[0].index
    columns = dfs[0].columns
    mean = pd.DataFrame(np.mean(np.asarray(dfs).astype(np.float32),axis=0),index=index,columns=columns)
    factor = np.sqrt(6*len(dfs)) if over_sqrt_n else 1
    std = pd.DataFrame(np.std(np.asarray(dfs).astype(np.float32),axis=0),index=index,columns=columns)/factor
    return (mean, std)

class myLabelEncoder:
    """
    Trying to mimic sklearn syntax, but no .fit and .transform
    """
    def __init__(self,labels):
        self.labels = labels
    def fit_transform(self,y):
        return np.array([self.labels.index(x) for x in y])
    def inverse_transform(self,y):
        return np.array([self.labels[x] for x in y])

def download_file(url, destination_folder="data/"):
    # Ensure the destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Get the file name from the URL
    file_name = url.split("/")[-1]

    # Full path to save the file
    file_path = os.path.join(destination_folder, file_name)

    # Download the file from the URL
    response = requests.get(url)
    with open(file_path, 'wb') as file:
        file.write(response.content)

    print(f"File downloaded and saved to {file_path}")

def uncompress_targz(compressed_file_path, destination_folder="data/csv/", clean=False):
    # Extract the contents of the compressed file
    with tarfile.open(compressed_file_path, 'r:gz') as tar:
        tar.extractall(destination_folder)
    if clean:
        os.remove(compressed_file_path)

tlabels = ["SNIa", "SNIbc", "SNII", "SLSN"]
slabels = ["QSO", "AGN", "Blazar", "YSO", "CV/Nova"]
plabels = ["LPV","E","DSCT","RRL","CEP","Periodic-Other"]

def divide_dataset(raw_data_path="data/csv/", destination_folder="data/pkl/divided/"):
    features_filename = "features_for_lc_classifier_20200609.csv"
    labels_filename = "labeled_set_lc_classifier_SanchezSaez_2020.csv"
    #test_filename = "ALeRCE_lc_classifier_outputs_ZTF_unlabeled_set_20200609.csv"
    
    features_file = pd.read_csv(raw_data_path + features_filename,index_col="oid")
    labels_file = pd.read_csv(raw_data_path + labels_filename,index_col="oid")
    #test_file = pd.read_csv(raw_data_path + test_filename,index_col="oid")

    # select "transient" (supernovae) samples and save them as .pkl
    
    transient_classes = tlabels
    transient_labeled = labels_file[labels_file["classALeRCE"].isin(transient_classes)]
    transient_features = features_file.loc[transient_labeled.index]
    transient_labeled.to_pickle(destination_folder + "transient_labeled.pkl")
    transient_features.to_pickle(destination_folder + "transient_features.pkl")
    
    # select "stochastic" (AGN) samples and save them as .pkl
    
    stochastic_classes = slabels
    stochastic_labeled = labels_file[labels_file["classALeRCE"].isin(stochastic_classes)]
    stochastic_features = features_file.loc[stochastic_labeled.index]
    stochastic_labeled.to_pickle(destination_folder + "stochastic_labeled.pkl")
    stochastic_features.to_pickle(destination_folder + "stochastic_features.pkl")
    
    # select "periodic" (variable stars) samples and save them as .pkl
    
    periodic_classes = plabels
    periodic_labeled = labels_file[labels_file["classALeRCE"].isin(periodic_classes)]
    periodic_features = features_file.loc[periodic_labeled.index]
    periodic_labeled.to_pickle(destination_folder + "periodic_labeled.pkl")
    periodic_features.to_pickle(destination_folder + "periodic_features.pkl")

SanchezSaez2020_url = "https://zenodo.org/records/4279623/files/files_for_lc_classifier_SanchezSaez2020.tar.gz?download=1"

## feature mask from rfecv(step=1 or even 0.5), 50 features
mask50 = np.array([True,True,True,True,False,False,False,False,False,False,False,False,False,False,True,False,True,True,True,True,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,False,False,True,False,False,False,False,False,False,False,True,False,True,False,False,False,True,False,False,True,False,True,True,False,False,True,False,False,False,False,True,False,False,False,True,False,True,False,True,False,True,False,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,True,False,False,False,True,True,False,False,False,False,False,False,False,False,True,True,True,True,True,False,False,False,False,False,False,False,True,False,False,False,False,True,False,False,False,False,True,True,True,False,False,False,True,True,False,False,False,False,False,False,False,False,True,True,True,True,True,True,True,True,True,True,False,False,False])

## feature mask from article, should be 30 but somehow are 43 idk
mask43 = np.array([True, True, True, True, False, False, True, True, False, False, False, False, False, False, True, True, True, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False, True, False, False, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, False, False, True, True, True, True, False, False, False, False, False, False, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, True, True, False, False])