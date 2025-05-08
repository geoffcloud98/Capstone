from typing import Union, List
from copy import deepcopy 
from pdb import set_trace
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns

class Standardization(BaseEstimator, TransformerMixin):
    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: pd.DataFrame = None):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        return self
    
    def transform(self, X):
        return (X  - self.mean) / (self.std + self.epsilon)

class AddBias(BaseEstimator, TransformerMixin):
    def fit(self,
            X: pd.DataFrame, 
            y: pd.DataFrame = None) -> pd.DataFrame:
        return self
    
    def transform(self,
                  X: pd.DataFrame, 
                  y: pd.DataFrame = None) -> pd.DataFrame:
        X = X.copy()
        X.insert(0, 'bias', 1)
        return X

class ImageNormalization(BaseEstimator, TransformerMixin): 
    def fit(self,
            X: pd.DataFrame, 
            y: pd.DataFrame = None) -> pd.DataFrame:
        return self
    
    def transform(self,
                  X: pd.DataFrame, 
                  y: pd.DataFrame = None) -> pd.DataFrame:
        return (X/255).astype(np.float16)

class OneHotEncoding(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names='auto'):
        self.feature_names = feature_names
        self.encoder = OneHotEncoder(categories=feature_names, sparse=False)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        
        self.encoder.fit(X)
        
        # Store names of features
        try:
            self.feature_names = self.encoder.get_feature_names_out()
        except AttributeError:
            self.feature_names = self.encoder.get_feature_names()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        
        one_hot =  self.encoder.transform(X)

        return pd.DataFrame(one_hot, columns=self.feature_names)
    
class ExtractStatistics(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        means = np.mean(X, axis=1)
        variances = np.var(X, axis=1)
        maxs = np.max(X, axis=1)
        sums = np.sum(X, axis=1)
        

        # Stack features, shape will be (n_samples, 4)
        features = np.column_stack((means, variances, maxs, sums))

        return features
    
class ExtractStandardDeviations(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        std = np.std(X, axis=1)
        

        # Stack features, shape will be (n_samples, 1)
        features = np.column_stack((X, std))

        return features

def feature_label_split(X_trn, X_vld=None):
    """
    X_trn: training data
    X_vld: validation data
    """
    y_trn = X_trn[:, 0]
    X_trn = X_trn[:, 1:]

    if X_vld is not None:
        y_vld = X_vld[:, 0]
        X_vld = X_vld[:, 1:]
        return X_trn, y_trn, X_vld, y_vld
        
    return X_trn, y_trn

import numpy as np
from typing import List, Tuple

def binarize_classes(
    X: np.ndarray,
    y: np.ndarray,
    pos_class: List,
    neg_class: List,
    neg_label: int = -1,
    pos_label: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Converts a multi-class classification problem into a binary classification problem

    Args:
        X: NumPy array of input features
        y: NumPy array of labels
        pos_class: List of labels to be grouped into the positive class
        neg_class: List of labels to be grouped into the negative class
        neg_label: Label assigned to the negative class
        pos_label: Label assigned to the positive class

    Returns:
        Tuple of new_X, new_y as NumPy arrays
    """
    y = np.asarray(y).flatten()

    pos_mask = np.isin(y, pos_class)
    neg_mask = np.isin(y, neg_class)

    pos_X = X[pos_mask]
    pos_y = np.full(pos_X.shape[0], pos_label)

    neg_X = X[neg_mask]
    neg_y = np.full(neg_X.shape[0], neg_label)

    new_X = np.concatenate([pos_X, neg_X], axis=0)
    new_y = np.concatenate([pos_y, neg_y], axis=0)

    return new_X, new_y


def dataframe_to_array(dfs: List[pd.DataFrame]):
    """ Converts any Pandas DataFrames into NumPy arrays
    
        Args:
            dfs: A list of Pandas DataFrames to be converted.
    
    """
    arrays = []
    for df in dfs:
        if isinstance(df, np.ndarray):
            arrays.append(df)
        else:
            arrays.append(df.values)
    return arrays

def copy_dataset(data: list):
    """ Deep copies all passed data.

        Args:
            data: A list of data objects such as NumPy arrays or Pandas DataFrames.
    """  
    return [deepcopy(d) for d in data]

def get_class_locs(y):
    """ Gets 1 index location for each class label

        Returns:
                List: indicies for each class
    """
    class_labels = np.unique(y)

    sample_locs = []
    for cls in class_labels:
        class_locs = np.where(y == cls)[0]

        for i in range(1):
            sample_locs.append(class_locs[i])

    return sample_locs

def display_cwt_image(trn_sample_locs, vld_sample_locs, X_cwt, validation_data=False):
    """ Plots the cwt ECG signals for training and validation datasets
        Args:
            trn_sample_locs: training sample indices to plot
            vld_sample_locs: validation sample indices to plot
            X_cwt: Continuous Wave Transformed ECG signals
            validation_data: boolean for validation samples
    """
    #display transformed sample image
    #extract and convert to absolute value
    class_labels = ['N', 'r', 'S', 'V', 'Q']
    
    #choose which sample to use
    sample_indices = trn_sample_locs

    if validation_data:
        sample_indices = vld_sample_locs

    #set up figure
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for ax, idx, label in zip(axes, sample_indices, class_labels):
        cwt_img = np.abs(X_cwt[idx])
        ax.imshow(cwt_img, aspect='auto', cmap='jet')
        ax.set_title(f'Class {label}')
        ax.axis('off')
    
    fig.suptitle("Training Samples by Class", fontsize=16)
    if validation_data:
        fig.suptitle("Validation Samples by Class", fontsize=16)
        
    plt.tight_layout()
    plt.show()

def get_target_names_class_labels(rename_map: dict):
    """ Returns target names and class labels
        Args:
            rename_map: dictionary containing class labels as keys and target names a values
    """

    class_labels = list(rename_map.keys())
    target_names = list(rename_map.values())

    return class_labels, target_names

def plot_ecg(trn_sample_locs, vld_sample_locs, target_names, X, validation_data=False):
    """ Plots an ECG signal for each class
        Args:
            trn_sample_locs: training sample indices to plot
            vld_sample_locs: validation sample indices to plot
            target_names: Target names of the class label
            X: dataset that will be indexed
            validation_data: flag switch between training and validation
    """

    sample_indices = vld_sample_locs if validation_data else trn_sample_locs

    #set up figure
    fig, axes = plt.subplots(1, len(sample_indices), figsize=(20, 4))
    for ax, idx, name in zip(axes, sample_indices, target_names):
        sample = X[idx]
        ax.plot(sample, label=f'Class {name}')
        ax.set_title(f'Class {name}')
        ax.axis('off')
    
    fig.suptitle("Training Samples by Class", fontsize=16)
    
    if validation_data:
        fig.suptitle("Validation Samples by Class", fontsize=16)
        
    plt.tight_layout()
    plt.show()

def run_report(cf_matrix, isCNN=False):

    plt.figure(figsize=(8, 6))

    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}" for v1, v2, in zip(group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(5,5)
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')

    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
