import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown
from IPython.core.display import display, HTML
import base64
from facets_overview.generic_feature_statistics_generator import GenericFeatureStatisticsGenerator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, accuracy_score, log_loss, balanced_accuracy_score,
                             average_precision_score,recall_score, 
                             f1_score, plot_roc_curve, plot_precision_recall_curve)
import xgboost as xgb
from xgboost import XGBClassifier
from abc import ABC


def train_test_split(df, split_col, pct_dev=0.15, pct_test=0.15, seed=3):
    """
    split df into train/dev/test sets based on split_col
    :param split_col: str, column name on which to split
    :param pct_dev: percent of split col unique in dev set (default = 0.15)
    :param pct_test: percent of split col unique in test set (default = 0.15)
    :param seed: int, seed for random number generator for reproducibility
    :return: df_train, df_dev, df_test
    """

    np.random.seed(seed)
    split_dev = np.random.choice(df[split_col].unique(),
                                 round(df[split_col].nunique( ) * pct_dev), replace=False)

    split_test = np.random.choice(df[~df[split_col].isin(split_dev)][split_col].unique(),
                                  round(df[split_col].nunique( ) * pct_test), replace=False)

    split_train = df[(~df[split_col].isin(split_dev)) & (~df[split_col].isin(split_test))][split_col].unique()

    df_train = df[df[split_col].isin(split_train)]
    df_dev = df[df[split_col].isin(split_dev)]
    df_test = df[df[split_col].isin(split_test)]

    return df_train, df_dev, df_test


def plot_facets(groups_list, elem_name):
    """
    This just shows the Facets plots for some data
    This code is modified from https://github.com/PAIR-code/facets/tree/master/facets_overview
    :param groups_list: list of dictionaries of the form {'name': '<name for data>', 'table': <DataFrame>}
    :param elem_name: a name for the facets element
    :return: None, displays html for Facets plots
    """
    gfsg = GenericFeatureStatisticsGenerator()
    proto = gfsg.ProtoFromDataFrames(groups_list)
    protostr = base64.b64encode(proto.SerializeToString()).decode("utf-8")

    HTML_TEMPLATE = """
            <script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"></script>
            <link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/1.0.0/facets-dist/facets-jupyter.html" >
            <facets-overview id='""" + elem_name + """'</facets-overview>
            <script>
              document.querySelector('#""" + elem_name + """').protoInput = "{protostr}";
            </script>"""
    html = HTML_TEMPLATE.format(protostr=protostr)
    display(HTML(html))

    
def plot_correlation_heatmap(df, cols):
    """
    This just shows the pairwise correlations heatmap
    This code is modified from https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec
    :param df: DataFrame containing data
    :param cols: list, columns to use for correlations
    :return: None, displays correaltion heatmap
    """
    corr = df[cols].corr(method='pearson')  # kendall, spearman
    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(
        corr,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )


def get_target_ahead(df, patient_id, time_col, label_col, time_ahead):
    """
    Get the target label for sepsis some number of hours in the future
    :param df: DataFrame
    :param patient_id: str, column name for patient ID for grouping data
    :param time_col: str, column name for time in the ICU
    :param label_col: str, column name for sepsis label
    :param time_ahead: int, number of hours ahead that we want to predict sepsis label
    :return: DataFrame containing columns for the label at the target prediction horizon
    """
    df = df.sort_values([patient_id,time_col]) # just make sure df is sorted for lagging

    df[label_col+'_next'+str(time_ahead)] = df.groupby(patient_id)[label_col].shift(-1*time_ahead) # get the sepsis label from 4 hours ahead

    return df


def filter_rows(df, time_idx, label_col, label_col_ahead, group_name, pred_hours_ahead):
    """
    Since we're interested in predicting patients that will have sepsis in 4 hours' time we will do some filtering of the data set as follows:
        - Remove patients that have sepsis within the first four hours
        - Remove rows after the first time the 4-hours ahead sepsis label is true
        - Remove the last four hours of data for patients that do not have a sepsis label at any point in the data set
          because we don't know whether they go on to develop sepsis after the time period for which we have data

    :param df: pandas DataFrame, DataFrame for filtering
    :param time_idx: str, column name that provides the time index (starts at 0 and counts hours in ICU)
    :param label_col: str, column name that provides the label (sepsis)
    :param label_col_ahead: str, column name that has the label at the prediction time window ahead
    :param group_name: str, column name for the column containing the patient ID for grouping data
    :param pred_hours_ahead: int, the number of hours ahead that we are interested in predicting sepsis
    :return: filtered DataFrame
    """
    # Filter patients that have sepsis within the prediction horizon because we do not have enough history to predict at the desired horizon
    firstWindow_groupedPatient = df[df[time_idx] <= pred_hours_ahead].groupby(group_name)[label_col].max().reset_index()
    patients_withinFirst_predWindow = firstWindow_groupedPatient[firstWindow_groupedPatient[label_col] == 1][
        group_name].unique()
    df = df[~df[group_name].isin(patients_withinFirst_predWindow)]

    # Filter rows after the pred_hours_ahead label is true
    # we'll use the lag of the target label so that we can find first occurrence of the target label
    # ie. after the first occurrence of the target label, the target and the lag1 of the target will both be 1
    df = df[~((df[label_col_ahead] == 1) & (df.groupby(group_name)[label_col_ahead].shift(1) == 1))]

    # Filter rows for the last pred_hours_ahead for a patient if the patient does not have a sepsis label in that time
    df = df[df[label_col_ahead].notnull()]

    return df


def impute_carryforward_group_simple(df, group_name, group_metric, metric, fcols, missing_indicator=True):
    """
    Impute missing values through carry-forward and simple imputation
    :param df: DataFrame with values for imputation
    :param group_name: patient ID for grouping for within-patient imputation
    :param group_metric: metric to use for imputation within-patient
    :param metric: metric to use for imputation across the entire DataFrame
    :param fcols: columns on which to perform imputation
    :param missing_indicator: boolean for whether to add a column with a missingness after carry forward indicator
    :return: the DataFrame with <col>_fill and, if desired, <col>_missing columns for all columns in fcol
    """
    # Carry forward within the patient's history
    df_fill = df.groupby(group_name)[fcols].ffill().rename(columns={fcol: fcol + '_fill' for fcol in fcols})
    df_fill = pd.concat([df, df_fill], axis=1)

    # Add missing indicator for values that are missing after carry forward
    if missing_indicator:
        df_missing = df_fill[(fcol + '_fill' for fcol in fcols)].isnull().astype(int) \
            .rename(columns={fcol + '_fill': fcol.split('_')[0] + '_missing' for fcol in fcols})
        df_fill = pd.concat([df_fill, df_missing], axis=1)

    # Simple impuation within patient histories
    for fcol in fcols:
        df_fill[fcol + '_fill'] = df_fill[fcol + '_fill'].fillna(
            df_fill.groupby(group_name)[fcol].transform(group_metric))

    # Simple imputation across patients for data that is missing entirely for a patient
    for fcol in fcols:
        df_fill[fcol + '_fill'] = df_fill[fcol + '_fill'].fillna(df_fill[fcol].agg(metric))

    return df_fill


def add_hist_features(df, lags, vitals_cols, patient_id):
    """
    Add lag features
    :param df: DataFrame
    :param lags: list of int, lags in hours for creating features
    :param vitals_cols: list of str, column names for vitals colunms to get lags for
    :param patient_id: str, column name for patient ID
    :return: dataframe, with lagged features
    """
    for v in vitals_cols:
        for lag in lags:
            df[v+'_fill_lag'+str(lag)] = df.groupby(patient_id)[v+'_fill'].shift(lag)
            df[v+'_fill_lag'+str(lag)+'_delta'] = df[v+'_fill'] - df[v+'_fill_lag'+str(lag)]
    return df


class Classifier(ABC):
    """
    This is an abstract class that other classes will inherit for making sklearn-like classifiers
    """
    # this line is needed for sklearn to recognize this estimator (for metrics plots in this case)
    _estimator_type = "classifier"
    
    @property
    def needs_normalization(self):
        """
        this property is here so that we can use the same pipeline for all of the classifiers
        and only normalize some of them as required
        :return: True or False for whether we want to normalize data for training classifier
        """
        pass
    
    def fit(self, X, y):
        """
        fit method for training classifier
        :param X: DataFrame or array of features
        :param y: DataFrame or array of target
        :return: trained model
        """
        pass
    
    def predict(self, X):
        """
        predict method for classifier
        :param X: DataFrame or array of features
        :return: predicted values
        """
        pass

    def predict_proba(self, X):
        """
        predict_proba method for classifier
        :param X: DataFrame or array of features
        :return: array, probability estimates
        """
        pass

class MyLogisticRegression(Classifier):
    """
    Logistic regression classifier
    """
    
    def __init__(self, myclass_weight=1, mymax_iter=100):
        
        self._Classifier = LogisticRegression(class_weight=myclass_weight, max_iter=mymax_iter)
        self.classes_ = np.array([-1, 1]) # needed for sklearn metrics
  
    @property
    def name(self): 
        return 'Logistic Regression'
    
    @property
    def needs_normalization(self): 
        return True
    
    def fit(self, X, y): 
        return self._Classifier.fit(X, y)
    
    def predict(self, X):
        return self._Classifier.predict(X)
                 
    def predict_proba(self, X):
        return self._Classifier.predict_proba(X)
                 
                 
class MyXGBClassifier(Classifier): 
    """
    XGBoost classifier
    """

    def __init__(self, pos_weight=1, reg_lambda=0.01):

        self._Classifier = XGBClassifier(scale_pos_weight=pos_weight, reg_lambda=reg_lambda)
        self.classes_ = np.array([-1, 1])  # needed for sklearn metrics

  
    @property
    def name(self): 
        return 'XGBoost'
    
    @property
    def needs_normalization(self): 
        return False
    
    def fit(self, X, y): 
        return self._Classifier.fit(X, y)
                 
    def predict(self, X):
        return self._Classifier.predict(X)
    
    def predict_proba(self, X):
        return self._Classifier.predict_proba(X)


def show_model_scores(clf, X, y_true, y_pred, datasetname):
    """
    Show metrics for trained model including AUC plots
    :param clf: trained classifier
    :param X: array, model input
    :param y_true: array, true labels (boolean)
    :param y_pred: arrray, predicted labels (boolean)
    :param datasetname: str, dataset (eg. "dev", "train")
    :return: None
    """
    display(Markdown('Classification Accuracy, {}: {:.4}'.format(datasetname, accuracy_score(y_true, y_pred))))
    display(Markdown('Balanced Classification Accuracy, {}: {:.4}'.format(datasetname, balanced_accuracy_score(y_true, y_pred))))
    display(Markdown('Precision, {}: {:.4}'.format(datasetname, average_precision_score(y_true, y_pred))))
    display(Markdown('Recall, {}: {:.4}'.format(datasetname, recall_score(y_true, y_pred))))
    display(Markdown('F1, {}: {:.4}'.format(datasetname, f1_score(y_true, y_pred))))
    
    f, ax = plt.subplots(1,3, figsize=(14,4))
    display(Markdown('Plots for {} data'.format(datasetname)))
    plot_roc_curve(clf, X, y_true, ax=ax[0]) 
    plot_precision_recall_curve(clf, X, y_true, ax=ax[1]) 
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, ax=ax[2])
    plt.show()
