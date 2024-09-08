import os
import pickle
from datetime import datetime
from itertools import product
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error, accuracy_score, mean_absolute_error, roc_auc_score,
    precision_score, recall_score, confusion_matrix, f1_score
)
from sklearn.svm import SVC
from tqdm import tqdm
import xgboost as xgb

from new_base import ClassifierWithImputation
from data_loaders import MissDataset, DataLoadersEnum
from complete_case_imputer import CompleteCaseImputer


import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.feature_selection import SelectKBest, RFE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from deap import base, creator, tools, algorithms  # For Genetic Algorithm
from sklearn.model_selection import cross_val_score
import random
from sklearn_genetic import GAFeatureSelectionCV

#genetic algorithm
import matplotlib.pyplot as plt
from sklearn_genetic import GAFeatureSelectionCV
from sklearn_genetic.plots import plot_fitness_evolution
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import numpy as np



class FeatureSelectPipeline:

    def __init__(
        self,
        dataset_object: MissDataset,
        missing_mechanism='mcar',
        pipeline_name='fs_pipeline',
        dataset_name='',
        name='Experiment_' + str(datetime.now()),
        base_dir=None,
        classifier_pool=None,
        random_state=42,
        fs_type='correlation_coefficient'
    ):
        self.dataset_object = dataset_object
        self.p_miss = self.dataset_object.p_miss
        self.n_folds = self.dataset_object.n_folds
        self.fs_type = fs_type
        
        # Initialize a OneHotEncoder for label encoding the target column
        self.label_enc = OneHotEncoder()
        # Fit the encoder to the target column data in the dataset
        self.label_enc.fit(np.array(self.dataset_object.data[self.dataset_object.target_col]).reshape(-1, 1))

        self.dataset_data = dataset_object.data
        self.dataset_name = dataset_name
        self.imputations = {}
        self.imputation_results = []
        self.classification_results = []

        ###############################################################################################################

        # Define the base directory for storing experiment results
        # Use the provided base directory or create a custom path if none is provided
        if base_dir is None:
            base_dir = '../Experiment_Trials/' + dataset_name  +'/' + name +'/' + pipeline_name

        # Create the base directory if it doesn't exist
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        # Define the directory for storing results, based on the missing data mechanism
        self.base_dir = os.path.join(base_dir, missing_mechanism)
        if not os.path.exists(self.base_dir):
            os.mkdir(self.base_dir)

        # Define the directory specifically for storing cross-validation results
        self.results_dir = os.path.join(self.base_dir, 'results')
        if not os.path.exists(self.results_dir):
            os.mkdir(self.results_dir)

        # Initialize a dictionary to store cross-validation results for each fold
        self.cv_results = {
            i: {} for i in range(self.n_folds)
        }

        self.results = {}
        self.random_state = random_state

        ###############################################################################################################

        self._init_pipelines(classifier_pool=classifier_pool)

        # print(self.pipelines)

        # Initialize a dictionary to store metrics for each pipeline, starting with empty lists
        self.prediction_metrics = {p: [] for p in self.pipelines}

        # Initialize a dictionary to store imputed evaluations for each pipeline, starting with empty lists
        self.imputed_evals = {p: [] for p in self.pipelines}

        # # Print the initialized dictionary of predictor imputer pipelines
        # print("Predictor Imputer Pipelines(all pipeline runs once per fold):")
        # for pipeline, evals in self.imputed_evals.items():
        #     print(f"\t{pipeline}")

    def _init_pipelines(self, classifier_pool=None):
        if classifier_pool is None:
            xgb_params = {'n_jobs': 1, 'max_depth': 4, 'n_estimators': 50, 'verbosity': 0}
            rf_params = {'n_jobs': 1, 'max_depth': 4, 'n_estimators': 50, 'verbose': 0}

            models = [
                SVC(probability=True),
                RandomForestClassifier(),
                xgb.XGBClassifier()
            ]
            imputers = [
                KNNImputer(n_neighbors=5),
                SimpleImputer(strategy='mean'),
                IterativeImputer(RandomForestRegressor(**rf_params))
            ]

            

            # clf_imputer_pairs = product(models, imputers)
            clf_imputer_pairs = product( imputers, models)
            pipelines_list = [
                ClassifierWithImputation(
                    
                    imputer=imp,
                    estimator=clf
                )
                for  imp,clf in clf_imputer_pairs
            ]


            pipeline_names = {
            "SVC": "SV-Classifier",
            "RandomForestClassifier": "RF-Classifier",
            "XGBClassifier": "XGB-Classifier",
            "KNNImputer": "KNN-Imputer",
            "SimpleImputer": "Mean-Imputer",
            "IterativeImputer(<class 'sklearn.ensemble._forest.RandomForestRegressor'>)": "RF-Imputer"
            }


            pipelines = {

                # 'Estim(' + pipeline_names[p.estimator_name] + ')_Imputer(' + pipeline_names[p.imputer_name] + ')': p
                'Imputer(' + pipeline_names[p.imputer_name] + ')_Estim(' + pipeline_names[p.estimator_name] + ')': p
                for p in pipelines_list
            }
        else:
            pipelines = classifier_pool   

        self.pipelines = pipelines

        self.unfitted_pipelines = deepcopy(self.pipelines)







    def ensure_20_percent_non_missing(self,X_train, y_train, imputation_strategy='mean'):
        """
        Ensure that at least 20% of X_train has no missing values. If the number of non-missing rows 
        is less than 20%, randomly impute missing values to reach at least 20%.
        
        Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        imputation_strategy (str): The strategy for imputing missing values ('mean', 'median', 'most_frequent', 'constant').
        
        Returns:
        pd.DataFrame: X_train with at least 20% non-missing rows.
        pd.Series: y_train corresponding to the returned X_train.
        """
        # Identify indices of rows with no missing values
        non_missing_indices = X_train.dropna().index
        total_rows = X_train.shape[0]
        threshold = 0.2 * total_rows

        # Check if the number of non-missing rows is less than 20% of the total rows
        if len(non_missing_indices) < threshold:
            print("Non-missing rows are less than 20% of total data, performing imputation...")

            # Find the number of rows needed to have 20% non-missing
            rows_needed = int(threshold - len(non_missing_indices))
            
            # Randomly select rows to impute to have at least 20% complete cases
            missing_indices = X_train.index.difference(non_missing_indices)
            impute_indices = np.random.choice(missing_indices, size=rows_needed, replace=False)
            
            # Impute missing values for the selected rows using SimpleImputer
            imputer = SimpleImputer(strategy=imputation_strategy)
            X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train.loc[impute_indices]),
                                        index=impute_indices,
                                        columns=X_train.columns)
            
            # Combine the indices to create a complete subset
            complete_indices = non_missing_indices.union(impute_indices)
            
            # Create the final X_train_complete and y_train_complete
            X_train_complete = pd.concat([X_train.loc[non_missing_indices], X_train_imputed])
            y_train_complete = y_train.loc[X_train_complete.index]
            
            return X_train_complete, y_train_complete
        
        else:
            # Return the original non-missing subset
            X_train_complete = X_train.loc[non_missing_indices]
            y_train_complete = y_train.loc[non_missing_indices]
            
            return X_train_complete, y_train_complete

 





    def do_kfold_experiments(self, fs_type):
        y_trues = []
        errors_dfs = []
        proba_predictions_dfs = []
        distances_dfs = []

        # Raw data is the original data from MissDataset
        original_data = self.dataset_object.raw_data
        X = original_data.drop(self.dataset_object.target_col, axis=1)
        X_cols = X.columns
        X_index = X.index
        
        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        original_data = scaler.fit_transform(X)
        original_data = pd.DataFrame(original_data, columns=X_cols, index=X_index)

        print(f"\n+++++++++++++++ FEATURE SELECTION <{fs_type}> Starting k-fold experiments for {self.dataset_name} +++++++++++++++")

        for fold in range(self.n_folds):
            print(f"Processing fold {fold + 1}/{self.dataset_object.n_folds}", end='\r')

            self._init_pipelines()

            train, val, test = self.dataset_object[fold]

            y_test = test[self.dataset_object.target_col]
            y_trues += list(y_test)

            y_train = train[self.dataset_object.target_col]
            y_val = val[self.dataset_object.target_col]

            X_test = test.drop(self.dataset_object.target_col, axis=1)
            X_val = val.drop(self.dataset_object.target_col, axis=1)
            X_train = train.drop(self.dataset_object.target_col, axis=1)
            X_train = round(X_train, 8)
            
            X_train_complete, y_train_complete = self.ensure_20_percent_non_missing(X_train, y_train, imputation_strategy='mean')

            if fs_type == "RFE":

                #########################
                # RFE FEATURE SELECTION

                X_train_complete_rfe, selected_features = self.select_features_rfe( X_train_complete, y_train_complete, k=10)
                # print(selected_features_rfe)

                # # If X_train is a DataFrame and selected_features_gd is an array of column indices
                # X_train_rfe = X_train.iloc[:, selected_features_rfe]
                # X_test_rfe = X_test.iloc[:, selected_features_rfe]
                # train_indices, val_indices, test_indices = self.dataset_object.train_val_test_triples[fold]

                # # Filter self.train_not_missing to include only the selected features
                # self.train_not_missing = original_data.iloc[train_indices].iloc[:, selected_features_rfe]
                # self.test_not_missing = original_data.iloc[test_indices].iloc[:, selected_features_rfe]

                # # X_val=X_val, y_val=y_val,
                # errors_df, proba_predictions_df = self.do_experiment_one_fold(
                #     X_train=X_train_rfe, y_train=y_train,
                #     X_val=X_val, y_val=y_val,
                #     X_test=X_test_rfe, y_test=y_test
                # )

            elif fs_type == "genetic_algorithm":



                #########################
                # Genetic Algorithm FEATURE SELECTION

                selected_features = self.genetic_algorithm_feature_selection(X_train_complete, y_train_complete, k=10, n_gen=20, pop_size=50)
                # print(selected_features_ga)

                # # If X_train is a DataFrame and selected_features_gd is an array of column indices
                # X_train_ga = X_train.iloc[:, selected_features_ga]
                # X_test_ga = X_test.iloc[:, selected_features_ga]
                # train_indices, val_indices, test_indices = self.dataset_object.train_val_test_triples[fold]

                # # Filter self.train_not_missing to include only the selected features
                # self.train_not_missing = original_data.iloc[train_indices].iloc[:, selected_features_ga]
                # self.test_not_missing = original_data.iloc[test_indices].iloc[:, selected_features_ga]

                # # X_val=X_val, y_val=y_val,
                # errors_df, proba_predictions_df = self.do_experiment_one_fold(
                #     X_train=X_train_ga, y_train=y_train,
                #     X_val=X_val, y_val=y_val,
                #     X_test=X_test_ga, y_test=y_test
                # )


            


            elif fs_type == "information_gain":



                #########################
                # INFORMATION GAIN FEATURE SELECTION
                X_train_complete_ig, selected_features = self.select_features_ig(X_train_complete, y_train_complete, k=10)
                # print(type(selected_features_ig))


                # # If X_train is a DataFrame and selected_features_ig is an array of column indices
                # X_train_ig = X_train.iloc[:, selected_features_ig]

                # X_test_ig = X_test.iloc[:, selected_features_ig]
                # train_indices, val_indices, test_indices = self.dataset_object.train_val_test_triples[fold]

                # # Filter self.train_not_missing to include only the selected features
                # self.train_not_missing = original_data.iloc[train_indices].iloc[:, selected_features_ig]
                # self.test_not_missing = original_data.iloc[test_indices].iloc[:, selected_features_ig]

                # # X_val=X_val, y_val=y_val,
                # errors_df, proba_predictions_df = self.do_experiment_one_fold(
                #     X_train=X_train_ig, y_train=y_train,
                #     X_val=X_val, y_val=y_val,
                #     X_test=X_test_ig, y_test=y_test
                # )


            ##########################
            elif fs_type == "chi_square":

                
                
                X_train_complete_chi, selected_features = self.select_features_chi2(X_train_complete, y_train_complete, k=10)
                # print(selected_features_chi)

                # # If X_train is a DataFrame and selected_features_chi is an array of column indices
                # X_train_chi = X_train.iloc[:, selected_features_chi]
                # X_test_chi = X_test.iloc[:, selected_features_chi]
                # train_indices, val_indices, test_indices = self.dataset_object.train_val_test_triples[fold]

                # # Filter self.train_not_missing to include only the selected features
                # self.train_not_missing = original_data.iloc[train_indices].iloc[:, selected_features_chi]
                # self.test_not_missing = original_data.iloc[test_indices].iloc[:, selected_features_chi]

                # # X_val=X_val, y_val=y_val,
                # errors_df, proba_predictions_df = self.do_experiment_one_fold(
                #     X_train=X_train_chi, y_train=y_train,
                #     X_val=X_val, y_val=y_val,
                #     X_test=X_test_chi, y_test=y_test
                # )


            else:
                # CORRELATION COEFFICIENT FEATURE SELECTION
                X_train_complete_corr, selected_features = self.select_features_corr(X_train_complete, y_train_complete, k=10)



                # print(type(selected_features_corr))

                # # If X_train is a DataFrame and selected_features_corr is an array of column indices
                # X_train_corr = X_train.loc[:, selected_features_corr]
                # X_test_corr = X_test.loc[:, selected_features_corr]
                # train_indices, val_indices, test_indices = self.dataset_object.train_val_test_triples[fold]

                # # Filter self.train_not_missing to include only the selected features
                # self.train_not_missing = original_data.iloc[train_indices].loc[:, selected_features_corr]
                # self.test_not_missing = original_data.iloc[test_indices].loc[:, selected_features_corr]

                # # X_val=X_val, y_val=y_val,
                # errors_df, proba_predictions_df = self.do_experiment_one_fold(
                #     X_train=X_train_corr, y_train=y_train,
                #     X_val=X_val, y_val=y_val,
                #     X_test=X_test_corr, y_test=y_test
                # )

            # If X_train is a DataFrame and selected_features_corr is an array of column indices
            X_train_feat = X_train.iloc[:, selected_features]
            X_test_feat = X_test.iloc[:, selected_features]
            train_indices, val_indices, test_indices = self.dataset_object.train_val_test_triples[fold]

            # Filter self.train_not_missing to include only the selected features
            self.train_not_missing = original_data.iloc[train_indices].iloc[:, selected_features]
            self.test_not_missing = original_data.iloc[test_indices].iloc[:, selected_features]

            # X_val=X_val, y_val=y_val,
            errors_df, proba_predictions_df = self.do_experiment_one_fold(
                X_train=X_train_feat, y_train=y_train,
                X_val=X_val, y_val=y_val,
                X_test=X_test_feat, y_test=y_test
            )


            errors_dfs.append(errors_df)
            proba_predictions_dfs.append(proba_predictions_df)

        self.errors_df_total = pd.concat(errors_dfs)
        self.proba_predictions_df_total = pd.concat(proba_predictions_dfs)



        # self.errors_df_total, self.proba_predictions_df_total
        

    def do_experiment_one_fold(self, X_train, y_train, X_val, y_val, X_test, y_test):
        # print(f"Running experiments for one fold...")
        self._init_pipelines()
        proba_predictions_per_pipeline = {}
        errors_df = pd.DataFrame({})
        y_train = np.array(y_train)
        y_val = np.array(y_val)
        y_test = np.array(y_test)

        for pipeline_type in self.pipelines:
            pipeline = self.pipelines[pipeline_type]

            # Probabilistic predictions and errors
            proba_predictions, errors = self._run_one_pipeline(
                pipeline,
                pipeline_type,
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test
            )
            proba_predictions_per_pipeline[pipeline_type] = proba_predictions
            errors_df[pipeline_type] = errors

        proba_predictions_per_pipeline_2d = {
            k: np.hstack([1 - np.ravel(probas).reshape(-1, 1), np.ravel(probas).reshape(-1, 1)])
            for k, probas in proba_predictions_per_pipeline.items()
        }
        fs_test_predictions_dict = deepcopy(proba_predictions_per_pipeline_2d)

        y_test_2d = np.array(self.label_enc.transform(y_test.reshape(-1, 1)).todense())
        errors = np.abs(y_test - proba_predictions)
        predictions = np.round(proba_predictions)
        single_label_y_test = np.argmax(y_test_2d, axis=1)
        roc_auc = roc_auc_score(y_test, proba_predictions)

        metrics = {
            f'roc_auc_{self.p_miss}': round(roc_auc, 4),
            f'accuracy_{self.p_miss}': round(1 - (np.sum(np.logical_xor(predictions, single_label_y_test)) / len(predictions)), 4),
            f'f1_score_{self.p_miss}': f1_score(single_label_y_test, predictions)
        }
        self.metric_name_cols = list(metrics.keys())

        return errors_df, pd.DataFrame(proba_predictions_per_pipeline)

    def _run_one_pipeline(self, pipeline, pipeline_name, X_train, y_train, X_test, y_test):
        # print(f"\tpipeline_name: {pipeline_name}")

        pipeline.fit(X_train, y_train)
        X_train_imputed = pipeline.X_train_imputed
        proba_predictions = pipeline.predict_proba(X_test)[:, 1]
        X_test_imputed = pipeline.X_test_imputed

        if isinstance(proba_predictions, list):
            proba_predictions = proba_predictions[0]

        X_train_imputed = pd.DataFrame(X_train_imputed, columns=self.train_not_missing.columns, index=self.train_not_missing.index)
        X_test_imputed = pd.DataFrame(X_test_imputed, columns=self.test_not_missing.columns, index=self.test_not_missing.index)

        X_combined_imputed = np.vstack((X_train_imputed, X_test_imputed))
        X_combined_original = pd.concat([self.train_not_missing, self.test_not_missing]).reset_index(drop=True)

        missing_mask_train = X_train.isnull()
        missing_mask_test = X_test.isnull()
        missing_mask_combined = pd.concat([missing_mask_train, missing_mask_test]).reset_index(drop=True)

        #initializing the imputation evaluation
        imputed_RMSE = {}
        imputed_roc_auc = {}

        # Adding Accuracy and MAE dictionaries
        imputed_accuracy = {}
        imputed_mae = {}

        for col in X_combined_original.columns:
            original_data = X_combined_original[col][missing_mask_combined[col]]
            imputed_data = X_combined_imputed[:, X_combined_original.columns.get_loc(col)][missing_mask_combined[col]]
            if not original_data.empty:
                if original_data.dtype.kind in 'iuf':
                    # RMSE for numerical data types only 
                    rmse = np.sqrt(mean_squared_error(original_data, imputed_data))
                    imputed_RMSE[col] = rmse

                    # MAE for numerical data types only
                    mae = mean_absolute_error(original_data, imputed_data)
                    imputed_mae[col] = mae

                else:
                    original_data = original_data.dropna()
                    imputed_data = pd.Series(imputed_data).dropna()
                    if len(original_data.unique()) > 1:
                        # ROC AUC for categorical data types only
                        roc_auc = roc_auc_score(
                            original_data.astype('category').cat.codes,
                            imputed_data.astype('category').cat.codes
                        )
                        imputed_roc_auc[col] = roc_auc
                        
                        # Accuracy for categorical data types only NEED TO ADD THIS

                        

        imputed_evals = {
            f'RMSE_{self.p_miss}': imputed_RMSE,
            f'AUC_ROC_{self.p_miss}': imputed_roc_auc,
            f'Accuracy_{self.p_miss}': imputed_accuracy,
            f'MAE_{self.p_miss}': imputed_mae
        }
        if pipeline_name not in self.imputed_evals:
            self.imputed_evals[pipeline_name] = []
        self.imputed_evals[pipeline_name].append(imputed_evals)

        y_test_2d = np.array(self.label_enc.transform(y_test.reshape(-1, 1)).todense())
        errors = np.abs(y_test - proba_predictions)
        predictions = np.round(proba_predictions)
        single_label_y_test = np.argmax(y_test_2d, axis=1)
        roc_auc = roc_auc_score(y_test, proba_predictions)

        metrics = {
            f'roc_auc_{self.p_miss}': round(roc_auc, 4),
            f'accuracy_{self.p_miss}': round(1 - (np.sum(np.logical_xor(predictions, single_label_y_test)) / len(predictions)), 4),
            f'f1_score_{self.p_miss}': f1_score(single_label_y_test, predictions)
        }
        self.prediction_metrics[pipeline_name].append(list(metrics.values()))

        return proba_predictions, errors

    def _mean_imputer(self, X_train, y_train, X_test, y_test):
        column_name = "Mean Imputation"
        train_means = X_train.mean()

        X_train_imputed = X_train.fillna(train_means)
        X_test_imputed = X_test.fillna(train_means)

        X_train_imputed = pd.DataFrame(X_train_imputed, columns=self.train_not_missing.columns, index=self.train_not_missing.index)
        X_test_imputed = pd.DataFrame(X_test_imputed, columns=self.test_not_missing.columns, index=self.test_not_missing.index)

        X_combined_imputed = np.vstack((X_train_imputed, X_test_imputed))
        X_combined_original = pd.concat([self.train_not_missing, self.test_not_missing]).reset_index(drop=True)

        missing_mask_train = X_train.isnull()
        missing_mask_test = X_test.isnull()
        missing_mask_combined = pd.concat([missing_mask_train, missing_mask_test]).reset_index(drop=True)

        imputed_RMSE = {}
        imputed_roc_auc = {}

        for col in X_combined_original.columns:
            original_data = X_combined_original[col][missing_mask_combined[col]]
            imputed_data = X_combined_imputed[:, X_combined_original.columns.get_loc(col)][missing_mask_combined[col]]
            if not original_data.empty:
                if original_data.dtype.kind in 'iuf':
                    rmse = np.sqrt(mean_squared_error(original_data, imputed_data))
                    imputed_RMSE[col] = rmse
                else:
                    original_data = original_data.dropna()
                    imputed_data = pd.Series(imputed_data).dropna()
                    if len(original_data.unique()) > 1:
                        roc_auc = roc_auc_score(
                            original_data.astype('category').cat.codes,
                            imputed_data.astype('category').cat.codes
                        )
                        imputed_roc_auc[col] = roc_auc

        imputed_evals = {
            f'RMSE_{self.p_miss}': imputed_RMSE,
            f'AUC_ROC_{self.p_miss}': imputed_roc_auc
            
        }
        if column_name not in self.imputed_evals:
            self.imputed_evals[column_name] = []
        self.imputed_evals[column_name].append(imputed_evals)

    def run(self):
        self.do_kfold_experiments(self.fs_type)

        prediction_metrics_df = pd.DataFrame({})
        imputed_evals_df= pd.DataFrame({})

        # prediction_metrics_filename_df = {}
        # imputed_evals_filename_df={}

        for m in self.prediction_metrics:
            # print("-----------------THIS IS M-----------------")
            # print(m)
            prediction_metrics_df[m] = np.mean(self.prediction_metrics[m], axis=0)
            # print("-----------------THIS IS dataframe-----------------")
            # print(type(prediction_metrics_df))
            # print(prediction_metrics_df[m])

        prediction_metrics_df.index = self.metric_name_cols

        for pipeline_name, evaluations in self.imputed_evals.items():
            pipeline_averages = {}
            # Collect all metric values
            for eval_dict in evaluations:
                for metric, values in eval_dict.items():
                    if metric not in pipeline_averages:
                        pipeline_averages[metric] = []
                    pipeline_averages[metric].extend(values.values())  # Flatten values if they are dicts

            # Calculate mean and update DataFrame
            for metric, values in pipeline_averages.items():
                # There are 5 folds. Get the mean of each feature imputation evaluation for all 5 folds to get one average
                mean_value = np.mean(values)  # Ensure this is a scalar
                imputed_evals_df.loc[metric,pipeline_name] = mean_value  # Use loc for setting value in potentially new row/column

        return prediction_metrics_df, self.errors_df_total, self.proba_predictions_df_total, imputed_evals_df

    def prepare_data(self, dataset, target_col):
        X = dataset.drop(columns=[target_col])
        y = dataset[target_col]
        y = y.astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def perform_imputations(self, X_train, X_test, y_train, y_test):
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

        knn_imputer = self.imputation_methods['KNN']
        self.X_train_knn = knn_imputer.fit_transform(X_train)
        self.X_test_knn = knn_imputer.transform(X_test)
        self.imputations['KNN'] = (self.X_train_knn, self.X_test_knn)

        mean_imputer = self.imputation_methods['Mean']
        self.X_train_mean = mean_imputer.fit_transform(X_train)
        self.X_test_mean = mean_imputer.transform(X_test)
        self.imputations['Mean'] = (self.X_train_mean, self.X_test_mean)

        mask_train = ~np.isnan(X_train).any(axis=1)
        mask_test = ~np.isnan(X_test).any(axis=1)
        self.X_train_cc = X_train[mask_train]
        self.y_train_cc = y_train[mask_train]
        self.X_test_cc = X_test[mask_test]
        self.y_test_cc = y_test[mask_test]
        self.imputations['CompleteCase'] = (self.X_train_cc, self.X_test_cc, self.y_train_cc, self.y_test_cc)

    def evaluate_imputations(self):
        for method, (X_train_imp, X_test_imp) in self.imputations.items():
            rmse = mean_squared_error(self.y_test_cc, X_test_imp, squared=False)
            mae = mean_absolute_error(self.y_test_cc, X_test_imp)
            accuracy = accuracy_score(self.y_test_cc, X_test_imp)
            auc = roc_auc_score(self.y_test_cc, X_test_imp)
            self.imputation_results.append({
                'Method': method,
                'RMSE': rmse,
                'MAE': mae,
                'Accuracy': accuracy,
                'AUC': auc
            })

    def save_results(self):
        imputation_df = pd.DataFrame(self.imputation_results)
        classification_df = pd.DataFrame(self.classification_results)

        imputation_df.to_csv(os.path.join(self.results_dir, 'imputation_evaluation.csv'), index=False)
        classification_df.to_csv(os.path.join(self.results_dir, 'classification_evaluation.csv'), index=False)

    def select_features_ig(self, X, y, k=10):
        ig_selector = SelectKBest(mutual_info_classif, k=k)
        X_new = ig_selector.fit_transform(X, y)
        return X_new, ig_selector.get_support(indices=True)

    # def select_features_corr(self, X, y, k=10):
    #     corr = X.corrwith(y)
    #     top_features = corr.abs().sort_values(ascending=False).head(k).index
    #     return X[top_features], top_features

    def select_features_corr(self, X, y, k=10):
        """
        Selects top k features based on correlation with the target variable y.
        
        Parameters:
        X (pd.DataFrame): The input features.
        y (pd.Series): The target variable.
        k (int): The number of top features to select.
        
        Returns:
        pd.DataFrame: The subset of X containing only the top k features.
        np.ndarray: The integer indices of the selected top k features.
        """

        corr = X.corrwith(y)
        top_features = corr.abs().sort_values(ascending=False).head(k).index
        top_feature_indices = X.columns.get_indexer(top_features)
        return X[top_features], top_feature_indices


    def select_features_chi2(self, X, y, k=10):
        chi2_selector = SelectKBest(chi2, k=k)
        X_new = chi2_selector.fit_transform(X, y)
        return X_new, chi2_selector.get_support(indices=True)
    
    def genetic_algorithm_feature_selection(self, X, y, k=10, n_gen=20, pop_size=50):

        X = X.to_numpy()
        y = y.to_numpy()
        clf = SVC(gamma='auto')

        evolved_estimator = GAFeatureSelectionCV(
            estimator=clf,
            cv=3,
            scoring="accuracy",
            population_size=5,
            generations=20,
            n_jobs=-1,
            verbose=False,
            keep_top_k=2,
            elitism=True,
        )

        evolved_estimator.fit(X, y)
        features = evolved_estimator.get_support(indices=True)


        return features






    # def genetic_algorithm_feature_selection(self, X, y, k=10, n_gen=20, pop_size=50):
    #     def evaluate(individual):
    #         selected_features = [idx for idx, val in enumerate(individual) if val == 1]
    #         if len(selected_features) == 0:
    #             return 0,
    #         X_selected = X[:, selected_features]
    #         clf = LogisticRegression(solver='liblinear')
    #         scores = cross_val_score(clf, X_selected, y, cv=5)
    #         return scores.mean(),
        
    #     n_features = X.shape[1]
    #     creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    #     creator.create("Individual", list, fitness=creator.FitnessMax)

    #     toolbox = base.Toolbox()
    #     toolbox.register("attr_bool", random.randint, 0, 1)
    #     toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_features)
    #     toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
    #     toolbox.register("evaluate", evaluate)
    #     toolbox.register("mate", tools.cxTwoPoint)
    #     toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    #     toolbox.register("select", tools.selTournament, tournsize=3)
        
    #     population = toolbox.population(n=pop_size)
    #     #broken line
    #     algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=n_gen, verbose=False)
        
    #     top_individual = tools.selBest(population, 1)[0]
    #     selected_features = [idx for idx, val in enumerate(top_individual) if val == 1]
    #     return X[:, selected_features], selected_features
    



    def select_features_rfe(self, X, y, k=10):
        model = LogisticRegression(solver='liblinear')
        rfe = RFE(model, n_features_to_select=k)
        X_new = rfe.fit_transform(X, y)
        return X_new, rfe.get_support(indices=True)
    
    def train_lightgbm(self, X_train, y_train, X_test, y_test):
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

        params = {
            'objective': 'binary',
            'metric': 'binary_error',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9
        }

        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=100,
                        valid_sets=lgb_test,
                        early_stopping_rounds=10)
        y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
        y_pred = np.round(y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy}')






        



