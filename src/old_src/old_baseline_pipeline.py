import os
import pickle
from datetime import datetime
from itertools import product
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error, roc_auc_score, precision_score, recall_score, confusion_matrix
from sklearn.svm import SVC
from tqdm import tqdm
from new_base import ClassifierWithImputation
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestRegressor, StackingClassifier, StackingRegressor, VotingClassifier





import xgboost as xgb

from data_loaders import MedicalDataset, MissDataset, DataLoadersEnum
from imputer_models import Imputers

# warnings.simplefilter(action='ignore', category=FutureWarning)


class BaselinePipeline:

    # construct pipelines
    def _init_pipelines(self, classifier_pool=None):

        if classifier_pool is None:
            xgb_params = {'n_jobs': 1, 'max_depth': 4, 'n_estimators': 50, 'verbosity': 0}
            rf_params = {'n_jobs': 1, 'max_depth': 4, 'n_estimators': 50, 'verbose': 0}
            # knn_params = {'n_jobs': 1, 'weights': 'distance', 'n_neighbors': 5}

            ################################################################################################################################
            # Pipeline Prep
            models = [
                SVC(probability=True),
                RandomForestClassifier(),
                xgb.XGBClassifier()
            ]
            imputers = [
                KNNImputer(n_neighbors=5),
                SimpleImputer(strategy='mean'),
                IterativeImputer(RandomForestRegressor(**rf_params)),
            ]
            ################################################################################################################################

            clf_imputer_pairs = product(models, imputers)
            pipelines_list = [
                ClassifierWithImputation(
                    estimator=clf,
                    imputer=imp
                )
                for clf, imp in clf_imputer_pairs
            ]
            pipelines = {
                'Estim(' + p.estimator_name + ')_Imputer(' + p.imputer_name + ')': p
                for p in pipelines_list
            }



        else:
            pipelines = classifier_pool

        assert isinstance(pipelines, dict), 'The Classifier Pool (Pipelines) must be a dictionary, not ' + str(
            type(pipelines))
        self.pipelines = pipelines
        self.unfitted_pipelines = deepcopy(self.pipelines)

       

        







    # def _init_sub_pipelines(self, classifier_pool=None):


    # def _run_one_pipeline(self, pipeline, pipeline_name, X_train, y_train, X_test, y_test):

    
    # def _mean_imputer(self, X_train, y_train, X_test, y_test):



    # def do_experiment_one_fold(self, X_train, y_train, X_val, y_val, X_test, y_test):


    

    # def do_kfold_experiments(self):
  


################################################################################################################################
    # if task_type == 'classification':
    #             #CUSTOM NAME
    #             dataset_name="testing MNAR"
    #             experiment = CustomClassificationExperiment(
    #                 dataset=dataset, dataset_name=dataset_name, 
    #                 exp_type=params['missing_mechanism'],
    #                 name=name
    #             )

    # def __init__(self, dataset_object, dataset_name, missing_mechanism):


    def __init__(
            self,
            dataset_object: MissDataset,
            missing_mechanism='mcar',
            dataset_name='',
            name='Experiment_' + str(datetime.now()),
            base_dir=None,
            classifier_pool=None,
            random_state=42
    ):
        self.dataset_object = dataset_object
        print("################################################################################################################################")
        print(f"This is the dataset_object.targetCol -> {self.dataset_object.target_col} <- in BaselinePipeline")
        print("################################################################################################################################")
        self.p_miss=self.dataset_object.p_miss
        self.n_folds = self.dataset_object.n_folds
        print(" This is the n_folds FfFFFFFFFFFFFFFF================================================================================================================================================================================================")
        print(self.n_folds)
        self.label_enc = OneHotEncoder()
        self.label_enc.fit(np.array(self.dataset_object.data[self.dataset_object.target_col]).reshape(-1, 1))



        
        # ################################################################################################################################
        # # Pipeline Prep
        self.dataset = dataset_object.data
        self.dataset_name = dataset_name
        # self.missing_mechanism = missing_mechanism
        self.imputations = {}
        self.imputation_results = []
        self.classification_results = []
        # ################################################################################################################################


        ################################################################################################################################
        # Naming the directory
        self.results_dir = os.path.join('results', dataset_name, missing_mechanism)
        os.makedirs(self.results_dir, exist_ok=True)
        print(f"Name of the experiment directory: {self.results_dir}")
        ################################################################################################################################
        self._init_pipelines(classifier_pool=classifier_pool)
        # # Add this call to the new data preparation method
        # self.prepare_data()
        self.metrics = {p: [] for p in self.pipelines}
    
        self.imputed_evals = {p: [] for p in self.pipelines}
        print(f"imputed_evals: {self.imputed_evals}")



    def do_kfold_experiments(self):
        y_trues = []
        errors_dfs = []
        proba_predictions_dfs = []
        distances_dfs = []

        original_data = self.dataset_object.raw_data  # This should be your original dataset before any missing values are introduced.
        
        # Only X data, no target
        X = original_data.drop(self.dataset_object.target_col, axis=1)
        X_cols = X.columns
        X_index = X.index
        scaler = MinMaxScaler(feature_range=(0, 1))
        original_data = scaler.fit_transform(X)
        #original data as a dataframe
        original_data = pd.DataFrame(original_data, columns=X_cols, index=X_index)





        print("\nStarting k-fold experiments...")  # Indicate start of k-fold processing
        # Note that self.n_folds is the same as self.dataset_object.n_folds
        for fold in range(self.n_folds):
            print(f"\nProcessing fold {fold + 1}/{self.dataset_object.n_folds}")  # Print current fold
            ###################
            #need to build the pipelines
            self._init_pipelines()
            ###################
            train, val, test = self.dataset_object[fold]
            # X_train, X_test, y_train, y_test=self.prepare_data(self.dataset[fold], self.dataset_object.target_col)
            # print(f"train before imputation: {train}")
            y_test = test[self.dataset_object.target_col]
            y_trues += list(y_test)

            y_train = train[self.dataset_object.target_col]
            y_val = val[self.dataset_object.target_col]

            X_test = test.drop(self.dataset_object.target_col, axis=1)
            X_val = val.drop(self.dataset_object.target_col, axis=1)
            X_train = train.drop(self.dataset_object.target_col, axis=1)
            X_train = round(X_train, 8)
            ###################

            train_indices, val_indices, test_indices = self.dataset_object.train_val_test_triples[fold]  # Access the first split as an example
            



            ######`#################################################################`
            #NOT USED
            # Extract non-missing parts using indices
            self.train_not_missing = original_data.iloc[train_indices]
            self.val_not_missing = original_data.iloc[val_indices]
            self.test_not_missing = original_data.iloc[test_indices]
            ######`#################################################################`
            
            
            errors_df, proba_predictions_df = self.do_experiment_one_fold(
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                X_test=X_test, y_test=y_test
            )

            errors_dfs.append(errors_df)
            proba_predictions_dfs.append(proba_predictions_df)
            # for top_n in weights_sets.keys():
            #     all_dew_weights[top_n].append(weights_sets[top_n])
            # distances_dfs.append(distances_df)

        # all_cols = list(self.pipelines.keys()) + [
        #     'Uniform Model Averaging',
        #     str(type(self.clf_stacked)),
        # ]
        # all_cols += [
        #     str(type(self.clf_dew)) + '_top_' + str(top_n)
        #     for top_n in self.clf_dew.n_top_to_choose
        # ]

        errors_df_total = pd.concat(errors_dfs)
        self.errors_df_total = errors_df_total

        # for top_n in all_dew_weights.keys():
        #     weights = np.vstack(all_dew_weights[top_n])
        #     self.weights_dfs[top_n] = pd.DataFrame(weights)

        self.proba_predictions_df_total = pd.concat(proba_predictions_dfs)
        self.distances_df_total = pd.concat(distances_dfs)
        print("Made it to the end of the kfold experiments")


    def do_experiment_one_fold(self, X_train, y_train, X_val, y_val, X_test, y_test):

        print(f"\nRunning experiments for one fold...")  # Indicate experiments for a single fold
        # run baselines
        self._init_pipelines()
        proba_predictions_per_pipeline = {}
        errors_df = pd.DataFrame({})
        y_train = np.array(y_train)
        y_val = np.array(y_val)
        y_test = np.array(y_test)

        pipelines_start = datetime.now()
        for pipeline_type in self.pipelines:
            pipeline = self.pipelines[pipeline_type]

            
            proba_predictions, errors = self._run_one_pipeline(
                pipeline,
                pipeline_type,
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test
            )
            # self.imputed_X_train = X_train_imputed  # Store imputed data Dylan
            proba_predictions_per_pipeline[pipeline_type] = proba_predictions
            errors_df[pipeline_type] = errors
            self._mean_imputer(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test
            )
        print('pipelines completed in ' + str(datetime.now() - pipelines_start))

        proba_predictions_per_pipeline_2d = {
            k: np.hstack([1 - np.ravel(probas).reshape(-1, 1), np.ravel(probas).reshape(-1, 1)])
            for k, probas in proba_predictions_per_pipeline.items()
        }
        baseline_test_predictions_dict = deepcopy(proba_predictions_per_pipeline_2d)

        # dstacked_predictions = np.dstack(list(proba_predictions_per_pipeline.values()))
        # dstacked_predictions = dstacked_predictions[0]  # shape[1,x,y] -> shape[x,y]
        # proba_predictions = np.mean(
        #     dstacked_predictions,
        #     axis=-1
        # )
        y_test_2d = np.array(self.label_enc.transform(y_test.reshape(-1, 1)).todense())
        errors = np.abs(y_test - proba_predictions)
        proba_predictions_per_pipeline['Uniform Model Averaging'] = proba_predictions
        errors_df['Uniform Model Averaging'] = errors
        predictions = np.round(proba_predictions)
        single_label_y_test = np.argmax(y_test_2d, axis=1)
        roc_auc = roc_auc_score(y_test, proba_predictions)

        
        metrics = {}
        metrics[f'roc_auc_{self.p_miss}'] = round(roc_auc, 4)
        accuracy = 1 - (np.sum(np.logical_xor(predictions, single_label_y_test)) / len(predictions))
        metrics[f'accuracy_{self.p_miss}'] = round(accuracy, 4)
        metrics[f'f1_score_{self.p_miss}' ] = f1_score(y_test, predictions)
        self.metrics['Uniform Model Averaging'].append(list(metrics.values()))
        # need to store metrics names somewhere accessible; not ideal way to do this but it works
        self.metric_type_cols = list(metrics.keys())

        
        # self.imputed_X_val = X_val_imputed  # Store imputed data Dylan
        proba_predictions_per_pipeline[str(type(self.clf_stacked))] = proba_predictions
        errors_df[str(type(self.clf_stacked))] = errors

        # # run DEW classifiers
        # print('running DEW models')
        # dew_start = datetime.now()
        # self.clf_dew.set_baseline_test_predictions(baseline_test_predictions_dict)

        # self.clf_dew.fit(X_val, y_val)
        # proba_predictions_sets, weights_sets, distances = self.clf_dew.predict_proba(X_test)

        # distances_df = pd.DataFrame(distances)

        # for top_n, proba_predictions in proba_predictions_sets.items():
        #     proba_predictions = proba_predictions[:, 1]  # remove slice if multiclass
        #     proba_predictions_per_pipeline['dew_top_' + str(top_n)] = proba_predictions
        #     y_test_2d = np.array(self.label_enc.transform(y_test.reshape(-1, 1)).todense())

        #     errors = np.abs(proba_predictions - y_test)
        #     errors_df['dew_top_' + str(top_n)] = errors
        #     predictions = np.round(proba_predictions)
        #     single_label_y_test = np.argmax(y_test_2d, axis=1)
        #     roc_auc = roc_auc_score(y_test, proba_predictions)
        #     metrics = {}
        #     metrics[f'roc_auc_{self.p_miss}'] = round(roc_auc, 4)
        #     accuracy = 1 - (np.sum(np.logical_xor(predictions, single_label_y_test)) / len(predictions))
        #     metrics[f'accuracy_{self.p_miss}'] = round(accuracy, 4)
        #     metrics[f'f1_score_{self.p_miss}'] = f1_score(single_label_y_test, predictions)
        #     self.metrics['dew_top_' + str(top_n)].append(list(metrics.values()))

        # == the following two lines are only necessary if multiclass
        # for k in proba_predictions_per_pipeline.keys():
        #     proba_predictions_per_pipeline[k] = proba_predictions_per_pipeline[k][:, 1]
        # TODO{i don't think those lines are correct, should do another way. but not doing multiclass now so later?}
        # print('DEW completed in ' + str(datetime.now() - dew_start))
        print("CLASSIFICATION")
        # print(self.metrics)

        

        return errors_df, pd.DataFrame(proba_predictions_per_pipeline)





    def _run_one_pipeline(self, pipeline, pipeline_name, X_train, y_train, X_test, y_test):
        print(f"pipeline_name: {pipeline_name}")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        # print("This is Test ")
        # print(y_train)
        print("=========================================")
        print("=========================================")
        print("=========================================")
        print("=========================================")
        
        # print(f"X_test shape: {X_test.shape}")
        # print(type(pipeline))
        if pipeline_name != "<class 'sklearn.ensemble._stacking.StackingClassifier'>":
            pipeline.fit(X_train, y_train)
            X_train_imputed = pipeline.X_train_imputed
            
            proba_predictions = pipeline.predict_proba(X_test)[:, 1]
            X_test_imputed = pipeline.X_test_imputed
            
            if isinstance(proba_predictions, list):
                proba_predictions = proba_predictions[0]

            X_train_imputed = pd.DataFrame(X_train_imputed, columns=self.train_not_missing.columns, index=self.train_not_missing.index)
            X_test_imputed = pd.DataFrame(X_test_imputed, columns=self.test_not_missing.columns, index=self.test_not_missing.index)

            # Dylan added Combine imputed training and testing data for evaluation
            X_combined_imputed = np.vstack((X_train_imputed, X_test_imputed))
            X_combined_original = pd.concat([self.train_not_missing, self.test_not_missing]).reset_index(drop=True)

            # Create masks for original missing data locations in train and test
            missing_mask_train = X_train.isnull()
            missing_mask_test = X_test.isnull()
            missing_mask_combined = pd.concat([missing_mask_train, missing_mask_test]).reset_index(drop=True)

            imputed_RMSE = {}
            imputed_roc_auc = {}

            # Evaluate imputation on numeric and categorical data separately
            for col in X_combined_original.columns:
                original_data = X_combined_original[col][missing_mask_combined[col]]
                imputed_data = X_combined_imputed[:, X_combined_original.columns.get_loc(col)][missing_mask_combined[col]]
                if not original_data.empty:
                    if original_data.dtype.kind in 'iuf':  # Numeric evaluation: RMSE
                        rmse = np.sqrt(mean_squared_error(original_data, imputed_data))
                        imputed_RMSE[col] = rmse
                    else:  # Categorical evaluation: ROC AUC
                        original_data = original_data.dropna()
                        imputed_data = pd.Series(imputed_data).dropna()
                        if len(original_data.unique()) > 1:
                            roc_auc = roc_auc_score(original_data.astype('category').cat.codes, imputed_data.astype('category').cat.codes)
                            imputed_roc_auc[col] = roc_auc

            imputed_evals = {
                f'RMSE_{self.p_miss}': imputed_RMSE,
                f'AUC_ROC_{self.p_miss}': imputed_roc_auc
            }
            if not hasattr(self, 'imputed_evals'):
                self.imputed_evals = {}
            if pipeline_name not in self.imputed_evals:
                self.imputed_evals[pipeline_name] = []
            self.imputed_evals[pipeline_name].append(imputed_evals)

        elif pipeline_name == "<class 'sklearn.ensemble._stacking.StackingClassifier'>":
            pipeline.fit(X_train, y_train)
            proba_predictions = pipeline.predict_proba(X_test)[:, 1]
            if isinstance(proba_predictions, list):
                proba_predictions = proba_predictions[0]

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
        self.metrics[pipeline_name].append(list(metrics.values()))

        return proba_predictions, errors
    


    def _mean_imputer(self, X_train, y_train, X_test, y_test):
        column_name="Mean Imputation"
        print(column_name)

        train_means = X_train.mean()

        # Apply mean imputation to the training data
        X_train_imputed = X_train.fillna(train_means)

        # Apply the same train means to the test data to maintain consistency
        X_test_imputed = X_test.fillna(train_means)

        X_train_imputed = pd.DataFrame(X_train_imputed, columns=self.train_not_missing.columns, index=self.train_not_missing.index)
        X_test_imputed = pd.DataFrame(X_test_imputed, columns=self.test_not_missing.columns, index=self.test_not_missing.index)

        # Dylan added Combine imputed training and testing data for evaluation
        X_combined_imputed = np.vstack((X_train_imputed, X_test_imputed))
        X_combined_original = pd.concat([self.train_not_missing, self.test_not_missing]).reset_index(drop=True)

        # Create masks for original missing data locations in train and test
        missing_mask_train = X_train.isnull()
        missing_mask_test = X_test.isnull()
        missing_mask_combined = pd.concat([missing_mask_train, missing_mask_test]).reset_index(drop=True)

        imputed_RMSE = {}
        imputed_roc_auc = {}

        # Evaluate imputation on numeric and categorical data separately
        for col in X_combined_original.columns:
            original_data = X_combined_original[col][missing_mask_combined[col]]
            imputed_data = X_combined_imputed[:, X_combined_original.columns.get_loc(col)][missing_mask_combined[col]]
            if not original_data.empty:
                if original_data.dtype.kind in 'iuf':  # Numeric evaluation: RMSE
                    rmse = np.sqrt(mean_squared_error(original_data, imputed_data))
                    imputed_RMSE[col] = rmse
                else:  # Categorical evaluation: ROC AUC
                    original_data = original_data.dropna()
                    imputed_data = pd.Series(imputed_data).dropna()
                    if len(original_data.unique()) > 1:
                        roc_auc = roc_auc_score(original_data.astype('category').cat.codes, imputed_data.astype('category').cat.codes)
                        imputed_roc_auc[col] = roc_auc

        imputed_evals = {
            f'RMSE_{self.p_miss}': imputed_RMSE,
            f'AUC_ROC_{self.p_miss}': imputed_roc_auc
        }
        if not hasattr(self, 'imputed_evals'):
            self.imputed_evals = {}
        if column_name not in self.imputed_evals:
            self.imputed_evals[column_name] = []
        self.imputed_evals[column_name].append(imputed_evals)





    # def _run_one_pipeline(self, pipeline, pipeline_name, X_train, y_train, X_test, y_test):
    #     print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    #     print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    #     print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    #     print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    #     print("This is X_Test in do_experiment_one_fold")
    #     print(type(X_test))
    #     print("=========================================")
    #     print("=========================================")
    #     print("=========================================")
    #     print("=========================================")
    #     print(f"X_test shape: {X_test.shape}")
    #     pipeline.fit(X_train, y_train)
    #     predictions = pipeline.predict(X_test)
    #     if isinstance(predictions, list):
    #         predictions = predictions[0]

    #     errors = np.abs(y_test - predictions)
    #     metrics = {}
    #     metrics['mae'] = round(np.mean(errors), 4)
    #     self.metrics[pipeline_name].append(list(metrics.values()))

    #     return predictions, errors


    

    def run(self):
        self.do_kfold_experiments()



        # self.prepare_data()
        # return the Xtrain and y_train and X_test and y_test here and pass to the other functions below
        


        # self.perform_imputations(X_train, X_test, y_train, y_test)
        # self.evaluate_imputations()
        # self.run_classifications()
        # self.save_results()

    def prepare_data(self, dataset, target_col):
        """
        Prepare the data by splitting into training and testing sets.
        """

        X = dataset.drop(columns=[target_col])  
        y = dataset[target_col]



        print("================================================================================================================================================================================================")

        print(type(y))
        print(y)

        print("================================================================================================================================================================================================")
        
        # Ensure y is integer type
        y = y.astype(int)
        
        # Perform train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # # Store the split data back into the object
        # self.dataset = (X_train, X_test, y_train, y_test)
        return X_train, X_test, y_train, y_test


    def perform_imputations(self,X_train, X_test, y_train, y_test):
        print("This is the dataset in perform_imputations")
        # print(self.dataset)
        # X_train, X_test, y_train, y_test = self.dataset

        print("================================================================================================================================================================================================")

        # print(y_train)

        print("================================================================================================================================================================================================")

        # Ensure y_train and y_test are integer type
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

        # KNN Imputation
        knn_imputer = self.imputation_methods['KNN']
        self.X_train_knn = knn_imputer.fit_transform(X_train)
        self.X_test_knn = knn_imputer.transform(X_test)
        self.imputations['KNN'] = (self.X_train_knn, self.X_test_knn)

        # Mean Imputation
        mean_imputer = self.imputation_methods['Mean']
        self.X_train_mean = mean_imputer.fit_transform(X_train)
        self.X_test_mean = mean_imputer.transform(X_test)
        self.imputations['Mean'] = (self.X_train_mean, self.X_test_mean)

        # # RandomForest Imputation using IterativeImputer
        # from sklearn.experimental import enable_iterative_imputer
        # from sklearn.impute import IterativeImputer
        # rf_imputer = IterativeImputer(estimator=RandomForestClassifier(), max_iter=10, random_state=0)
        # print("This is the X_train in RandomForest imputation")
        # print(X_train)



        # print("THIS IS THE FIT TRANSFORM ================================================================================================================================================================================================")
        # X_train_rf = rf_imputer.fit_transform(X_train)
        # X_test_rf = rf_imputer.transform(X_test)
        # self.imputations['RandomForest'] = (X_train_rf, X_test_rf)

        # Complete Case Analysis
        mask_train = ~np.isnan(X_train).any(axis=1)
        mask_test = ~np.isnan(X_test).any(axis=1)
        self.X_train_cc = X_train[mask_train]
        self.y_train_cc = y_train[mask_train]
        self.X_test_cc = X_test[mask_test]
        self.y_test_cc = y_test[mask_test]
        self.imputations['CompleteCase'] = (self.X_train_cc, self.X_test_cc, self.y_train_cc, self.y_test_cc)


    def evaluate_imputations(self):
        # X_train, X_test, y_train, y_test = self.dataset
        print(" This is the imputations ================================================================================================================================================================================================")
        print(self.imputations.keys())

        for method, (X_train_imp, X_test_imp) in self.imputations.items():
            rmse = mean_squared_error(y_test, X_test_imp, squared=False)
            mae = mean_absolute_error(y_test, X_test_imp)
            accuracy = accuracy_score(y_test, X_test_imp)
            auc = roc_auc_score(y_test, X_test_imp)
            self.imputation_results.append({
                'Method': method,
                'RMSE': rmse,
                'MAE': mae,
                'Accuracy': accuracy,
                'AUC': auc
            })

    # def run_classifications(self):
    #     X_train, X_test, y_train, y_test = self.dataset

    #     for method, (X_train_imp, X_test_imp) in self.imputations.items():
    #         for clf_name, clf in self.classifiers.items():
    #             clf.fit(X_train_imp, y_train)
    #             y_pred = clf.predict(X_test_imp)
    #             accuracy = accuracy_score(y_test, y_pred)
    #             precision = precision_score(y_test, y_pred, average='binary')
    #             recall = recall_score(y_test, y_pred, average='binary')
    #             tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    #             specificity = tn / (tn + fp)
    #             self.classification_results.append({
    #                 'Imputation': method,
    #                 'Classifier': clf_name,
    #                 'Accuracy': accuracy,
    #                 'Precision': precision,
    #                 'Recall': recall,
    #                 'Specificity': specificity
    #             })

    def save_results(self):
        imputation_df = pd.DataFrame(self.imputation_results)
        classification_df = pd.DataFrame(self.classification_results)

        imputation_df.to_csv(os.path.join(self.results_dir, 'imputation_evaluation.csv'), index=False)
        classification_df.to_csv(os.path.join(self.results_dir, 'classification_evaluation.csv'), index=False)
