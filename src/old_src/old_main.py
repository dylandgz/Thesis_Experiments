import os
from data_loaders import MedicalDataset, MissDataset, DataLoadersEnum
from itertools import product
from datetime import datetime
from tqdm import tqdm
from copy import deepcopy
import pickle

from baseline_pipeline import BaselinePipeline
from feature_selection_pipeline import FeatureSelectionPipeline




# # driver for classification task
# def run_custom_experiments(data, dataset_name, miss_param_dict, target_col, task_type='classification'):
#     miss_param_grid = list(product(*tuple(miss_param_dict.values())))  # print:[(0.3, 'MNAR', 'logistic', 0.3, None)]
#     print(f"total length of miss parameters= {len(miss_param_grid)}")
#     print(f"miss_param_grid= {miss_param_grid}")
#     param_lookup_dict = {}
#     metrics_dfs = []
#     imputation_eval_results = []
#     miss_type = miss_param_dict['missing_mechanism']
#     name = miss_type[0] + '_Experiment_' + str(datetime.now())
#     with tqdm(total=len(miss_param_grid)) as pbar:
#         for i, params in enumerate(miss_param_grid):
#             print(f"\nStarting experiment with params: {params}")  # Print current experiment parameters
#             data_copy = deepcopy(data)
#             params = {
#                 k: p 
#                 for k, p in zip(list(miss_param_dict.keys()), params)
#             }
#             param_lookup_dict[i] = params

            
#             #Create missing data as dataset
#             dataset = MissDataset( #returned an object
#                 data=data_copy,
#                 target_col=target_col,
#                 n_folds=5,
#                 **params,
#             )
#             print("This is the dataset.data")
#             print(dataset.data)
#             print("Number of missing values in the dataset")
#             print(dataset.data.isnull().sum())


def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def run_custom_experiments(original_data, dataset_name, missing_param_dict, target_col, task_type='classification'):
    missing_param_grid = list(product(*tuple(missing_param_dict.values())))
    print(f"total length of missing parameters= {len(missing_param_grid)}")
    print(f"miss_param_grid= {missing_param_grid}")
    param_lookup_dict = {}
    metrics_dfs = []
    imputation_eval_results = []
    miss_type = missing_param_dict['missing_mechanism']
    name = miss_type[0] + '_Experiment_' + str(datetime.now()).replace(':', '-').replace(' ', '_')
    
    # base_dir = os.path.join('experiments', name)
    # os.makedirs(base_dir, exist_ok=True)
    # imputer_models_dir = os.path.join(base_dir, 'imputer_models')
    # os.makedirs(imputer_models_dir, exist_ok=True)

    with tqdm(total=len(missing_param_grid)) as pbar:
        for i, params in enumerate(missing_param_grid):
            print(f"\nStarting experiment with params: {params}")  # Print current experiment parameters
            original_data_copy = deepcopy(original_data)
            params = {
                k: p 
                for k, p in zip(list(missing_param_dict.keys()), params)
            }
            param_lookup_dict[i] = params

            # Create missing data as dataset
            dataset_object = MissDataset(
                data=original_data_copy,
                target_col=target_col,
                n_folds=5,
                **params,
            )
            
            print("This is the dataset name")
            print(f"{dataset_name}")
            print("This is the dataset.data")
            print(dataset_object.data)
            print("Number of missing values in the dataset")
            print(dataset_object.data.isnull().sum())

            ############################################################################################################
            ############################################################################################################
            #Running pipelines 

            # Run Baseline Pipeline(FOCUS ON RIGHT NOW)
            print("Running Baseline Pipeline")
            baseline_pipeline_experiment = BaselinePipeline(dataset_object, dataset_name, params['missing_mechanism'])
            baseline_metrics_df, baseline_errors_df, baseline_weights_dfs, baseline_preds_df, baseline_distances_df, baseline_imputation_eval_df = baseline_pipeline_experiment.run()
            

            # Run Feature Selection Pipeline
            print("Running Feature Selection Pipeline")
            feature_selection_pipeline = FeatureSelectionPipeline(dataset_object, dataset_name, params['missing_mechanism'])
            fs_metrics_df, fs_errors_df, fs_weights_dfs, fs_preds_df, fs_distances_df, fs_imputation_eval_df = feature_selection_pipeline.run()

            # # Run Missingness Independent Feature Selection Pipeline
            # print("Running MIFS Pipeline")
            # mifs_pipeline = MIFSPipeline(dataset, dataset_name, params['missing_mechanism'])
            # mifs_metrics_df, mifs_errors_df, mifs_weights_dfs, mifs_preds_df, mifs_distances_df, mifs_imputation_eval_df = mifs_pipeline.run()




            ############################################################################################################
            ############################################################################################################










            # # Impute missing data iteratively
            # imputer = Imputers(dataset.data)
            # imputers_dict = {
            #     'MICE': imputer.MICE,
            #     'MissForest': imputer.MissForest,
            #     'XGBoostImputer': imputer.XGBoostImputer,
            #     'BayesianRidge': imputer.BayesianRidge
            # }

            # for name, method in imputers_dict.items():
            #     print(f"Imputing with {name}")
            #     model = method()
            #     imputed_data = model.transform(dataset.data)
            #     # Save imputer model
            #     model_filename = os.path.join(imputer_models_dir, f"{name}_imputer_model_{i}.pkl")
            #     save_model(model, model_filename)

            #     # Perform evaluation on imputed_data...

            # pbar.update(1)




   







CURRENT_SUPPORTED_DATALOADERS = {
    'eeg_eye_state': DataLoadersEnum.prepare_eeg_eye_data
    # 'Cleveland Heart Disease': DataLoadersEnum.prepare_cleveland_heart_data
    # 'diabetic_retinopathy': DataLoadersEnum.prepare_diabetic_retinopathy_dataset,
    # 'wdbc': DataLoadersEnum.prepare_wdbc_data
   
}

def run(custom_experiment_data_object, task_type='classification'):
    MCAR_PARAM_DICT = {
        # 'p_miss': [x/10 for x in range(3,9)], 
        'p_miss': [0.1, 0.2, 0.3, 0.4, 0.5],
        'missing_mechanism': ["MCAR"],
        'opt': [None],
        'p_obs': [None],
        'q': [None],
    }

    MAR_PARAM_DICT = {
        # 'p_miss': [x/10 for x in range(3,9)], 
        'p_miss': [0.1],
        'missing_mechanism': ["MAR"],
        'opt': [None],
        'p_obs': [0.3],
        'q': [None],
    }

    MNAR_PARAM_DICT = {
        'p_miss': [0.1, 0.2, 0.3, 0.4, 0.5],
        'missing_mechanism': ["MNAR"],
        'opt': ['logistic'],
        'p_obs': [0.3],
        'q': [None],
    }
    # , MAR_PARAM_DICT, MNAR_PARAM_DICT
    for d in [MAR_PARAM_DICT]:
        run_custom_experiments(
            original_data=custom_experiment_data_object.data,
            dataset_name=custom_experiment_data_object.dataset_name,
            missing_param_dict=d,
            target_col=custom_experiment_data_object.target_col,
            task_type=task_type
        )



def main():
    for i in range(1, 3):
        print("This is trial number:", i)
        for dataset_name, data_preparation_function_object in CURRENT_SUPPORTED_DATALOADERS.items():
            print(f"This is the current dataset: {dataset_name}")
            run(data_preparation_function_object(), task_type='classification')

if __name__ == '__main__':
    main()

