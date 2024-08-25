import faulthandler
import tracemalloc



import os
from data_loaders import MedicalDataset, MissDataset, DataLoadersEnum
from itertools import product
from datetime import datetime
from tqdm import tqdm
from copy import deepcopy
import pickle
import json
import pandas as pd

from baseline_pipeline import BaselinePipeline
from feature_selection_pipeline import FeatureSelectionPipeline


def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)


# Running experiment after setting up the parameters for one MISSING MECHANISM. Each missing mechanism is run separately
def run_custom_experiments(original_data, dataset_name, missing_param_dict, target_col, task_type='classification'):
    name='Experiment_' + str(datetime.now()),
    # Generate the list of all possible combinations of missing parameters
    missing_param_grid = list(product(*tuple(missing_param_dict.values())))
    print(f"\n-----------------------------   Experiment Missing Parameters  -----------------------------\n")
    # Print the total number of combinations
    print(f"Total length of experiment missing parameters: {len(missing_param_grid)}")

    # Print the actual combinations
    print("Experiment missing parameter grid:")

    for params in missing_param_grid:
        print(f"\t{params}")



    param_lookup_dict = {}
    metrics_dfs = []
    imputation_eval_results = []
    miss_type = missing_param_dict['missing_mechanism']
    name = miss_type[0] + '_Experiment_' + str(datetime.now()).replace(':', '-').replace(' ', '_')

    
    
    # with tqdm(total=len(missing_param_grid)) as pbar:
    for i, params in enumerate(missing_param_grid):
        print(f"\n-----------------------------  Starting experiments for {missing_param_dict['missing_mechanism']} {dataset_name}  -----------------------------\n")
        print(f"Starting experiment with params: {params}")
        original_data_copy = deepcopy(original_data)
        params = {k: p for k, p in zip(list(missing_param_dict.keys()), params)}
        param_lookup_dict[i] = params
        


        print("Main.py works till here -------------------")
        dataset_object = MissDataset(
            data=original_data_copy,
            target_col=target_col,
            n_folds=5,
            **params,
        )
            # Stop tracing memory allocations and get the current and peak memory usage
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage: {current / 10**6} MB")
        print(f"Peak memory usage: {peak / 10**6} MB")

        # Stop the tracing
        tracemalloc.stop()

        


        # Running Baseline Pipeline
        print("Running Baseline Pipeline")
        baseline_pipeline_experiment = BaselinePipeline(dataset_object=dataset_object, dataset_name=dataset_name, missing_mechanism=params['missing_mechanism'], name=name)
        baseline_metrics_df, baseline_errors_df, baseline_preds_df, baseline_imputation_eval_df = baseline_pipeline_experiment.run()

        # # Running Feature Selection Pipeline
        # print("Running Feature Selection Pipeline")
        # feature_selection_pipeline = FeatureSelectionPipeline(dataset_object, dataset_name, params['missing_mechanism'])
        # fs_metrics_df, fs_errors_df, fs_weights_dfs, fs_preds_df, fs_distances_df, fs_imputation_eval_df = feature_selection_pipeline.run()


        ###################
        filename = 'combined-missing_param_' + str(i) + '.csv'
        
        errors_filename = os.path.join(baseline_pipeline_experiment.results_dir, 'errors_' + filename)
        
        baseline_errors_df.to_csv(errors_filename)
        
        metrics_dfs.append(baseline_metrics_df)
        metrics_filename = os.path.join(baseline_pipeline_experiment.results_dir, 'prediction_metrics_' + filename)
        baseline_metrics_df.to_csv(metrics_filename)

        imputation_eval_results.append(baseline_imputation_eval_df)
        imputation_eval_filename = os.path.join(baseline_pipeline_experiment.results_dir, 'imputation_eval_' + filename)
        # baseline_imputation_eval_df = baseline_imputation_eval_df.T
        baseline_imputation_eval_df.to_csv(imputation_eval_filename)

        preds_filename = os.path.join(baseline_pipeline_experiment.results_dir, 'predictions_' + filename)
        baseline_preds_df.to_csv(preds_filename)
        ###################

        
        print('Updating progress bar after missing param index ' + str(i))
            # pbar.update(1)

    print("Combining and saving final results")
    final_results = pd.concat(metrics_dfs)
    final_results.to_csv(os.path.join(baseline_pipeline_experiment.base_dir, 'prediction_metrics_final_results.csv'))

    combined_folds_imputation_eval_results_df = pd.concat(imputation_eval_results)
    # combined_folds_imputation_eval_results_df = combined_folds_imputation_eval_results_df.T
    combined_folds_imputation_eval_results_df.to_csv(os.path.join(baseline_pipeline_experiment.base_dir, 'imputation_eval_final_results.csv'))

    param_lookup_dict_json = json.dumps(param_lookup_dict)
    with open(os.path.join(baseline_pipeline_experiment.base_dir, 'params_lookup.json'), 'w') as f:
        f.write(param_lookup_dict_json)


CURRENT_SUPPORTED_DATALOADERS = {
    'eeg_eye_state': DataLoadersEnum.prepare_eeg_eye_data
    # 'Cleveland Heart Disease': DataLoadersEnum.prepare_cleveland_heart_data
    # 'diabetic_retinopathy': DataLoadersEnum.prepare_diabetic_retinopathy_dataset
    # 'wdbc': DataLoadersEnum.prepare_wdbc_data
   
}


# Setting up experiment parameters
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




# Driver Function
def main():
    faulthandler.enable()
    total_trials = 10
    for i in range(0, total_trials):
        for dataset_name, data_preparation_function_object in CURRENT_SUPPORTED_DATALOADERS.items():
            print(f"\nTrial: {i+1}/{total_trials} for Dataset: {dataset_name}")
            run(data_preparation_function_object(), task_type='classification')

if __name__ == '__main__':
    tracemalloc.start()
    main()