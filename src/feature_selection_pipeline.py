# # import os
# # import pandas as pd
# # import numpy as np
# # from sklearn.feature_selection import mutual_info_classif, SelectKBest, chi2, RFE
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.linear_model import LogisticRegression
# # from sklearn.svm import SVC
# # import xgboost as xgb
# # from sklearn.impute import KNNImputer
# # from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error, roc_auc_score, precision_score, recall_score, confusion_matrix
# # from deap import base, creator, tools, algorithms

# # class FeatureSelectionPipeline:
# #     def __init__(self, dataset, dataset_name, missing_mechanism):
# #         self.dataset = dataset
# #         self.dataset_name = dataset_name
# #         self.missing_mechanism = missing_mechanism
# #         self.selected_features = {}
# #         self.imputations = {}
# #         self.imputation_results = []
# #         self.classification_results = []
# #         self.feature_selection_metrics = []
# #         self.imputation_methods = {
# #             'KNN': KNNImputer(),
# #             'RandomForest': RandomForestClassifier()
# #         }
# #         self.classifiers = {
# #             'SVM': SVC(),
# #             'RandomForest': RandomForestClassifier(),
# #             'XGBoost': xgb.XGBClassifier()
# #         }
# #         self.results_dir = os.path.join('results', dataset_name, missing_mechanism)
# #         os.makedirs(self.results_dir, exist_ok=True)

# #     def run(self):
# #         self.perform_feature_selection()
# #         self.perform_imputations()
# #         self.evaluate_imputations()
# #         self.run_classifications()
# #         self.save_results()



# #     def perform_feature_selection(self):
# #         X_train, X_test, y_train, y_test = self.dataset  # assuming dataset is split

# #         # Filter Methods
# #         self.selected_features['Information Gain'] = self.select_features_filter(X_train, y_train, mutual_info_classif)
# #         self.selected_features['Chi-Squared'] = self.select_features_filter(X_train, y_train, chi2)

# #         # Wrapper Methods
# #         self.selected_features['Genetic Algorithm'] = self.select_features_genetic_algorithm(X_train, y_train)
# #         self.selected_features['RFE'] = self.select_features_rfe(X_train, y_train)

# #         # Save feature selection metrics
# #         for method, features in self.selected_features.items():
# #             self.feature_selection_metrics.append({
# #                 'Method': method,
# #                 'Selected Features': features
# #             })

# #     def select_features_filter(self, X, y, method):
# #         selector = SelectKBest(method, k='all')
# #         selector.fit(X, y)
# #         return selector.get_support(indices=True)

# #     def select_features_genetic_algorithm(self, X, y):
# #         # Genetic Algorithm for feature selection
# #         def eval_genome(individual):
# #             features = [index for index, value in enumerate(individual) if value == 1]
# #             if len(features) == 0:
# #                 return 0,
# #             X_selected = X[:, features]
# #             clf = LogisticRegression()
# #             clf.fit(X_selected, y)
# #             return clf.score(X_selected, y),

# #         creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# #         creator.create("Individual", list, fitness=creator.FitnessMax)

# #         toolbox = base.Toolbox()
# #         toolbox.register("attr_bool", np.random.randint, 0, 2)
# #         toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=X.shape[1])
# #         toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# #         toolbox.register("evaluate", eval_genome)
# #         toolbox.register("mate", tools.cxTwoPoint)
# #         toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
# #         toolbox.register("select", tools.selTournament, tournsize=3)

# #         pop = toolbox.population(n=50)
# #         algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, verbose=False)

# #         best_individual = tools.selBest(pop, 1)[0]
# #         return [index for index, value in enumerate(best_individual) if value == 1]

# #     def select_features_rfe(self, X, y):
# #         selector = RFE(RandomForestClassifier(), n_features_to_select=10)
# #         selector.fit(X, y)
# #         return selector.get_support(indices=True)


# #     def perform_imputations(self):
# #         X_train, X_test, y_train, y_test = self.dataset  # assuming dataset is split

# #         for method, features in self.selected_features.items():
# #             X_train_selected = X_train[:, features]
# #             X_test_selected = X_test[:, features]

# #             # KNN Imputation
# #             knn_imputer = self.imputation_methods['KNN']
# #             X_train_knn = knn_imputer.fit_transform(X_train_selected)
# #             self.imputations[f'KNN_{method}'] = (X_train_knn, X_test_selected)

# #             # RandomForest Imputation (simple complete case analysis)
# #             rf_classifier = self.imputation_methods['RandomForest']
# #             rf_classifier.fit(X_train_selected, y_train)
# #             X_train_rf = rf_classifier.apply(X_train_selected)
# #             self.imputations[f'RandomForest_{method}'] = (X_train_rf, X_test_selected)


# #     def evaluate_imputations(self):
# #         X_train, X_test, y_train, y_test = self.dataset

# #         for method, (X_train_imp, X_test_imp) in self.imputations.items():
# #             rmse = mean_squared_error(y_test, X_test_imp, squared=False)
# #             mae = mean_absolute_error(y_test, X_test_imp)
# #             accuracy = accuracy_score(y_test, X_test_imp)
# #             auc = roc_auc_score(y_test, X_test_imp)
# #             self.imputation_results.append({
# #                 'Method': method,
# #                 'RMSE': rmse,
# #                 'MAE': mae,
# #                 'Accuracy': accuracy,
# #                 'AUC': auc
# #             })



# #     def run_classifications(self):
# #         X_train, X_test, y_train, y_test = self.dataset

# #         for method, (X_train_imp, X_test_imp) in self.imputations.items():
# #             for clf_name, clf in self.classifiers.items():
# #                 clf.fit(X_train_imp, y_train)
# #                 y_pred = clf.predict(X_test_imp)
# #                 accuracy = accuracy_score(y_test, y_pred)
# #                 precision = precision_score(y_test, y_pred, average='binary')
# #                 recall = recall_score(y_test, y_pred, average='binary')
# #                 tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
# #                 specificity = tn / (tn + fp)
# #                 self.classification_results.append({
# #                     'Imputation': method,
# #                     'Classifier': clf_name,
# #                     'Accuracy': accuracy,
# #                     'Precision': precision,
# #                     'Recall': recall,
# #                     'Specificity': specificity
# #                 })



# #     def save_results(self):
# #         imputation_df = pd.DataFrame(self.imputation_results)
# #         classification_df = pd.DataFrame(self.classification_results)
# #         feature_selection_df = pd.DataFrame(self.feature_selection_metrics)

# #         imputation_df.to_csv(os.path.join(self.results_dir, 'imputation_evaluation.csv'), index=False)
# #         classification_df.to_csv(os.path.join(self.results_dir, 'classification_evaluation.csv'), index=False)
# #         feature_selection_df.to_csv(os.path.join(self.results_dir, 'feature_selection.csv'), index=False)




















# import os
# import pandas as pd
# import numpy as np
# from sklearn.feature_selection import mutual_info_classif, SelectKBest, chi2, RFE
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# import xgboost as xgb
# from sklearn.impute import KNNImputer
# from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error, roc_auc_score, precision_score, recall_score, confusion_matrix
# import lightgbm as lgb
# from deap import base, creator, tools, algorithms

# class FeatureSelectionPipeline:
#     def __init__(self, dataset, dataset_name, missing_mechanism):
#         self.dataset = dataset
#         self.dataset_name = dataset_name
#         self.missing_mechanism = missing_mechanism
#         self.selected_features = {}
#         self.imputations = {}
#         self.imputation_results = []
#         self.classification_results = []
#         self.feature_selection_metrics = []
#         self.imputation_methods = {
#             'KNN': KNNImputer(),
#             'RandomForest': RandomForestClassifier()
#         }
#         self.classifiers = {
#             'SVM': SVC(),
#             'RandomForest': RandomForestClassifier(),
#             'XGBoost': xgb.XGBClassifier()
#         }
#         self.results_dir = os.path.join('results', dataset_name, missing_mechanism)
#         os.makedirs(self.results_dir, exist_ok=True)

#     def run(self):
#         self.perform_feature_selection()
#         self.perform_lightgbm_feature_selection()
#         self.perform_imputations()
#         self.evaluate_imputations()
#         self.run_classifications()
#         self.save_results()

#     def perform_feature_selection(self):
#         X_train, X_test, y_train, y_test = self.dataset  # assuming dataset is split

#         # Filter Methods
#         self.selected_features['Information Gain'] = self.select_features_filter(X_train, y_train, mutual_info_classif)
#         self.selected_features['Chi-Squared'] = self.select_features_filter(X_train, y_train, chi2)

#         # Wrapper Methods
#         self.selected_features['Genetic Algorithm'] = self.select_features_genetic_algorithm(X_train, y_train)
#         self.selected_features['RFE'] = self.select_features_rfe(X_train, y_train)

#         # Save feature selection metrics
#         for method, features in self.selected_features.items():
#             self.feature_selection_metrics.append({
#                 'Method': method,
#                 'Selected Features': features
#             })

#     def select_features_filter(self, X, y, method):
#         selector = SelectKBest(method, k='all')
#         selector.fit(X, y)
#         return selector.get_support(indices=True)

#     def select_features_genetic_algorithm(self, X, y):
#         # Genetic Algorithm for feature selection
#         def eval_genome(individual):
#             features = [index for index, value in enumerate(individual) if value == 1]
#             if len(features) == 0:
#                 return 0,
#             X_selected = X[:, features]
#             clf = LogisticRegression()
#             clf.fit(X_selected, y)
#             return clf.score(X_selected, y),

#         creator.create("FitnessMax", base.Fitness, weights=(1.0,))
#         creator.create("Individual", list, fitness=creator.FitnessMax)

#         toolbox = base.Toolbox()
#         toolbox.register("attr_bool", np.random.randint, 0, 2)
#         toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=X.shape[1])
#         toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#         toolbox.register("evaluate", eval_genome)
#         toolbox.register("mate", tools.cxTwoPoint)
#         toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
#         toolbox.register("select", tools.selTournament, tournsize=3)

#         pop = toolbox.population(n=50)
#         algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, verbose=False)

#         best_individual = tools.selBest(pop, 1)[0]
#         return [index for index, value in enumerate(best_individual) if value == 1]

#     def select_features_rfe(self, X, y):
#         selector = RFE(RandomForestClassifier(), n_features_to_select=10)
#         selector.fit(X, y)
#         return selector.get_support(indices=True)

#     def perform_lightgbm_feature_selection(self):
#         X_train, X_test, y_train, y_test = self.dataset  # assuming dataset is split

#         lgb_train = lgb.Dataset(X_train, y_train)
#         params = {
#             'objective': 'binary',
#             'metric': 'binary_logloss',
#             'verbosity': -1
#         }
#         gbm = lgb.train(params, lgb_train, num_boost_round=100)
#         importance = gbm.feature_importance()
#         selected_features = np.argsort(importance)[::-1][:10]  # select top 10 features
#         self.selected_features['LightGBM'] = selected_features

#         # Save feature selection metrics
#         self.feature_selection_metrics.append({
#             'Method': 'LightGBM',
#             'Selected Features': selected_features
#         })

#     def perform_imputations(self):
#         X_train, X_test, y_train, y_test = self.dataset  # assuming dataset is split

#         for method, features in self.selected_features.items():
#             X_train_selected = X_train[:, features]
#             X_test_selected = X_test[:, features]

#             # KNN Imputation
#             knn_imputer = self.imputation_methods['KNN']
#             X_train_knn = knn_imputer.fit_transform(X_train_selected)
#             self.imputations[f'KNN_{method}'] = (X_train_knn, X_test_selected)

#             # RandomForest Imputation (simple complete case analysis)
#             rf_classifier = self.imputation_methods['RandomForest']
#             rf_classifier.fit(X_train_selected, y_train)
#             X_train_rf = rf_classifier.apply(X_train_selected)
#             self.imputations[f'RandomForest_{method}'] = (X_train_rf, X_test_selected)

#     def evaluate_imputations(self):
#         X_train, X_test, y_train, y_test = self.dataset

#         for method, (X_train_imp, X_test_imp) in self.imputations.items():
#             rmse = mean_squared_error(y_test, X_test_imp, squared=False)
#             mae = mean_absolute_error(y_test, X_test_imp)
#             accuracy = accuracy_score(y_test, X_test_imp)
#             auc = roc_auc_score(y_test, X_test_imp)
#             self.imputation_results.append({
#                 'Method': method,
#                 'RMSE': rmse,
#                 'MAE': mae,
#                 'Accuracy': accuracy,
#                 'AUC': auc
#             })

#     def run_classifications(self):
#         X_train, X_test, y_train, y_test = self.dataset

#         for method, (X_train_imp, X_test_imp) in self.imputations.items():
#             for clf_name, clf in self.classifiers.items():
#                 clf.fit(X_train_imp, y_train)
#                 y_pred = clf.predict(X_test_imp)
#                 accuracy = accuracy_score(y_test, y_pred)
#                 precision = precision_score(y_test, y_pred, average='binary')
#                 recall = recall_score(y_test, y_pred, average='binary')
#                 tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
#                 specificity = tn / (tn + fp)
#                 self.classification_results.append({
#                     'Imputation': method,
#                     'Classifier': clf_name,
#                     'Accuracy': accuracy,
#                     'Precision': precision,
#                     'Recall': recall,
#                     'Specificity': specificity
#                 })

#     def save_results(self):
#         imputation_df = pd.DataFrame(self.imputation_results)
#         classification_df = pd.DataFrame(self.classification_results)
#         feature_selection_df = pd.DataFrame(self.feature_selection_metrics)

#         imputation_df.to_csv(os.path.join(self.results_dir, 'imputation_evaluation.csv'), index=False)
#         classification_df.to_csv(os.path.join(self.results_dir, 'classification_evaluation.csv'), index=False)
#         feature_selection_df.to_csv(os.path.join(self.results_dir, 'feature_selection.csv'), index=False)

