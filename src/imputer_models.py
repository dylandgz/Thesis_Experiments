import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from missingpy import MissForest
import xgboost as xgb
from sklearn.linear_model import BayesianRidge

class Imputers:
    def __init__(self, data):
        self.data = data
        self.mask = self.create_mask()
        
    def create_mask(self):
        return self.data.isnull()

    def MICE(self):
        mice_imputer = IterativeImputer(random_state=0)
        mice_imputer.fit(self.data)
        return mice_imputer

    def MissForest(self):
        miss_forest_imputer = MissForest(random_state=0)
        miss_forest_imputer.fit(self.data)
        return miss_forest_imputer

    def XGBoostImputer(self):
        xgboost_imputer = IterativeImputer(estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=0), random_state=0)
        xgboost_imputer.fit(self.data)
        return xgboost_imputer

    def BayesianRidge(self):
        bayesian_ridge_imputer = IterativeImputer(estimator=BayesianRidge(), random_state=0)
        bayesian_ridge_imputer.fit(self.data)
        return bayesian_ridge_imputer
