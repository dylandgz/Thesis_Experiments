import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

class CompleteCaseImputer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        # No fitting necessary for complete case analysis
        return self
    
    def transform(self, X, y=None):
        # Drop rows with any missing values
        return X.dropna(axis=0, how='any')

