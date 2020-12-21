import pandas as pd
import os.path as path
from sklearn.ensemble import RandomForestRegressor
import pickle

class RF_Regressor:
    def __init__(self, n_estimators=500, max_depth=50, min_samples_leaf=50, **rf_params):
        self.curr_path = path.abspath("__file__") # Full path to current class-definition script
        self.root_path = path.dirname(path.dirname(path.dirname(self.curr_path)))
        
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            min_samples_leaf=min_samples_leaf, 
            **rf_params
        )

    
    def train(self, X, y):
        """Fits the regressor model

        Args:
            X (pd.DataFrame): df with features
            y (pd.Series): contains true training return rates
        """
        self.model.fit(X, y)


    def load_model(self, filename):
        """Loads a pre-trained model (pkl or other data type)

        Args:
            filename (str): location of model to load
        """
        filename = path.join(self.root_path, f'models/{filename}.pkl')
        self.model = pickle.load(open(filename, "rb"))
        print('Successfully loaded model from '+filename)
    

    def predict(self, X):
        """Regression model predicts on new dataset

        Args:
            X (pd.DataFrame): df with features to predict on
        """
        return self.model.predict(X)

    def _rates_to_outlier_bool(self, rates, mkcnames, threshold):
        """Convert raw float rates to outlier boolean values
        Args:
            rates (pd.Series): raw float product return rates
            mkcnames (pd.Series): contains market category names for each rate in pred_rates
            threshold (float): top threshold % of return rates are denoted outliers
        """
        # get cutoff for each market category name
        mkcname_cutoffs = rates.groupby(mkcnames).quantile(1-threshold)
        mkcname_cutoffs_dict = mkcname_cutoffs.to_dict()
        
        # return whether rates are above cutoffs
        return (rates >= mkcnames.map(mkcname_cutoffs_dict)).astype(int)


    def predict_outliers(self, X, mkcnames, threshold=0.1):
        """Predict outliers based on market categories

        Args:
            X (pd.DataFrame): df with features to predict on
            mkcnames (array-like): contains market category names for each row in X
            threshold (float): top threshold % of return rates are denoted outliers
        """
        pred_rates = self.model.predict(X)
        pred_rate_series = pd.Series(pred_rates, index=X.index)
        mkcname_series = pd.Series(mkcnames, index=X.index)
        return self._rates_to_outlier_bool(pred_rate_series, mkcname_series, threshold)