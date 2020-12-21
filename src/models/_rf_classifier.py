import os.path as path
from sklearn.ensemble import RandomForestClassifier
import pickle

class RF_Classifier:
    def __init__(self, n_estimators=500, max_depth=50, min_samples_leaf=50, **rf_params):
        self.curr_path = path.abspath("__file__") # Full path to current class-definition script
        self.root_path = path.dirname(path.dirname(path.dirname(self.curr_path)))
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            min_samples_leaf=min_samples_leaf, 
            **rf_params
        )

    
    def train(self, X, y):
        """Fits the classifier model

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
    

    def predict_proba(self, X):
        """Classifier model predicts probabilities on new dataset

        Args:
            X (pd.DataFrame): df with features to predict on
        """
        return self.model.predict_proba(X)


    def predict(self, X):
        """Classifier model predicts on new dataset

        Args:
            X (pd.DataFrame): df with features to predict on
        """
        return self.model.predict(X)