from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import pickle
import numpy as np

class DecisionTree(object):
    def __init__(self, 
                max_depth : int = None, 
                min_samples_split : int = None, 
                min_samples_leaf : int = None
    ) -> None:
        """The model module that orchestrates everything regarding the model

        Args:
            max_depth int: The maximum depth of the tree
            min_samples_split int: The minimum number of samples required to split and internal node
            min_samples_leaf int: Where the trained model will be stored
        """
        self.model_parameters = {
            'max_depth' : max_depth,
            'min_samples_split' : min_samples_split,
            'min_samples_leaf' : min_samples_leaf,
            'criterion' : 'entropy',
            'random_state' : 0
        }

        self.model = None

    def get_model(self) -> DecisionTreeClassifier:
        return self.model
    
    def fit_model(self,
                    X : pd.DataFrame,
                    y : pd.Series
        ) -> DecisionTreeClassifier:
        """Fit the model with the given hyperparamters

            Args:
                X pd.DataFrame: The attributes of the data
                y pd.DataFrame: The label of the data
            Return:
                DecisionTreeClassifier: The trained model
        """
        self.model = DecisionTreeClassifier(**self.model_parameters)
        return self.model.fit(X, y)

    def predict(self, 
                X : pd.DataFrame
    ) -> np.array:
        """Predict the classes of the given X
        
        Args:
            X pd.DataFrame: The attributes of the data that will be predicted

        Return:
            np.array: The classes of the given X
        """
        return self.model.predict(X)

    def save_model(self,
                    path : str
        ) -> str:
        """Save the model

        Args:
            path str: Where the trained model will be stored
        
        Return:
            str: Where the trained model is saved
        """
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        return path
    
    def load_model(self,
                    path : str
        ) -> None:
        """Load the model to self.model

        Args:
            path str: Where did the model be saved
        """
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        return