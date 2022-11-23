import pandas as pd
from typing import Tuple
from sklearn.preprocessing import StandardScaler

class AnnealDataModule(object):
    def __init__(self, 
                train_path : str,
                test_path : str
    ) -> None:
        """The data module that orchestrates everything regarding the anneal data
        
        Args:
            train_path str: The path of the file where train data is stored
            test_path str: The path of the file where test data is stored
        """
        self.train_path = train_path
        self.test_path = test_path

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Loads the files where the data is stored to dataframes 
        """
        pd.DataFrame = pd.read_csv(self.train_path)
        pd.DataFrame = pd.read_csv(self.test_path)
        pass
    
    def preprocess_data(self,
                        train_df : pd.DataFrame,
                        test_df : pd.DataFrame,
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Preprocess the model so that it can learn the data

        Args:
            train_df pd.DataFrame: The dataframe of the file where the train data is stored
            test_df pd.DataFrame: The dataframe of the file where the test data is stored
        
        Return:
            pd.DataFrame: The datafarme of the train data that is preprocessed for the model
            pd.DataFrame: The datafarme of the test data that is preprocessed for the model
        """
        scaler = StandardScaler()
        train_df_scaled = scaler.fit_transform(train_df)
        test_df_scaled = scaler.fit_transform(test_df)
        pd.DataFrame = train_df_scaled
        pd.DataFrame = test_df_scaled
        pass

    def get_dataset(self) -> dict:
        """Return the attributes and the label of the data
        
        Return:
            dict: The dictionary that have attributes X and the label y
        """
        train_df, test_df = self.load_data()
        train_df, test_df = self.preprocess_data(train_df, test_df) 

        X_train = train_df.drop('classes', axis=1)
        y_train = train_df['classes']

        X_test = test_df.drop('classes', axis=1)
        y_test = test_df['classes']

        return {
            'X_train' : X_train,
            'y_train' : y_train,
            'X_test' : X_test,
            'y_test' : y_test
        }