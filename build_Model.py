# Some necessary imports
from model import NLPModel
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import joblib


def take_input():
        """
        Reads and returns data
        """
        data = pd.read_csv('airline_sentiment_analysis.csv', index_col=0) # takes the first column as the index
        data.index = range(1, 11542) # resets the index according to the size of the dataframe
        data = pd.get_dummies(data, columns=['airline_sentiment']) # Creates dummy variables for the text : 0-> Negative ; and 1->Positive
        data.drop('airline_sentiment_negative', axis=1, inplace=True) # drops the extra columns
        data.columns = ['text', 'sentiment']
        return data

class build():

    def __init__(self):
        self.data=take_input()

    def data_split(self,data,num_split):
        x_train, x_test, y_train, y_test = train_test_split(self.data['text'], self.data['sentiment'], test_size=num_split, random_state=42)
        return x_train, y_train

    def build_model(self):
        print("Starting to Train Model....")
        model = NLPModel()
        x_train, y_train = self.data_split(self.data,num_split=0.3)
        model.train(x_train, y_train)
        print('Model Training Complete....')
        model.save_as_pickle()
        print('Model saved as pickle....')
    
    def load_pipeline(self):
        """
        Loads the pickled pipeline.
        If the pickle isn't there, it builds the model and then proceeds.
        """
        file = Path('NLPModel_joblib.pkl')
        path = 'NLPModel_joblib.pkl'
        if not file.is_file():
            print("There is no pickled model....")
            self.build_model()
        with open(path, 'rb') as file:
            pipeline = joblib.load(path)
            print("Pipeline loaded from {}".format(path))
        return pipeline


if __name__ == "__main__":
    new_model=build()
    new_model.build_model()
