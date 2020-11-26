# Some necessary imports
from model import NLPModel
import pandas as pd
from sklearn.model_selection import train_test_split


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


def build_model():
    print("Starting to Train Model....")
    model = NLPModel()
    data = take_input()
    x_train, x_test, y_train, y_test = train_test_split(data['text'], data['sentiment'], test_size=0.3, random_state=42)
    model.train(x_train, y_train)
    print('Model Training Complete....')
    model.save_as_pickle()
    print('Model saved as pickle....')


if __name__ == "__main__":
    build_model()
