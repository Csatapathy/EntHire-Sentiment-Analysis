# Some necessary import statements
import re
import string
import joblib
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline


class NLPModel(object):
    """
    This class is the Main Model and applies the RandomForestClassifier which has ~ 90%  accuracy as outlined in /EDA.ipynb
    """

    def __init__(self):
        """
        This defines the Pipeline that fine tunes the model and then is used to train and fit the data.
        """
        self.pipeline = Pipeline([
            # creates a bag of words after cleaning the data
            ('bag_of_words', CountVectorizer(analyzer=self.text_clean)),
            # Finds out the TF-IDF nature of the bag of words
            ('tfidf', TfidfTransformer()),
            # Is the MAIN classifier that fits and trains the data
            ('classifier', RandomForestClassifier(n_estimators=200))
        ])

    def text_clean(self, text):
        """
            1.removes punctuation
            2.removes stop words
            3.removes the tags, hashtags,emojis and url links
            4.returns list of clean text words
            """
        regrex_pattern = re.compile(pattern="["
                                            u"\U0001F600-\U0001F64F"  # emoticons
                                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                            # flags (iOS)
                                            u"\U0001F1E0-\U0001F1FF"
                                            "]+", flags=re.UNICODE)
        text = regrex_pattern.sub(r'', text)  # Removing the regex
        ll = text.split(" ")
        c = ['@', '#', 'http']
        ll = [ele for ele in ll if all(d not in ele for d in c)]
        m = [c for c in ll if c not in string.punctuation]
        return [word for word in m if word.lower() not in stopwords.words('english')]

    def train(self, x, y):
        """
        Fits the data to the Pipeline
        """
        self.pipeline.fit(x, y)

    def predict(self, test):
        """
        Sends the predictions :
        1. '0' for negative sentiment
        2. '1' for positive sentiment
        """
        y_pred = self.pipeline.predict(test)
        return y_pred

    def save_as_pickle(self):
        """
        Saves the model as a pickle to be loaded later on
        """
        path = 'NLPModel_joblib.pkl'
        with open(path, 'wb') as file:
            joblib.dump(self.pipeline, file)
            print('Pickled the pipeline at {}'.format(path))
