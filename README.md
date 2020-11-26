# EntHire Sentiment Analysis Project
## Project Objective:
To create an API endpoint that can accept a text and return associated sentiment with it.
## The data:
The data is in "airline_sentiment_analysis.csv". It has 2 columns:

1. text : has 11541 rows of tweets in text format.
2. airline_sentiment : has 11541 rows of the infered sentiment.

## Languages and Libraries Used:

1.  Python - main language
2.  Pandas - for data analysis
3.  Numpy - for numerical analysis
4.  Seaborn - for graphical analysis
5.  Scikit-Learn Library - for ML algorithms
6.  Pyramid - for web dev framework
7.  Pyramid_openapi3 - for OpenAPI (the successor of Swagger)
8.  Pyramid_swagger - for Swagger documentation

## File Structure:

1. api_docs/swagger.yaml : this file contains the YAML code for the SWAGGER documentation.
2. Exploratory_data_analysis.ipynb : This Jupyter Notebook is where all the data analysis has been done.
3. model.py : contains the MAIN NLPModel() class that has various methods to train and predict data. 
4. build_model.py : is the helper class to model.py that helps build and train the model
5. inference.py : is the class that helps in inferring the results of the data in api.py
6. api.py : is the main API that takes in input at http://localhost:6543/predict and returns the predictions. This also creates the model if the pickled data isn.t present.

## Project Overview:
For this project, we move forward in 4 steps:

1. Exploring the data and cleaning it.
2. Finding a suitable Machine Learning Model and training it.
3. Making the API using Pyramid web framework, that returns a JSON item containing the predictions.
4. Implementing the Swagger Documentation for the same API.

## Step 1 : Exploratory Data Analysis
Has been done completely in the exploratory_data_analysis.ipynb

1. In this I saw that the number of datapoints were 11541, but the index was askew. So I reset the index according to the size of the data.
2. I found that there are 9178 negative tweets in comparison to 2363 positive tweets. 
3. I found that there were 6 major airlines tagged in all of the tweets. I found out that dividing the data according to the airlines would be futile as the amount of data would be very less and lead to low accuracy.
4. I implemented pd.get_dummies() to get numerical values for the sentiment.
5. I found out the major reasons for the negative and positive sentiments using wordCloud.For Negative Reasons:

![image](https://user-images.githubusercontent.com/68659873/100383670-2d3fd700-3044-11eb-956f-5965c64e5993.png)

For Positive Reasons:

![image](https://user-images.githubusercontent.com/68659873/100383676-3466e500-3044-11eb-8665-7bc680b17f75.png)



6. I found out that the majority of negative tweets were against United Airlines and the least against Virgin America.

![image](https://user-images.githubusercontent.com/68659873/100383631-0f727200-3044-11eb-9aa5-9d6ccfdede22.png)

## Step 2 : Finding the best ML model:
For this project of classification, we can try out the following ML models from the scikit-learn library and compare them by the following metrics:

1. The confusion matrices : has the data about False and True Positives and Negatives.
2. The classification reports : has the accuracy, the macro avgs.
3. The accuracy scores : is the overall accuracy of the model.
4. The f1 scores : is the weighted average of the precision and recall.

![image](https://user-images.githubusercontent.com/68659873/100383772-75f79000-3044-11eb-8e12-66a9e6a95b4a.png)

We find that the best classifier is RandomForestClassifier(). Now we fit and train the model using the pipeline from sklearn library.

```python
from sklearn.pipeline import Pipeline
pipeline=Pipeline([
    ('bow',CountVectorizer(analyzer=text_clean)),
    ('tfidf',TfidfTransformer()),
    ('classifier',classifier)
    ])
```
## Step 3 : Making the API using pyramid
The API is in api.py and is a simple API with 2 endpoints:

1. /predict: this is the MAIN endpoint that is give a POST input and it returns the predictions
2. /home : this is the starting page of the API and has different instructions
3. /docs/ : this is the endpoint that leads to the SWAGGER documentation of the API

## Step 4 : Implementing the Swagger Documentation:
The Swagger documentation is done in http://localhost:6543/docs/

![image](https://user-images.githubusercontent.com/68659873/100384191-93792980-3045-11eb-99b9-b6bd89bd1a8e.png)



