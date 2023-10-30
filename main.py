import pandas as pd 
from googletrans import Translator 
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
from icecream import ic
from data_cleaning import clean


translator = Translator()
sia=SentimentIntensityAnalyzer()
data = pd.read_csv('E:\\Projects\\Sentiment_Analysis\\tweets.csv')

ic(data.info())
ic(data)
ic(type(data))
ic(data.shape)
ic(data.describe())

clean(data)