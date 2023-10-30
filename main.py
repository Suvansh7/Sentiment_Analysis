import pandas as pd 
from googletrans import Translator 
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
from icecream import ic
from data_cleaning import clean
from data_preprocessing import process


translator = Translator()
sia=SentimentIntensityAnalyzer()
data = pd.read_csv('C:\\Users\\ishan\\Desktop\\sentiment analysis\\Sentiment_Analysis\\tweets.csv')

ic(data.info())
ic(data)
ic(type(data))
ic(data.shape)
ic(data.describe())

cleaned_data = clean(data)
ic(cleaned_data)
preprocessed_data = process(data,translator)
ic(preprocessed_data)
