import pandas as pd 
from googletrans import Translator 
from icecream import ic
from data_cleaning import clean
from sklearn.model_selection import train_test_split
from data_preprocessing import process
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from algo import vader_algorithm,support_vector_classifier,naive_bayes

translator = Translator()
data = pd.read_csv('E:\\Projects\\Sentiment_Analysis\\tweets.csv')

ic(data.info())
ic(data)
ic(type(data))
ic(data.shape)
ic(data.describe())

x_train, x_test, y_train, y_test = train_test_split(data['Tweet'], data['Date'], test_size=0.2, random_state=5)
data = clean(data)
data = process(data)

data['scores'] = data['new'].apply(lambda review: SentimentIntensityAnalyzer.polarity_scores(review)) # Calculating polarity
data['compound'] = data['scores'].apply(lambda score_dict: score_dict["compound"])
data['Comp_score'] = data['compound'].apply(lambda score: "pos" if score>=0 else "neg")

vader_algorithm(data)
support_vector_classifier(data)
naive_bayes(data)
