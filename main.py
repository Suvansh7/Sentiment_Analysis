import pandas as pd 
from googletrans import Translator 
from icecream import ic
from data_cleaning import clean
from sklearn.model_selection import train_test_split
from data_preprocessing import process
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from algo import vader_algorithm,support_vector_classifier,naive_bayes

translator = Translator()
sia = SentimentIntensityAnalyzer()
data = pd.read_csv('C:\\Users\\ishan\\Desktop\\sentiment analysis\\Sentiment_Analysis\\tweets.csv')

ic(data.info())
ic(data)
ic(type(data))
ic(data.shape)
ic(data.describe())

data = clean(data)
data = process(data,translator)
ic(data)
ic(data['new'])
# for i in data['new']:
#     data['scores'] = SentimentIntensityAnalyzer.polarity_scores(text = i)
    

data['scores'] = data['new'].apply(lambda new: sia.polarity_scores(new)) # Calculating polarity
data['compound'] = data['scores'].apply(lambda score_dict: score_dict["compound"])
data['Comp_score'] = data['compound'].apply(lambda score: "pos" if score>=0 else "neg")
x_train, x_test, y_train, y_test = train_test_split(data['new'], data['Comp_score'], test_size=0.2, random_state=5)

v=vader_algorithm(data)
s = support_vector_classifier(data)
n = naive_bayes(x_train,y_train, x_test,  y_test)
ll = [v,s,n]
print(ll)