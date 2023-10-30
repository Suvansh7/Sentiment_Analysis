from icecream import ic
def Vader_Algorithm(sia, raw_data):
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
    import matplotlib.pyplot as plt
    import seaborn as sns

    raw_data['scores'] = raw_data['review'].apply(lambda reviews: sia.polarity_scores(reviews))
    raw_data['compound'] = raw_data['scores'].apply(lambda score_dict: score_dict["compound"])
    raw_data['Comp_score'] = raw_data['compound'].apply(lambda score: "pos" if score>=0 else "neg")


    cm = confusion_matrix(raw_data['label'], raw_data['Comp_score']) 
    ic(cm)

    classification_report(raw_data['label'], raw_data['Comp_score'])
    accuracy_score(raw_data['label'], raw_data['Comp_score']) 

    plt.title('Heatmap of Confusion Matrix', fontsize = 15)
    sns.heatmap(cm, annot = True)
    plt.show()

def Support_vector_classifier(raw_data):
    from sklearn import svm
    from sklearn.feature_extraction.text import TfidfVectorizer
# Create feature vectors
    vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)
    train_vectors = vectorizer.fit_transform(raw_data['review'])
    test_vectors = vectorizer.transform(raw_data['review'])   