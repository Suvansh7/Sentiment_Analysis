from icecream import ic
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

def vader_algorithm(raw_data):
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import matplotlib.pyplot as plt
    import seaborn as sns

    raw_data['scores'] = raw_data['review'].apply(lambda reviews: SentimentIntensityAnalyzer.polarity_scores(reviews))
    raw_data['compound'] = raw_data['scores'].apply(lambda score_dict: score_dict["compound"])
    raw_data['Comp_score'] = raw_data['compound'].apply(lambda score: "pos" if score>=0 else "neg")


    cm = confusion_matrix(raw_data['label'], raw_data['Comp_score']) 
    ic(cm)

    classification_report(raw_data['label'], raw_data['Comp_score'])
    accuracy_score(raw_data['label'], raw_data['Comp_score']) 

    plt.title('Heatmap of Confusion Matrix', fontsize = 15)
    sns.heatmap(cm, annot = True)
    plt.show()

def support_vector_classifier(raw_data):
    from sklearn import svm
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
    from sklearn.feature_extraction.text import TfidfVectorizer
    import matplotlib.pyplot as plt
    import seaborn as sns


    vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)
    train_vectors = vectorizer.fit_transform(raw_data['review'])
    test_vectors = vectorizer.transform(raw_data['review'])   

    classifier_linear = svm.SVC(kernel='linear')
    classifier_linear.fit(train_vectors, raw_data['label']) 
    prediction_linear = classifier_linear.predict(test_vectors) 

    classification_report(raw_data['label'], prediction_linear, output_dict=True)
    accuracy_score(raw_data['label'], prediction_linear)

    cm = confusion_matrix(raw_data['label'], prediction_linear) 
    ic(cm)

    plt.title('Heatmap of Confusion Matrix', fontsize = 15)
    sns.heatmap(cm, annot = True)
    plt.show()

def naive_bayes(x_train , y_train, x_test , y_test):

    from sklearn.pipeline import Pipeline
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from sklearn.model_selection import train_test_split, GridSearchCV
    
    text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])
    tuned_parameters = {
        'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),
        'clf__alpha': [1, 1e-1, 1e-2]
    }

    score = 'f1_macro'
    ic("# Tuning hyper-parameters for %s" % score)
    ic()
    np.errstate(divide='ignore')
    clf = GridSearchCV(text_clf, tuned_parameters, cv=10, scoring=score)
    clf.fit(x_train, y_train)

    ic("Best parameters set found on development set:")
    ic()
    ic(clf.best_params_)
    ic()
    ic("Grid scores on development set:")
    ic()
    for mean, std, params in zip(clf.cv_results_['mean_test_score'], 
                                clf.cv_results_['std_test_score'], 
                                clf.cv_results_['params']):
        ic("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    ic()

    classification_report(y_test, clf.predict(x_test), digits=4)
    accuracy_score(y_test, clf.predict(x_test))
    cm = confusion_matrix(y_test, clf.predict(x_test)) 
    ic(cm)

    plt.title('Heatmap of Confusion Matrix', fontsize = 15)
    sns.heatmap(cm, annot = True)
    plt.show()
