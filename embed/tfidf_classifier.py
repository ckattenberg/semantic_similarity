import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity
from progress.bar import Bar
from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from math import floor
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

classifier = naive_bayes.MultinomialNB()
# classifier = RandomForestClassifier(n_estimators=1000, random_state = 0)
# classifier = SVC(gamma=2, C=1)
# classifier = MLPClassifier(alpha=1, max_iter=1000)

def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

'''remove punctuation, lowercase, stem'''
def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer= normalize, max_features=10000, ngram_range=(1,2))

def concat_df(data):
    combined = []
    with Bar('concatting', max=len(data)) as bar:
        for index, row in data.iterrows():
            text1 = row['question1']
            text2 = row['question2']
            combined.append(text1+text2)
            bar.next()
    return combined

def split_train_test_concat(X, Y):
    partition = floor(len(Y)*0.7)
    X_train = X[:partition]
    X_test = X[partition:]
    y_train = Y[:partition]
    y_test = Y[partition:]
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    pass
   
