import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer
from readdata import read
import preprocess
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity
from progress.bar import Bar


stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

'''remove punctuation, lowercase, stem'''
def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer=normalize)

def cosine_sim(text1, text2):
    tfidf = vectorizer.transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]

def predict_cosine(X_test):
    y_pred = []
    with Bar('scoring', max=len(X_test)) as bar:
        for index, row in X_test.iterrows():
            text1 = row['question1']
            text2 = row['question2']
            
            score = cosine_sim(text1, text2)

            # Threshold
            if score > 0.8:
                y_pred.append(1)
            else:
                y_pred.append(0)
            bar.next()
    return y_pred

if __name__ == "__main__":
    print('--- reading data ---')
    data = read()
    X_train, y_train, X_test, y_test = preprocess.split_train_test(data)
    corpus = list(X_train['question1']) + list(X_train['question2'])

    print('--- vectorizing ---')
    # Vectorize and train on corpus
    vectorizer.fit_transform(corpus)
    
    print('--- testing model ---')
    # Predict similarities on test set
    y_pred = predict_cosine(X_test)

    # Compare predictions actual correct answers of test set
    print('Accuracy: ', accuracy_score(y_test, y_pred))
    print('Precision: ', precision_score(y_test, y_pred))
    print('Recall: ', recall_score(y_test, y_pred))
   
