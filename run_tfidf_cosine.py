from process import preprocess, readdata
from embed import tfidf_cosine

vectorizer = tfidf_cosine.vectorizer

if __name__ == "__main__":
    print('--- reading data ---')
    data = readdata.read()
    X_train, y_train, X_test, y_test = preprocess.split_train_test(data)
    corpus = list(X_train['question1']) + list(X_train['question2'])

    print('--- vectorizing ---')
    # Vectorize and train on corpus
    vectorizer.fit_transform(corpus)
    
    print('--- testing model ---')
    # Predict similarities on test set
    y_pred = tfidf_cosine.predict_cosine(X_test)

    print('Accuracy: ', tfidf_cosine.accuracy_score(y_test, y_pred))
    print('Precision: ', tfidf_cosine.precision_score(y_test, y_pred))
    print('Recall: ', tfidf_cosine.recall_score(y_test, y_pred))
    print('F1: ', tfidf_cosine.f1_score(y_test, y_pred))