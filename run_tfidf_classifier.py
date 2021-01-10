from process import preprocess, readdata
from embed import tfidf_classifier

vectorizer = tfidf_classifier.vectorizer
classifier = tfidf_classifier.classifier

if __name__ == "__main__":
    result = []
    print('--- reading data ---')
    data = readdata.read()[:10000]
    print(data['is_duplicate'].value_counts())

    print('--- concatting ---')
    combined = tfidf_classifier.concat_df(data)

    print('--- vectorizing ---')
    X = vectorizer.fit_transform(combined)
    
    X_train, X_test, y_train, y_test = tfidf_classifier.split_train_test_concat(X, data['is_duplicate'])

    print('--- training model ---')
    model = classifier.fit(X_train, y_train)

    print('--- testing model ---')
    y_pred = model.predict(X_test)

    # Print scores
    print('Accuracy: ', tfidf_classifier.accuracy_score(y_test, y_pred))
    print('Precision: ', tfidf_classifier.precision_score(y_test, y_pred))
    print('Recall: ', tfidf_classifier.recall_score(y_test, y_pred))