from process import preprocess, readdata
from embed import tfidf_classifier
from classifiers import binaryclassification as bc

vectorizer = tfidf_classifier.vectorizer
classifier = tfidf_classifier.classifier

if __name__ == "__main__":
    print('--- reading data ---')
    data = readdata.read()
    print(data['is_duplicate'].value_counts())

    print('--- concatting ---')
    combined = tfidf_classifier.concat_df(data)

    print('--- vectorizing ---')
    X = vectorizer.fit_transform(combined)
    
    X_train, X_test, y_train, y_test = tfidf_classifier.split_train_test_concat(X, data['is_duplicate'])

    print('--- training model ---')
    model = classifier.fit(X_train, y_train)

    print('--- testing model ---')
    print(bc.test_model_tfidf(X_test,y_test,model))

    