from process import preprocess, readdata
from math import floor
from embed import w2vec, doc2vec
from classifiers import binaryclassification as bc
import run_use
import numpy as np
from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing

if __name__ == "__main__":
    print('--- reading data ---')
    data = preprocess.clean_process(readdata.read())

    partition = floor(len(data.index)*1)
    w2v_model = w2vec.get_model(data, partition)
    w2v_model.save("models/w2vmodel.mod")
    w2v_vectors = w2v_model.wv

    d2v_model = doc2vec.load_model("models/doc2vec.model")

    ''' Run once to create pickle to speed up vectorizing in future. (600 MB) '''
    # bc.create_w2v_pickle(data, w2v_vectors)
    # doc2vec.create_d2v_pickle(data, d2v_model)

    X = data[['question1','question2']]
    Y = data['is_duplicate']

    ''' w2v '''
    X_w2v_vectorized = bc.vectorize_data_w2v(X, w2v_vectors)
    X_train, X_test, y_train, y_test = preprocess.split_train_test_vect(X_w2v_vectorized, Y)
    model = bc.train_model(X_train, y_train, 200)
    results_w2v = bc.test_model(X_test, y_test, model)
    ''' Accuracy, Precision, Recall, F1 '''
    print(results_w2v)

    ''' d2v '''
    X_d2v_vectorized = doc2vec.vectorize_data_d2v(X, d2v_model)
    X_train, X_test, y_train, y_test = preprocess.split_train_test_vect(X_d2v_vectorized, Y)
    model = bc.train_model(X_train, y_train, 200)
    results_d2v = bc.test_model(X_test, y_test, model)
    print(results_d2v)

    ''' USE '''
    data = readdata.read()
    X = data[['question1','question2']]
    Y = data['is_duplicate']
    X_use_vectorized = run_use.vectorize_data(X)
    X_train, X_test, y_train, y_test = preprocess.split_train_test_vect(X_use_vectorized, Y)
    model = bc.train_model_use(X_train, y_train, 25)
    results_use = bc.test_model(X_test, y_test, model)
    print(results_use)
    



    print('w2v: ', results_w2v)
    print('d2v: ', results_d2v)
    print('use: ', results_use)