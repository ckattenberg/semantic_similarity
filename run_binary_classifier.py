from process import preprocess, readdata
from math import floor
from embed import w2vec, doc2vec, tfidf_classifier
from classifiers import binaryclassification as bc
import run_use
import numpy as np
from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing
import pickle

if __name__ == "__main__":
    print('--- reading data ---')
    data = preprocess.clean_process(readdata.read())

    ''' Embeddings '''
    partition = floor(len(data.index)*0.7)
    w2v_model = w2vec.get_model(data, partition)
    w2v_vectors = w2v_model.wv
    d2v_model = doc2vec.load_model("models/d2v/doc2vec.model")
    tfidf_NB = pickle.load(open("models/vectorizer_NB.p", "rb"))

    ''' Classifiers '''
    NB_classifier = pickle.load(open("neuralnets/NB_classifier.pickle", "rb"))


    ''' Run once to create pickle to speed up vectorizing in future. (600 MB) '''
    # bc.create_w2v_pickle(data, w2v_vectors)
    # doc2vec.create_d2v_pickle(data, d2v_model)
    # run_use.create_use_pickle(data)

    X = data[['question1','question2']]
    Y = data['is_duplicate']

    preds = []

    ''' w2v '''
    X_w2v_vectorized = bc.vectorize_data_w2v(X, w2v_vectors)
    X_train, X_test, y_train, y_test = preprocess.split_train_test_vect(X_w2v_vectorized, Y)
    model = bc.train_model(X_train, y_train, 'w2v', 200)
    results_w2v = bc.test_model(X_test, y_test, model)
    print(results_w2v)
    preds.append((model.predict(X_test) > 0.5).astype("int32"))

    ''' d2v '''
    X_d2v_vectorized = doc2vec.vectorize_data_d2v(X, d2v_model)
    X_train, X_test, y_train, y_test = preprocess.split_train_test_vect(X_d2v_vectorized, Y)
    model = bc.train_model(X_train, y_train, 'd2v', 200)
    results_d2v = bc.test_model(X_test, y_test, model)
    print(results_d2v)
    preds.append((model.predict(X_test) > 0.5).astype("int32"))

    ''' USE '''
    data = readdata.read()
    X = data[['question1','question2']]
    Y = data['is_duplicate']
    X_use_vectorized = run_use.vectorize_data(X)
    X_train, X_test, y_train, y_test = preprocess.split_train_test_vect(X_use_vectorized, Y)
    model = bc.train_model_use(X_train, y_train, 'use', batch_size = 25)
    results_use = bc.test_model(X_test, y_test, model)
    print(results_use)
    preds.append((model.predict(X_test, batch_size = 25) > 0.5).astype("int32"))
    
    ''' tfidf '''
    combined = tfidf_classifier.concat_df(data)
    X_tfidf_vectorized = tfidf_NB.transform(combined)
    X_train, X_test, y_train, y_test = tfidf_classifier.split_train_test_concat(X_tfidf_vectorized, Y)
    model = NB_classifier
    results_tfidf = bc.test_model_tfidf(X_test, y_test, model)
    print(results_tfidf)
    preds.append(model.predict(X_test).astype("int32"))



    ''' Calculate overlapping predictions '''
    overlap = bc.overlap(bc.zip_preds(preds), y_test)
    print(overlap)


    ''' Results '''
    print('w2v: ', results_w2v)
    print('d2v: ', results_d2v)
    print('use: ', results_use)
    print('tfidf: ', results_tfidf)





''' Overlap results '''
''' Percentage of uniquely predicted predictions against other methods

\begin{table}[h!]
\begin{center}
\begin{tabular}{c|cc}
w2v   & d2v   & 0.125 \\
      & tfidf & 0.115 \\
      & use   & 0.079 \\\hline
d2v   & w2v   & 0.132 \\
      & tfidf & 0.112 \\
      & use   & 0.079 \\\hline
use   & w2v   & 0.388 \\
      & d2v   & 0.380 \\
      & tfidf & 0.178 \\\hline
tfidf & w2v   & 0.319 \\
      & d2v   & 0.308 \\
      & use   & 0.073 \\
\end{tabular}
\caption{Percentage of uniquely predicted predictions against other methods}
\end{center}
\end{table}

'''