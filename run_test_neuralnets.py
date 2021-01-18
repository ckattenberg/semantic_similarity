from process import preprocess, readdata
from math import floor
from embed import w2vec, doc2vec
from classifiers import binaryclassification as bc
import numpy as np
from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing
import pandas as pd
import run_use

def join_results(list_results):
    i = 0
    for r in list_results:
        if i == 0:
            df = pd.DataFrame(data=r)
            i+=1
        else:
            df = df.join(pd.DataFrame(data=r), how = 'outer', sort=False)
    # Reorganize
    # df = df.reindex(['single_layer_200','double_layer_200','single_layer_1024','double_layer_1024'])
    return df

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

    ''' All models are located in classifiers/binaryclassification.py '''
    models = ['double_layer_200_200','triple_layer_200_200_200']

    ''' w2v '''
    X_w2v_vectorized = bc.vectorize_data_w2v(X, w2v_vectors)
    results_w2v = bc.train_test_models(X_w2v_vectorized,Y,'w2v',models, 200)

    ''' d2v '''
    X_d2v_vectorized = doc2vec.vectorize_data_d2v(X, d2v_model)
    results_d2v = bc.train_test_models(X_d2v_vectorized,Y,'d2v',models, 200)

    ''' use '''
    # USE uses raw data instead of clean_process data
    data = readdata.read()
    # run_use.create_use_pickle(data)

    X = data[['question1','question2']]
    Y = data['is_duplicate']
    models = ['double_layer_1024_1024','triple_layer_1024']

    X_use_vectorized = run_use.vectorize_data(X)
    results_use = bc.train_test_models(X_use_vectorized,Y,'use',models, 25)
    del X_use_vectorized

    # Append results to list so they can be joined into one DataFrame
    list_results = []
    list_results.append(results_w2v) 
    list_results.append(results_d2v)
    list_results.append(results_use)
    
    results = join_results(list_results)
    print(results)
    results.to_csv('results/results.csv')

''' Total Results (Clean)
\begin{table}[]
\begin{tabular}{llll}
Neural net                   & w2v   & d2v   & use   \\
single\_layer\_200           & 0.783 & 0.740 &       \\
double\_layer\_200           & 0.764 & 0.724 &       \\
double\_layer\_200\_200      & 0.763 & 0.721 &       \\
triple\_layer\_200\_200\_200 & 0.763 & 0.719 &       \\
single\_layer\_1024          &       &       & 0.842 \\
double\_layer\_1024          &       &       & 0.843 \\
triple\_layer\_1024          &       &       & 0.844
\end{tabular}
\end{table}
'''


''' Lemmatized vs not Lemmatized
\begin{table}[]
\begin{tabular}{lllll}
                   & \multicolumn{2}{l}{Lemmatized} & \multicolumn{2}{l}{Not Lemmatized} \\
Neural net         & w2v            & d2v           & w2v              & d2v             \\
single\_layer\_200 & 0.786          & 0.730         & 0.783            & 0.740           \\
double\_layer\_200 & 0.780          & 0.713         & 0.764            & 0.724          
\end{tabular}
\end{table}
'''

''' TFIDF Cosine vs NB
\begin{table}[]
\begin{tabular}{lll}
          & Cosine & Naive Bayes \\
Accuracy  & 0.668  & 0.740       \\
Precision & 0.586  & 0.709       \\
Recall    & 0.303  & 0.480       \\
F-1       & 0.400  & 0.572      
\end{tabular}
\end{table}
'''