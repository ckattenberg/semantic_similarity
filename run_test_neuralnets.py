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
    df = df.reindex(['single_layer_200','double_layer_200','single_layer_1024','double_layer_1024'])
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
    models = ['single_layer_200','double_layer_200']

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
    models = ['single_layer_1024','double_layer_1024']

    X_use_vectorized = run_use.vectorize_data(X)
    results_use = bc.train_test_models(X_use_vectorized,Y,'use',models, 25)
  

    # Append results to list so they can be joined into one DataFrame
    list_results = []
    list_results.append(results_w2v) 
    list_results.append(results_d2v)
    list_results.append(results_use)
    
    results = join_results(list_results)
    print(results)
    results.to_csv('results/results.csv')
    