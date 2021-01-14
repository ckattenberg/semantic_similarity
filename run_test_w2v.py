from process import preprocess, readdata
from math import floor
from embed import w2vec
from classifiers import binaryclassification as bc
import pandas as pd

if __name__ == "__main__":
    print('--- reading data ---')
    data = preprocess.clean_process(readdata.read())

    partition = floor(len(data.index)*1)
    w2v_model = w2vec.get_model(data, partition)
    w2v_model.save("models/w2vmodel.mod")
    w2v_vectors = w2v_model.wv

    X = data[['question1','question2']]
    Y = data['is_duplicate']

    ''' All models are located in classifiers/binaryclassification.py '''
    models = ['single_layer_200','double_layer_200']

    ''' vectorize and test '''
    X_w2v_vectorized = bc.vectorize_data_w2v(X, w2v_vectors)
    results_w2v = bc.train_test_models(X_w2v_vectorized,Y,'w2v',models, 200)

    print(pd.DataFrame(data=results_w2v))
    ''' Results are automatically saved in results/ '''
    # pd.DataFrame(data=results_w2v).to_csv('results/results_w2v.csv')
    