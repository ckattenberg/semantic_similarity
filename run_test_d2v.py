from process import preprocess, readdata
from embed import doc2vec
from classifiers import binaryclassification as bc
import pandas as pd

if __name__ == "__main__":
    print('--- reading data ---')
    data = preprocess.clean_process(readdata.read())

    d2v_model = doc2vec.load_model("models/doc2vec.model")
    
    X = data[['question1','question2']]
    Y = data['is_duplicate']

    ''' All models are located in classifiers/binaryclassification.py '''
    models = ['single_layer_200','double_layer_200']

    ''' d2v '''
    X_d2v_vectorized = doc2vec.vectorize_data_d2v(X, d2v_model)
    X_train, X_test, y_train, y_test = preprocess.split_train_test_vect(X_d2v_vectorized, Y)
    model = bc.train_model(X_train, y_train, 'd2v', 200)
    results_d2v = bc.test_model(X_test, y_test, model)
    print(results_d2v)

    print(pd.DataFrame(data=results_d2v))
    ''' Results are automatically saved in results/ '''
    # pd.DataFrame(data=results_d2v).to_csv('results/results_d2v.csv')
    
