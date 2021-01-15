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
    results_d2v = bc.train_test_models(X_d2v_vectorized, Y, 'd2v', models, 200)

    print(pd.DataFrame(data=results_d2v))
    ''' Results are automatically saved in results/ '''
    # pd.DataFrame(data=results_d2v).to_csv('results/results_d2v.csv')
    
