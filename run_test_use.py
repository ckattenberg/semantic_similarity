from process import preprocess, readdata
from classifiers import binaryclassification as bc
import pandas as pd
import run_use

if __name__ == "__main__":
    print('--- reading data ---')
    data = readdata.read()

    X = data[['question1','question2']]
    Y = data['is_duplicate']

    ''' All models are located in classifiers/binaryclassification.py '''
    models = ['single_layer_1024','double_layer_1024']

    ''' use '''
    X_use_vectorized = run_use.vectorize_data(X)
    results_use = bc.train_test_models(X_use_vectorized,Y,'use',models, 25)

    print(pd.DataFrame(data=results_use))
    ''' Results are automatically saved in results/ '''
    # pd.DataFrame(data=results_use).to_csv('results/results_use.csv')