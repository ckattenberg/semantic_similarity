import binaryclassification as bc
import preprocess
import readdata
from math import floor
import numpy as np
from wikipedia2vec import Wikipedia2Vec as W2V
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

'''
Modification using binaryclassification.py using a Wikipedia trained vector list instead.
'''

# Magic number for standard gensim vector size
vector_size = 100

def vectorize_sent(vectors, s1, s2):
    sen_vector1 = [0] * vector_size
    for word in s1:
        try:
            sen_vector1 += vectors.get_word_vector(word)
        except:
            continue
    avg1 = np.array(sen_vector1)/len(s1)

    sen_vector2 = [0] * vector_size
    for word in s2:
        try:
            sen_vector2 += vectors.get_word_vector(word)
        except:
            continue
    avg2 = np.array(sen_vector2)/len(s2)
    res = np.concatenate((avg1,avg2))
    return res

if __name__ == "__main__":
    raw_data = preprocess.clean_process(readdata.read())
    partition = floor(len(raw_data.index)*1)

    vectors = W2V.load("enwiki_20180420_100d.pkl")

    vector_list = []

    # Create list of vectors corresponding to the combined vector of q1 and q2.
    for id in raw_data['id']:
        text1 = raw_data['question1'][id]
        text2 = raw_data['question2'][id]
        vector_list.append(vectorize_sent(vectors, text1, text2))

    X = np.array(vector_list)
    Y = np.array(list(raw_data['is_duplicate']))

    print('--- testing ---')
    estimator = KerasClassifier(build_fn=bc.create_baseline, epochs=100, batch_size=200, verbose=1)
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    results = cross_val_score(estimator, X, Y, cv=kfold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))