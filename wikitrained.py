import binaryclassification as bc
import preprocess
import readdata
from math import floor
import numpy as np
from gensim.models import KeyedVectors
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from sklearn.metrics import accuracy_score

'''
Modification using binaryclassification.py using a Wikipedia trained vector list instead.
'''

# Magic number for standard gensim vector size
vector_size = 100

def get_vectors(raw_data, model_vectors):
    vector_list = []
    print('--- preparing data ---')
    # Create list of combined vector of q1 and q2. List index corresponds to data entry.
    for id in raw_data['id']:
        text1 = raw_data['question1'][id]
        text2 = raw_data['question2'][id]
        vector_list.append(vectorize_sent(model_vectors, text1, text2))
    return vector_list

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

def load_vectors():
    ''''First checks for a more compatible format, otherwise tries to load
    the wikitrained model from a txt file.'''
    try:
        return KeyedVectors.load("wiki.vectors")
    except:
        print("Loading from text file. Will take a while.")
        vectors = KeyedVectors.load_word2vec_format(fname="enwiki_20180420_100d.txt", binary=False)
        vectors.save("wiki.vectors")
        return vectors
    

def make_parameters(vectors, raw_data):
    X = np.array(vectors)
    Y = np.array(list(raw_data['is_duplicate']))
    return X, Y

def train_test_model(X, Y, partition_size = 0.7, batch_size = 25):
    partition = floor(len(X)*partition_size)
    X_train = X[:partition]
    X_test = X[partition:]
    y_train = Y[:partition]
    y_test = Y[partition:]

    # Scaling data barely improves accuracy:
    # scaler = preprocessing.StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.fit_transform(X_test)

    estimator = KerasClassifier(build_fn=bc.create_baseline, epochs=100, batch_size=batch_size, verbose=1)
    estimator.fit(X_train, y_train)

    # User warning, but this doesn't fix it, same predictions:
    # y_pred = (estimator.predict(X_test) > 0.5).astype('int32')

    y_pred = estimator.predict(X_test)

    return accuracy_score(y_test,y_pred)

def experiment(X, Y):
    print('--- testing ---')
    estimator = KerasClassifier(build_fn=bc.create_baseline, epochs=100, batch_size=200, verbose=1)
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    results = cross_val_score(estimator, X, Y, cv=kfold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

if __name__ == "__main__":
    raw_data = preprocess.clean_process(readdata.read())
    model_vectors = load_vectors().vectors
    data_vectors = get_vectors(raw_data, model_vectors)
    X, Y = make_parameters(data_vectors, raw_data)

    print('--- training/testing ---')
    # print(train_test_model_kfold(X,Y, batch_size = 200))
    print('Accuracy: ', bc.train_test_model(X, Y, batch_size = 200))

    