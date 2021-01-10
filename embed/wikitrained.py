from . import binaryclassification as bc
from math import floor
import numpy as np
from gensim.models import KeyedVectors
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cosine

'''
Modification using binaryclassification.py using a Wikipedia trained vector list instead.
'''

# Magic number for standard gensim vector size
vector_size = 100

def get_vectors(raw_data, model_vectors):
    vector_list = []
    print('--- preparing data ---')
    # Create list of combined vector of q1 and q2. List index corresponds to data entry.
    for index, data in raw_data.iterrows():
        text1 = data['question1']
        text2 = data['question2']
        vector_list.append(vectorize_sent(model_vectors, text1, text2))
    return np.array(vector_list)

def vectorize_sent(vectors, s1, s2):
    sen_vector1 = [0] * vector_size
    for word in s1:
        try:
            fac = 1/vectors.vocab[word].count**3
            sen_vector1 += fac * vectors.get_word_vector(word)
        except:
            continue
    avg1 = np.array(sen_vector1)/len(s1)

    sen_vector2 = [0] * vector_size
    for word in s2:
        try:
            fac = 1/vectors.vocab[word].count**3
            sen_vector2 += fac * vectors.get_word_vector(word)
        except:
            continue
    avg2 = np.array(sen_vector2)/len(s2)
    res = np.concatenate((avg1,avg2))

    return res

def load_vectors():
    ''''First checks for a more compatible format, otherwise tries to load
    the wikitrained model from a txt file.'''
    try:
        return KeyedVectors.load("models/wiki.vectors")
    except:
        print("Loading from text file. Will take a while.")
        vectors = KeyedVectors.load_word2vec_format(fname="enwiki_20180420_100d.txt", binary=False)
        vectors.save("models/wiki.vectors")
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

def train_test_model(X_train_w2v_vectorized, y_train, X_test_w2v_vectorized, y_test):
    # Train model on training set
    print('--- training model ---')
    model_w2v = bc.train_model(X_train_w2v_vectorized, y_train)
    # model_d2v = train_model(X_train_d2v_vectorized, y_train)

    # Test model on test set
    print('--- testing model ---')
    accuracy = bc.test_model(X_test_w2v_vectorized, y_test, model_w2v)
    return accuracy

if __name__ == "__main__":
    pass