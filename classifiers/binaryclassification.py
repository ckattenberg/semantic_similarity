import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from gensim.models import word2vec
from scipy import spatial
import numpy as np
from math import floor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
from progress.bar import Bar
from sklearn import naive_bayes
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from collections import defaultdict
import os
import tensorflow as tf
from tensorflow import keras

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow import keras



from sklearn import datasets, linear_model

# baseline model
def create_baseline():
	# create model
    model = Sequential()
    model.add(Dense(200, input_dim=200, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def single_layer_200():
	# create model
    model = Sequential()
    model.add(Dense(200, input_dim=200, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def double_layer_200():
	# create model
    model = Sequential()
    model.add(Dense(200, input_dim=200, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def double_layer_400():
    # create model
    model = Sequential()
    model.add(Dense(200, input_dim=400, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def double_layer_200_200():
	# create model
    model = Sequential()
    model.add(Dense(200, input_dim=200, activation='relu'))
    model.add(Dense(200, input_dim=200, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def triple_layer_200_200_200():
	# create model
    model = Sequential()
    model.add(Dense(200, input_dim=200, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def single_layer_1024():
	# create model
    model = Sequential()
    model.add(Dense(1024, input_dim=1024, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def double_layer_1024():
	# create model
    model = Sequential()
    model.add(Dense(1024, input_dim=1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def double_layer_1024_1024():
	# create model
    model = Sequential()
    model.add(Dense(1024, input_dim=1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def triple_layer_1024():
	# create model
    model = Sequential()
    model.add(Dense(1024, input_dim=1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

vector_size = 100

# Returns average vector representation of q1 followed by avg vector representation of q2
def vectorize_w2v(vectors, s1, s2):
    sen_vector1 = [0] * vector_size
    for word in s1:
        try:
            sen_vector1 += vectors[word]
        except:
            continue
    avg1 = np.array(sen_vector1)/len(s1)

    sen_vector2 = [0] * vector_size
    for word in s2:
        try:
            sen_vector2 += vectors[word]
        except:
            continue
    avg2 = np.array(sen_vector2)/len(s2)
    res = np.concatenate((avg1,avg2))
    return res

def create_w2v_pickle(data, vectors):
    vector_list = []
    with Bar('creating_pickle', max=len(data)) as bar:
        for index, row in data.iterrows():
            text1 = row['question1']
            text2 = row['question2']
            vector_list.append(vectorize_w2v(vectors, text1, text2))
            bar.next()

    with open('data/w2v_vectors.p', 'wb') as f:
        pickle.dump(vector_list, f)

# Returns a list of combined vectors of q1 and q2
def vectorize_data_w2v(data, vectors):
    vector_list = []
    try:
        with open('data/w2v_vectors.p', 'rb') as f:
            vector_list = pickle.load(f)[:len(data.index)]
            print('Found pickle')
    except:
        with Bar('w2v_vectorizing', max=len(data)) as bar:
            for index, row in data.iterrows():
                text1 = row['question1']
                text2 = row['question2']
                vector_list.append(vectorize_w2v(vectors, text1, text2))
                bar.next()
    # scaler = MinMaxScaler()
    # vector_list = scaler.fit_transform(vector_list)
    return np.array(vector_list)

# Takes X and Y as input, where X = list of list with combined vectors of q1 and q2, and Y = list of is_duplicate values.
# Split dataset into 90% train 10% test, trains data on train data, test on test data,
# Repeat 10 times (kfold), calculate accuracy for each fold, return average accuracy of all folds.
def train_test_model_kfold(X, Y, batch_size = 25):
    estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=batch_size, verbose=1)
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    results = cross_val_score(estimator, X, Y, cv=kfold)
    return (results.mean()*100, results.std()*100)


def train_model_400(X_train, y_train, batch_size = 200):
    estimator = KerasClassifier(build_fn=double_layer_400, epochs=100, batch_size=batch_size, verbose=1)
    estimator.fit(X_train, y_train)
    return estimator

def train_model(X_train, y_train, model_name, batch_size = 200):
    try:
        estimator = keras.models.load_model('neuralnets/'+model_name+'.h5')
        print(model_name, ' neural net loaded')
    except:
        estimator = create_baseline()
        print('training neural net')
        estimator.fit(X_train, y_train, epochs = 100, batch_size = batch_size, verbose = 1)
        estimator.save('neuralnets/'+model_name+'.h5', save_format = 'h5')

    return estimator

def train_model_use(X_train, y_train, model_name, batch_size = 25):
    try:
        estimator = keras.models.load_model('neuralnets/'+model_name+'.h5')
        print(model_name, ' neural net loaded')
    except:
        estimator = single_layer_1024()
        print('training neural net')
        estimator.fit(X_train, y_train, epochs = 100, batch_size = batch_size, verbose = 1)
        estimator.save('neuralnets/'+model_name+'.h5', save_format = 'h5')

    return estimator

def split_train_test_vect(X, Y, partition_size = 0.7):
    partition = floor(len(Y)*partition_size)
    X_train = X[:partition]
    X_test = X[partition:]
    y_train = Y[:partition]
    y_test = Y[partition:]
    return X_train, X_test, y_train, y_test

    

def test_model(X_test, y_test, model):
    y_pred = (model.predict(X_test) > 0.5).astype("int32")

    return accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1_score(y_test, y_pred)
<<<<<<< HEAD
=======

>>>>>>> 8727bec5adf9722e095585afd369cba3762dd2b1

def train_test_models(X_vectorized, Y, method, models = ['create_baseline'], batch_size = 200):
    accuracies = defaultdict(dict)

    X_train, X_test, y_train, y_test = split_train_test_vect(X_vectorized, Y)
    for model in models:
        estimator = KerasClassifier(build_fn=eval(model), epochs=100, batch_size=batch_size, verbose=1)
        estimator.fit(X_train, y_train)
        accuracies[method][model] = test_model(X_test, y_test, estimator)
        print('Accuracy: ', accuracies[method][model])
	
    # Create a /models folder if it does not exist yet.
    if not os.path.exists('results'):
        os.makedirs('results')

    pd.DataFrame(data=accuracies).to_csv('results/results_'+method+'.csv')
    return accuracies

if __name__ == "__main__":
    pass

