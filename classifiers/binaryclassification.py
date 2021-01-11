import pandas
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
from sklearn.metrics import accuracy_score

from progress.bar import Bar
from sklearn import naive_bayes
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier



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

vector_size = 100

# Returns average vector representation of q1 followed by avg vector representation of q2
def vectorize_w2v(vectors, s1, s2):
    sen_vector1 = [0] * vector_size
    for word in s1:
        sen_vector1 += vectors[word]
    avg1 = np.array(sen_vector1)/len(s1)

    sen_vector2 = [0] * vector_size
    for word in s2:
        sen_vector2 += vectors[word]
    avg2 = np.array(sen_vector2)/len(s2)
    res = np.concatenate((avg1,avg2))
    return res

# Returns a list of combined vectors of q1 and q2
def vectorize_data_w2v(data, vectors):
    vector_list = []
    
    with Bar('w2v_vectorizing', max=len(data)) as bar:
        for index, row in data.iterrows():
            text1 = row['question1']
            text2 = row['question2']
            vector_list.append(vectorize_w2v(vectors, text1, text2))
            bar.next()
    # scaler = StandardScaler()
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

def train_model(X_train, y_train, batch_size = 200):
    estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=batch_size, verbose=1)
    # scaler = preprocessing.MinMaxScaler()
    # X_train= scaler.fit_transform(X_train)
    estimator.fit(X_train, y_train)
    return estimator

def test_model(X_test, y_test, model):
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

if __name__ == "__main__":
    pass
