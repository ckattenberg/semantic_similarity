from process import preprocess
from process import readdata
from embed import w2vec
from math import floor
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from classifiers import binaryclassification as bc
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import numpy as np

vectorsize = 100

def train_model(X_train, y_train, batch_size = 200):
    estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=batch_size, verbose=1)
    estimator.fit(X_train, y_train)
    return estimator

def create_baseline():
    	# create model
    model = Sequential()
    model.add(Dense(2 * vectorsize, input_dim=2 * vectorsize, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def binary_classify(data, w2v_vectors, vectorsize):
    X = data[['question1','question2']]
    Y = data['is_duplicate']
    X_w2v_vectorized = vectorize_data_w2v(X, w2v_vectors, vectorsize)
    X_train, X_test, y_train, y_test = preprocess.split_train_test_vect(X_w2v_vectorized, Y)
    model = train_model(X_train, y_train, 5000)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, prec, rec, f1

def vectorize_w2v(model, s1, s2, vector_size):
    vectors = model.wv
    sen_vector1 = [0] * vector_size
    for word in s1:
        try:
            try:
                fac = 50/vectors.vocab[word].count
            except:
                fac = 1
            sen_vector1 += fac * vectors[word]
        except:
            continue
    avg1 = np.array(sen_vector1)/len(s1)

    sen_vector2 = [0] * vector_size
    for word in s2:
        try:
            try:
                fac = 50/vectors.vocab[word].count
            except:
                fac = 1
            sen_vector2 += fac * vectors[word]
        except:
            continue
    avg2 = np.array(sen_vector2)/len(s2)
    res = np.concatenate((avg1,avg2))
    return res

def vectorize_data_w2v(data, vectors, vector_size):
    vector_list = []
    for index, row in data.iterrows():
        text1 = row['question1']
        text2 = row['question2']
        vector_list.append(vectorize_w2v(vectors, text1, text2, vector_size))
    # scaler = MinMaxScaler()
    # vector_list = scaler.fit_transform(vector_list)
    return np.array(vector_list)

def test_vectorsize(raw_data, tokenized):
    print('TESTING vectorsize')
    data = pd.DataFrame(columns=['vectorsize', 'acc', 'prec', 'rec', 'f1'])
    net_data = data.copy()
    for i in [25, 50, 64, 100, 128, 256, 512]:
        print('vectorsize: ', i)
        global vectorsize
        w2vec.vectorsize = i
        vectorsize = i
        model = w2vec.make_space(raw_data, vector_size=i)
        tp, tf, fp, fn = w2vec.experiment(raw_data, model)
        acc, prec, rec, f1 = binary_classify(tokenized, model, i)
        net_data = net_data.append({'acc':acc, 'prec':prec, 'rec':rec, 'f1':f1, 'vectorsize':i}, ignore_index=True)

        acc = (tp+tf)/sum([tp, tf, fp, fn])
        prec = tp/(tp+fp)
        rec = tp/(tp + fn)
        f1 = 2 * (prec*rec / (prec+rec))
        data = data.append({'vectorsize':i, 'acc':acc, 'prec':prec, 'rec':rec, 'f1':f1}, ignore_index=True)
    print(net_data)
    data.to_csv('results/w2v/log_vsize.csv', mode='a')
    net_data.to_csv('results/w2v/log_vsize_net.csv', mode='a')

def main():
    raw_data = preprocess.clean_process(readdata.read())
    partition = floor(len(raw_data.index)*0.7)
    model = w2vec.get_model(raw_data, partition)

    test = raw_data[partition:]
    model.save("models/w2vmodel.mod")
    
    w2vec.experiment(test, model)

if __name__ == "__main__":
    raw_data = preprocess.clean_process(readdata.read())
    test_vectorsize(raw_data, raw_data)
    