from embed import universal_sentence_encoder as use
from process import readdata, preprocess
import pandas as pd
import numpy as np
from classifiers import binaryclassification as bc
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf

def embed_data(data):
    q1 = data["question1"]
    q2 = data["question2"]
    em1 = q1.apply(use.encode)
    em2 = q2.apply(use.encode)
    em_df = pd.DataFrame({"em1": em1, "em2": em2})
    return em_df

def sim_data(em_df):
    similarities = em_df.apply(lambda row:use.cossim(row.em1, row.em2), axis=1)
    return similarities

def compare(sim, is_duplicate):
    threshold = 0.9
    tp = sum(np.where((sim >= threshold) & (is_duplicate == 1), 1, 0))
    tf = sum(np.where((sim < threshold) & (is_duplicate == 0), 1, 0))
    fp = sum(np.where((sim >= threshold) & (is_duplicate == 0), 1, 0))
    fn = sum(np.where((sim < threshold) & (is_duplicate == 1), 1, 0))

    print("Accuracy test: ", (tp+tf)/len(sim))
    print("Precision test: ", tp/(tp+fp))
    print("Recall test: ", tp/(tp + fn))
    return tp, tf, fp, fn

# baseline model
def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(1024, input_dim=1024, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def concat_col(em_df):
    result = []
    em_df.apply(lambda row: result.append(np.concatenate((row.em1, row.em2))), axis=1)
    return np.array(result)


def train_model(X_train, y_train, batch_size = 200):
    estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=batch_size, verbose=1)
    # scaler = preprocessing.MinMaxScaler()
    # X_train= scaler.fit_transform(X_train)
    estimator.fit(X_train, y_train)
    return estimator

if __name__ == "__main__":
    data = readdata.read()

    # Split raw_data into train/test set
    X_train, y_train, X_test, y_test = preprocess.split_train_test(data)
    print('--- vectorizing data ---')
    train_em = embed_data(X_train)
    test_em = embed_data(X_test)
    concat_em_train = concat_col(train_em)
    concat_em_test = concat_col(test_em)
    # data_train = pd.DataFrame({"train": concat_em_train})
    # data_test = pd.DataFrame({"test": concat_em_test})

    # Train model on training set
    print('--- training model ---')
    model = train_model(concat_em_train, y_train, 25)

    # Test model on test set
    print('--- testing model ---')
    accuracy = bc.test_model(concat_em_test, y_test, model)

    ''' Test Kfold '''
    # X = np.concatenate((X_train_d2v_vectorized, X_test_d2v_vectorized))
    # Y = np.append(y_train, y_test)
    # print(bc.train_test_model_kfold(X,Y, batch_size = 200))

    print('Accuracy: ', accuracy)