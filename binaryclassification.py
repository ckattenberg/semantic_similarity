import pandas
import w2vec
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import preprocess
import readdata
from gensim.models import word2vec
from scipy import spatial
import numpy as np
from math import floor

# baseline model
def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(200, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

vector_size = 100

# Calculates average vector of question1 and appends average vector of question 2
def vectorize_sent(vectors, s1, s2):
    if len(s1) == 0 or len(s2) == 0:
        print('beep')
        return 0
    
    sen_vector1 = [0] * vector_size
    for word in s1:
        sen_vector1 += vectors[word]
    avg1 = np.array(sen_vector1)/len(s1)

    sen_vector2 = [0] * vector_size
    for word in s2:
        sen_vector2 += vectors[word]
    avg2 = np.array(sen_vector2)/len(s2)
    print(avg2)

    return avg1+avg2

if __name__ == "__main__":
    # 
    raw_data = preprocess.clean_process(readdata.read())
    partition = floor(len(raw_data.index)*0.7)
    try:
        model = word2vec.Word2Vec.load("w2vmodel.mod")
    except:
        model = make_space(raw_data, partition)

    test = raw_data[partition:]
    model.save("w2vmodel.mod")
    vectors = model.wv
   
    vector_list = []
    # test = test.loc[test['is_duplicate'] == 1][:200].append(test.loc[test['is_duplicate'] == 0][:200])

    for id in test['id']:
        text1 = test['question1'][id]
        text2 = test['question2'][id]
        vector_list.append(vectorize_sent(vectors, text1, text2))
        

    X = np.array(vector_list)
    Y = np.array(list(test['is_duplicate']))

    print(test['is_duplicate'].value_counts())

    estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=5, verbose=0)
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    results = cross_val_score(estimator, X[:500], Y[:500], cv=kfold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

