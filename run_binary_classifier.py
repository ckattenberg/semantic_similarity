from process import preprocess, readdata
from math import floor
from embed import w2vec, doc2vec
from classifiers import binaryclassification as bc

if __name__ == "__main__":
    print('--- reading data ---')
    raw_data = preprocess.clean_process(readdata.read()[:5000])
    partition = floor(len(raw_data.index)*1)

    w2v_model = w2vec.get_model(raw_data, partition)

    w2v_model.save("models/w2vmodel.mod")
    w2v_vectors = w2v_model.wv

    d2v_model = doc2vec.load_model("models/doc2vec.model")
    print('d2v_model loaded')
   
    # Split raw_data into train/test set
    X_train, y_train, X_test, y_test = preprocess.split_train_test(raw_data)

    print('--- vectorizing data ---')
    ''' Vectorize w2v '''
    X_train_w2v_vectorized = bc.vectorize_data_w2v(X_train, w2v_vectors)
    X_test_w2v_vectorized = bc.vectorize_data_w2v(X_test, w2v_vectors)

    ''' Vectorize d2v '''
    X_train_d2v_vectorized = doc2vec.vectorize_data_d2v(X_train, d2v_model)
    X_test_d2v_vectorized = doc2vec.vectorize_data_d2v(X_test, d2v_model)

    # Train model on training set
    print('--- training model ---')
    model_w2v = bc.train_model(X_train_w2v_vectorized, y_train, 25)
    model_d2v = bc.train_model(X_train_d2v_vectorized, y_train, 25)

    # Test model on test set
    print('--- testing model ---')
    accuracy_w2v = bc.test_model(X_test_w2v_vectorized, y_test, model_w2v)
    accuracy_d2v = bc.test_model(X_test_d2v_vectorized, y_test, model_d2v)

    ''' Test Kfold '''
    # X = np.concatenate((X_train_d2v_vectorized, X_test_d2v_vectorized))
    # Y = np.append(y_train, y_test)
    # print(bc.train_test_model_kfold(X,Y, batch_size = 200))

    print('Accuracy w2v: ', accuracy_w2v)
    print('Accuracy d2v: ', accuracy_d2v)