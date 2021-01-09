from process import preprocess
from process import readdata
from gensim.models import word2vec
from embed import wikitrained as wt

if __name__ == "__main__":
    raw_data = preprocess.clean_process(readdata.read())
    model_vectors = wt.load_vectors()


    X_train, y_train, X_test, y_test = preprocess.split_train_test(raw_data)
    X_train_w2v_vectorized = wt.get_vectors(X_train, model_vectors)
    X_test_w2v_vectorized = wt.get_vectors(X_test, model_vectors)

    print('--- training/testing ---')
    # print(train_test_model_kfold(X,Y, batch_size = 200))
    print('Accuracy: ', wt.train_test_model(X_train_w2v_vectorized, y_train, X_test_w2v_vectorized, y_test))