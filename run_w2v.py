from process import preprocess
from process import readdata
from embed import w2vec
import pandas as pd
from math import floor
import classifiers.binaryclassification as bc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def binary_classify(data, w2v_vectors):
    X = data[['question1','question2']]
    Y = data['is_duplicate']
    X_w2v_vectorized = bc.vectorize_data_w2v(X, w2v_vectors)
    X_train, X_test, y_train, y_test = preprocess.split_train_test_vect(X_w2v_vectorized, Y)
    model = bc.train_model(X_train, y_train, 5000)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, prec, rec, f1

def test_alpha(raw_data, tokenized):
    print('TESTING ALPHA')
    data = pd.DataFrame(columns=['alpha', 'acc', 'prec', 'rec', 'f1'])
    net_data = data.copy()
    for i in range(1, 11):
        print('Alpha: ', i/100)
        model = w2vec.make_space(raw_data, alpha=i/100)
        tp, tf, fp, fn = w2vec.experiment(raw_data, model)
        acc, prec, rec, f1 = binary_classify(tokenized, model)
        net_data = net_data.append({'acc':acc, 'prec':prec, 'rec':rec, 'f1':f1, 'alpha':i/100}, ignore_index=True)

        acc = (tp+tf)/sum([tp, tf, fp, fn])
        prec = tp/(tp+fp)
        rec = tp/(tp + fn)
        f1 = 2 * (prec*rec / (prec+rec))
        data = data.append({'alpha':i/100, 'acc':acc, 'prec':prec, 'rec':rec, 'f1':f1}, ignore_index=True)
    print(net_data)
    data.to_csv('results/w2v/log_alpha.csv', mode='a')
    net_data.to_csv('results/w2v/log_alpha_net.csv', mode='a')

def test_sample(raw_data, tokenized):
    print('TESTING SAMPLE')
    data = pd.DataFrame(columns=['sample', 'acc', 'prec', 'rec', 'f1'])
    net_data = pd.DataFrame(columns=['sample', 'acc', 'prec', 'rec', 'f1'])
    for i in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
        print('sample: ', i)
        model = w2vec.make_space(raw_data, sample = i)
        tp, tf, fp, fn = w2vec.experiment(raw_data, model)
        acc, prec, rec, f1 =  binary_classify(tokenized, model)
        net_data = net_data.append({'sample':i, 'acc':acc, 'prec':prec, 'rec':rec, 'f1':f1, }, ignore_index=True)

        acc = (tp+tf)/sum([tp, tf, fp, fn])
        prec = tp/(tp+fp)
        rec = tp/(tp + fn)
        f1 = 2 * (prec*rec / (prec+rec))
        data = data.append({'sample':i, 'acc':acc, 'prec':prec, 'rec':rec, 'f1':f1}, ignore_index=True)
    data.to_csv('results/w2v/log_sample.csv', mode='a')
    net_data.to_csv('results/w2v/log_sample_net.csv', mode='a')

def test_neg(raw_data, tokenized):
    print('TESTING neg')
    data = pd.DataFrame(columns=['neg', 'acc', 'prec', 'rec', 'f1'])
    net_data = pd.DataFrame(columns=['neg', 'acc', 'prec', 'rec', 'f1'])
    for i in [0, 5, 10, 20]:
        print('neg: ', i)
        model = w2vec.make_space(raw_data, negative = i)
        tp, tf, fp, fn = w2vec.experiment(raw_data, model)
        acc, prec, rec, f1 =  binary_classify(tokenized, model)
        net_data = net_data.append({'neg':i, 'acc':acc, 'prec':prec, 'rec':rec, 'f1':f1, }, ignore_index=True)

        acc = (tp+tf)/sum([tp, tf, fp, fn])
        prec = tp/(tp+fp)
        rec = tp/(tp + fn)
        f1 = 2 * (prec*rec / (prec+rec))
        data = data.append({'neg':i, 'acc':acc, 'prec':prec, 'rec':rec, 'f1':f1}, ignore_index=True)
    data.to_csv('results/w2v/log_neg.csv', mode='a')
    net_data.to_csv('results/w2v/log_neg_net.csv', mode='a')

def test_neg(raw_data, tokenized):
    print('TESTING ns')
    data = pd.DataFrame(columns=['ns', 'acc', 'prec', 'rec', 'f1'])
    net_data = pd.DataFrame(columns=['ns', 'acc', 'prec', 'rec', 'f1'])
    for i in range(-10, 11):
        print('ns: ', i/100)
        model = w2vec.make_space(raw_data, ns_exponent = i/100)
        tp, tf, fp, fn = w2vec.experiment(raw_data, model)
        acc, prec, rec, f1 =  binary_classify(tokenized, model)
        net_data = net_data.append({'ns':i, 'acc':acc, 'prec':prec, 'rec':rec, 'f1':f1, }, ignore_index=True)

        acc = (tp+tf)/sum([tp, tf, fp, fn])
        prec = tp/(tp+fp)
        rec = tp/(tp + fn)
        f1 = 2 * (prec*rec / (prec+rec))
        data = data.append({'ns':i, 'acc':acc, 'prec':prec, 'rec':rec, 'f1':f1}, ignore_index=True)
    data.to_csv('results/w2v/log_ns.csv', mode='a')
    net_data.to_csv('results/w2v/log_ns_net.csv', mode='a')

def main():
    raw_data = preprocess.clean_process(readdata.read())
    partition = floor(len(raw_data.index)*0.7)
    model = w2vec.get_model(raw_data, partition)

    test = raw_data[partition:]
    model.save("models/w2vmodel.mod")
    
    result = w2vec.calc_similarity(test, model)
    w2vec.experiment(test, model)

if __name__ == "__main__":
    raw_data = readdata.read()
    token_data = preprocess.clean_process(raw_data.copy())
    clean_data = preprocess.stem_data(preprocess.clean_process(preprocess.clear_stopwords(raw_data)))
    print('-Using fully cleaned data-')
    test_neg(clean_data, token_data)

    print('-Using only tokenized data-')
    test_neg(token_data, token_data)