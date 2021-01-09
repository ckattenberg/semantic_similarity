from gensim.models import word2vec
from scipy import spatial
import numpy as np
from math import floor

# Magic number; gensim's word2vec seems to use vectors of size 100
vector_size = 100

def make_space(data, partition=None):
    '''Combines the two lists of questions to make a single list, the result is a list of list of tokens.
    This is what the model needs for its vocab training.'''
    if partition == None:
        partition = floor(len(data.index)*0.7)
    traindata = data[:partition]
    vocab = list(data.question1.values) + list(data.question2.values)
    sentences = list(traindata.question1.values) + list(traindata.question2.values)

    model = word2vec.Word2Vec(window=5, min_count=1, sample=0.1, ns_exponent=1)
    model.build_vocab(vocab)
    model.train(sentences, total_examples=len(sentences), epochs=model.epochs)

    return model

def string_similarity(model, s1, s2):
    '''Calculates the similarity of a "sentence", a list of words, by summing
    the word-vectors of those words and taking the average.'''
    vectors = model.wv
    sen_vector1 = [0] * vector_size
    for word in s1:
        fac = 50/vectors.vocab[word].count
        sen_vector1 += fac * vectors[word]
    avg1 = np.array(sen_vector1)/len(s1)

    sen_vector2 = [0] * vector_size
    for word in s2:
        fac = 50/vectors.vocab[word].count
        sen_vector2 += fac * vectors[word]
    avg2 = np.array(sen_vector2)/len(s2)

    return 1 - spatial.distance.cosine(avg1, avg2)

def calc_similarity(data, model):
    '''Calculates the similarity for each question pair, returning this.'''
    return data.apply(lambda row: string_similarity(model, row.question1, row.question2), axis=1)

def get_model(raw_data, partition):
    try:
        print('--- loading model ---')
        w2v_model = word2vec.Word2Vec.load("models/w2vmodel.mod")
        print('w2v_model loaded')
    except:
        print('--- embedding ---')
        w2v_model = make_space(raw_data, partition)

    return w2v_model

def experiment(data, model, threshold=0.9):
    '''Counts results, then calculates resulting statistics.'''
    result = calc_similarity(data, model)

    tp = sum(np.where((result >= threshold) & (data.is_duplicate == 1), 1, 0))
    tf = sum(np.where((result < threshold) & (data.is_duplicate == 0), 1, 0))
    fp = sum(np.where((result >= threshold) & (data.is_duplicate == 0), 1, 0))
    fn = sum(np.where((result < threshold) & (data.is_duplicate == 1), 1, 0))

    print("Accuracy test: ", (tp+tf)/len(result))
    print("Precision test: ", tp/(tp+fp))
    print("Recall test: ", tp/(tp + fn))
    return tp, tf, fp, fn

if __name__ == "__main__":
    pass