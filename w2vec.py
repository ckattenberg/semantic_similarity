import preprocess
import readdata
from gensim.models import word2vec
from scipy import spatial
import numpy as np
from math import floor
from numpy import log

# Magic number; gensim's word2vec seems to use vectors of size 100
vector_size = 100
# Magic number; parameters for B25
# k_1 = 1.7
# b = 0.75

# def B25_score(Q, D, model, doc_count, avgdl):
#     score = 0
#     for term in Q:
#         if term not in D:
#             continue
#         inter = IDF_calc(term, model, doc_count)
#         inter *= (D.count(term) * (k_1 + 1))/(D.count(term) + k_1 * (1-b + b * len(D)/avgdl))
#         score += inter
#     return score
    

# def IDF_calc(word, model, doc_count):
#     q = model.wv.vocab[word].count
#     ans = log((doc_count - q + 0.5)/(q + 0.5) + 1)
#     return max(ans, 0.01)

# def calc_B25(data, model, doc_count, avgdl):
#     return data.apply(lambda row: B25_score(row.question1, row.question2, model, doc_count, avgdl),axis=1)

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

def experiment(data, threshold=0.9):
    '''Counts results, then calculates resulting statistics.'''
    tp = sum(np.where((result >= threshold) & (data.is_duplicate == 1), 1, 0))
    tf = sum(np.where((result < threshold) & (data.is_duplicate == 0), 1, 0))
    fp = sum(np.where((result >= threshold) & (data.is_duplicate == 0), 1, 0))
    fn = sum(np.where((result < threshold) & (data.is_duplicate == 1), 1, 0))

    print("Accuracy test: ", (tp+tf)/len(result))
    print("Precision test: ", tp/(tp+fp))
    print("Recall test: ", tp/(tp + fn))
    return tp, tf, fp, fn

if __name__ == "__main__":
    raw_data = preprocess.stem_data(preprocess.clean_process(readdata.read()))
    try:
        model = word2vec.Word2Vec.load("w2vmodel.mod")
    except:
        model = make_space(raw_data)

    test = raw_data[floor(len(raw_data.index)*0.7):]
    model.save("w2vmodel.mod")
    
    result = calc_similarity(test, model)
    experiment(test)
