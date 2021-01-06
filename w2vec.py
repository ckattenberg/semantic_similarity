import preprocess
import readdata
from gensim.models import word2vec
from scipy import spatial
import numpy as np
from math import floor

# Magic number; gensim's word2vec seems to use vectors of size 100
vector_size = 100

def make_space(data, partition):
    '''Combines the two lists of questions to make a single list, the result is a list of list of tokens.
    This is what the model needs for its vocab training. We then only require the vectors, returning those.'''
    traindata = data[:partition]
    vocab = list(data.question1.values) + list(data.question2.values)
    sentences = list(traindata.question1.values) + list(traindata.question2.values)

    model = word2vec.Word2Vec(window=5, min_count=1)
    model.build_vocab(vocab)
    model.train(sentences, total_examples=len(sentences), epochs=model.epochs)

    return model

def string_similarity(vectors, s1, s2):
    '''Calculates the similarity of a "sentence", a list of words, by summing
    the word-vectors of those words and taking the average.'''
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

    return 1 - spatial.distance.cosine(avg1, avg2)

def calc_similarity(data, vectors):
    '''Calculates the similarity for each question pair, returning this.'''
    return data.apply(lambda row: string_similarity(vectors, row.question1, row.question2), axis=1)

if __name__ == "__main__":
    raw_data = preprocess.clean_process(readdata.read())
    partition = floor(len(raw_data.index)*0.7)
    try:
        model = word2vec.Word2Vec.load("w2vmodel.mod")
    except:
        model = make_space(raw_data, partition)

    test = raw_data[partition:]
    model.save("w2vmodel.mod")
    vectors = model.wv

    
    result = calc_similarity(test, vectors)
    
    # Counts true positives (similarity > 0.9 and duplicate) and true negatives ()
    tp = sum(np.where((result >= 0.9) & (test.is_duplicate == 1), 1, 0))
    tf = sum(np.where((result < 0.9) & (test.is_duplicate == 0), 1, 0))
    fp = sum(np.where((result >= 0.9) & (test.is_duplicate == 0), 1, 0))
    fn = sum(np.where((result < 0.9) & (test.is_duplicate == 1), 1, 0))

    print("Accuracy test: ", (tp+tf)/len(result))
    print("Precision test: ", tp/(tp+fp))
    print("Recall test: ", tp/(tp + fn))