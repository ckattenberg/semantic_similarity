from nltk.tokenize import word_tokenize, RegexpTokenizer
import pandas as pd
from math import floor
from nltk.stem.snowball import SnowballStemmer
import numpy as np
from gensim.parsing.preprocessing import remove_stopwords

def process(data):
    '''Applies standard nltk word tokenizer to the questions, replacing the original
    strings with the tokenized versions.'''
    data.question1 = data.question1.apply(word_tokenize)
    data.question2 = data.question2.apply(word_tokenize)
    return data

def clean_process(data):
    '''Applies a regex-based nltk word tokenizer to the questions, replacing the original
    strings with the tokenized versions.'''
    tokenizer = RegexpTokenizer(r'\w+\'*\w*')
    data.question1 = data.question1.apply(tokenizer.tokenize)
    data.question2 = data.question2.apply(tokenizer.tokenize)
    data1 = data[data.question1.str.len() != 0]
    data2 = data1[data1.question2.str.len() != 0]
    return data2

def clean_split(data):
    '''Cleans the dataset, then splits it into a training and a testing dataframe.'''
    clean = clean_process(data)
    partition = floor(len(clean.index)*0.7)
    train = pd.DataFrame(clean[:partition])
    test = pd.DataFrame(clean[partition:])

    return train, test

# def sentence_remove_stopwords(sentence):
#     return [w for w in sentence if not w in stopwords.words('english')]

def clear_stopwords(data):
    data.question1 = data.question1.apply(remove_stopwords)
    data.question2 = data.question2.apply(remove_stopwords)
    return data

def untokenized_split(data):
    partition = floor(len(data.index)*0.7)
    train = pd.DataFrame(data[:partition])
    test = pd.DataFrame(data[partition:])

    return train, test

def stem_sentence(sentence, stemmer):
    result = []
    for word in sentence:
        result.append(stemmer.stem(word))
    return result

def stem_data(data):
    '''Stems the words in the sentences. Requires tokenization first.'''
    stemmer = SnowballStemmer("english")
    data.question1 = data.question1.apply(lambda sentence: stem_sentence(sentence, stemmer))
    data.question2 = data.question2.apply(lambda sentence: stem_sentence(sentence, stemmer))
    return data
    
def split_train_test(data, partition_size = 0.7):
    partition = floor(len(data)*partition_size)
    X_train = data[['question1','question2']][:partition]
    X_test = data[['question1','question2']][partition:]
    y_train = np.array(data['is_duplicate'][:partition])
    y_test = np.array(data['is_duplicate'][partition:])

    return X_train, y_train, X_test, y_test

def split_train_test_vect(X, Y, partition_size = 0.7):
    partition = floor(len(Y)*partition_size)
    X_train = X[:partition]
    X_test = X[partition:]
    y_train = Y[:partition]
    y_test = Y[partition:]
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    pass