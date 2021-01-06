from nltk.tokenize import word_tokenize, RegexpTokenizer
import pandas as pd
from readdata import read
from math import floor

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
    data.dropna()
    return data

def clean_split(data):
    '''Cleans the dataset, then splits it into a training and a testing dataframe.'''
    clean = clean_process(data)
    partition = floor(len(clean.index)*0.7)
    train = pd.DataFrame(clean[:partition])
    test = pd.DataFrame(clean[partition:])

    return train, test

if __name__ == "__main__":
   print(clean_process(read()[:10]))