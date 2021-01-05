from nltk.tokenize import word_tokenize, RegexpTokenizer
import pandas as pd
from readdata import read
import re

def process(data):
    data.question1 = data.question1.apply(word_tokenize)
    data.question2 = data.question2.apply(word_tokenize)
    return data

def clean_process(data):
    tokenizer = RegexpTokenizer(r'\w+\'*\w*')
    data.question1 = data.question1.apply(tokenizer.tokenize)
    data.question2 = data.question2.apply(tokenizer.tokenize)
    return data

if __name__ == "__main__":
   print(clean_process(read()[:10]))