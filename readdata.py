import pandas
import numpy

def read():
    data = pandas.read_csv("../raw data/quora_duplicate_questions.tsv", converters={'question1': str, 'question2':str}, sep="\t")
    return data

if __name__ == "__main__":
    print(read().dtypes)