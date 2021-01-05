import pandas
import numpy
from sklearn.model_selection import train_test_split

def read():
    data = pandas.read_csv("quora_duplicate_questions.tsv", converters={'question1': str, 'question2':str}, sep="\t")
    return data

if __name__ == "__main__":
    print(read().dtypes)