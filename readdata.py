import pandas
import numpy
from sklearn.model_selection import train_test_split

def read(loc="quora_duplicate_questions.tsv"):
    '''Reads the .tsv file into a pandas dataFrame, converting the two questions per line to string.'''
    data = pandas.read_csv("quora_duplicate_questions.tsv", converters={'question1': str, 'question2':str}, sep="\t")
    data.question1 = data.apply(lambda row: row.question1.lower(), axis=1)
    data.question2 = data.apply(lambda row: row.question2.lower(), axis=1)
    return data

if __name__ == "__main__":
    print(read().dtypes)