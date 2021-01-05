import pandas
import numpy
from sklearn.model_selection import train_test_split

def read(test_percentage):
    data = pandas.read_csv("quora_duplicate_questions.tsv", converters={'question1': str, 'question2':str}, sep="\t")
    train, test = train_test_split(data, test_size=test_percentage)
    return(train, test)

if __name__ == "__main__":
    print(read().dtypes)