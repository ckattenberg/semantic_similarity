import pandas
import numpy

def read():
    data = pandas.read_csv("quora_duplicate_questions.tsv", sep="\t")
    return data

if __name__ == "__main__":
    print(read())