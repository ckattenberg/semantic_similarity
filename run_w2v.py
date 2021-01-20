from process import preprocess
from process import readdata
from embed import w2vec
from math import floor
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main():
    print("--- Reading data ---")
    raw_data = preprocess.clean_process(readdata.read())
    partition = floor(len(raw_data.index)*0.7)
    print("--- Training model ---")
    model = w2vec.make_space(raw_data)

    test = raw_data[partition:]
    model.save("models/w2vmodel.mod")
    print("Model saved.")
    
    print("--- Running experiment ---")
    w2vec.experiment(test, model)

if __name__ == "__main__":
    main()
    