from process import preprocess
from process import readdata
from embed import w2vec
from math import floor

def main():
    raw_data = preprocess.clean_process(readdata.read())
    partition = floor(len(raw_data.index)*0.7)
    model = w2vec.get_model(raw_data, partition)

    test = raw_data[partition:]
    model.save("models/w2vmodel.mod")
    
    w2vec.experiment(test, model)

if __name__ == "__main__":
    main()
    