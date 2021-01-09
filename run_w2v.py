from process import preprocess
from process import readdata
from embed import w2vec
from gensim.models import word2vec
from math import floor

if __name__ == "__main__":
    raw_data = preprocess.stem_data(preprocess.clean_process(readdata.read()))
    model = w2vec.get_model(raw_data, 0.7)

    test = raw_data[floor(len(raw_data.index)*0.7):]
    model.save("models/w2vmodel.mod")
    
    result = w2vec.calc_similarity(test, model)
    w2vec.experiment(test, model)