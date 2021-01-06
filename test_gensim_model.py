import gensim

def load_model(filepath):
	model = gensim.models.Word2Vec.load(filepath)
	return(model)