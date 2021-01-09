
from process import preprocess
from process import readdata
from embed import doc2vec as d2v

def read_data():
	print("Reading in the data.")
	data = readdata.read()
	training_data = preprocess.clean_process(data)
	# In gensim, a "corpus" is a list of documents.
	training_corpus = list(training_data['question1']) + list(training_data['question2'])
	return(training_corpus)

if __name__ == "__main__":
	
	# Read in the data
	training_corpus = read_data()
	tagged_training_corpus = list(d2v.read_corpus(training_corpus))

	# If a model already exists, we just need to load it.
	try:
		model = d2v.load_model()
		print("Found an existing model.")
	except:
		print("Could not find an existing model. Creating a new one. (Saved as doc2vec.model)")
		model = d2v.doc2vec_model(tagged_training_corpus)
		model.save("models/doc2vec.model")