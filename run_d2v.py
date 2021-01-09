
from process import preprocess
from process import readdata
from embed import doc2vec as d2v

if __name__ == "__main__":	

	# If a model already exists, we just need to load it.
	try:
		model = d2v.load_model()
		print("Found an existing model.")
	except:
		print("Could not find an existing model. Creating a new one. (Saved as doc2vec.model)")
		data = readdata.read()
		training_data = preprocess.clean_process(data)
		training_corpus = d2v.get_corpus(training_data)
		d2v.create_model(training_corpus, "doc2vec.model")
		
