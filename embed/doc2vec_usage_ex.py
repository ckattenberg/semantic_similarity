from . import doc2vec

def read_data():
	print("Reading in the data.")
	data = readdata.read()
	training_data = preprocess.clean_process(data)
	# In gensim, a "corpus" is a list of documents.
	training_corpus = list(training_data['question1']) + list(training_data['question2'])
	return(training_corpus)
	
# Read in the data
training_corpus = read_data()
doc2vec.create_model(training_corpus, "doc2vec.model")
model = doc2vec.load_model("doc2vec.model")
embedding = doc2vec.doc2vec(model, ["The quick brown fox jumps over the lazy dog."])

print(embedding)
