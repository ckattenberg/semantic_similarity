"""
The code in this document can be used to create a Doc2Vec model with the gensim package.
Use the function load_model(filepath) to load in a model and
doc2vec(model, sent) to get the vector embedding of a sentence sent.
"""
import os
import gensim
import readdata
import preprocess
from scipy.spatial import distance
import numpy as np

# Read in the data and preprocess it.
def read_data():
	print("Reading in the data.")
	data = readdata.read()[:500]
	training_data = preprocess.clean_process(data)
	# In gensim, a "corpus" is a list of documents.
	training_corpus = list(training_data['question1']) + list(training_data['question2'])
	return(training_corpus)

# Turn corpus into TaggedDocument (as input for model training)
def read_corpus(corpus, tokens_only=False):
	for i, line in enumerate(corpus):
		if(tokens_only):
			yield line
		else:
			yield gensim.models.doc2vec.TaggedDocument(line, [i])

def doc2vec_model(train_data):
	print("Training the doc2vec model.")
	model = gensim.models.doc2vec.Doc2Vec(vector_size=100, 
		min_count=2, epochs=40)

	# Build a vocabulary
	model.build_vocab(train_data)

	# Train the doc2vec model on the training data
	model.train(train_data, total_examples=model.corpus_count, 
		epochs=model.epochs)

	return(model)

def load_model(filepath):
	model = gensim.models.Doc2Vec.load(filepath)
	return(model)

def doc2vec(model, sent):
	vector = model.infer_vector(sent)
	return(vector)


# Main function
if __name__ == "__main__":

	# Read in the data
	training_corpus = read_data()
	tagged_training_corpus = list(read_corpus(training_corpus))

	# If a model already exists, we just need to load it.
	try:
		model = gensim.models.Doc2Vec.load("doc2vec.model")
		print("Found an existing model.")
	except:
		print("Could not find an existing model. Creating a new one. (Saved as doc2vec.model)")
		model = doc2vec_model(tagged_training_corpus)
		model.save("doc2vec.model")
