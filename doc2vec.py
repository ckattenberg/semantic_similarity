"""
The code in this document can be used to create a Doc2Vec model with the gensim package.
There is also a function to test the accuracy of the model.
"""
import os
import gensim
import readdata
import preprocess
from scipy.spatial import distance
import numpy as np

# Read in the data and preprocess it.
def read_data():
	data = readdata.read()
	train, test = preprocess.clean_split(data)
	train_corpus = list(train['question1']) + list(train['question2'])
	true_duplicate = list(test['is_duplicate'])
	return(train_corpus, test, true_duplicate)

# Turn corpus into TaggedDocument (as input for model training)
def read_corpus(corpus, tokens_only=False):
	for i, line in enumerate(corpus):
		if(tokens_only):
			yield line
		else:
			yield gensim.models.doc2vec.TaggedDocument(line, [i])

def doc2vec_model(train_data):
	model = gensim.models.doc2vec.Doc2Vec(vector_size=5, 
		min_count=2, epochs=40)

	# Build a vocabulary
	model.build_vocab(train_data)

	# Train the doc2vec model on the training data
	model.train(train_data, total_examples=model.corpus_count, 
		epochs=model.epochs)

	return(model)

def doc2vec(model, sent):
	vector = model.infer_vector(sent)
	return(vector)

def get_model_duplicates(model, test_data, threshold=0.6):
	print("Trying model out on test data.\n")

	test1 = list(test_data['question1'])
	test2 = list(test_data['question2'])
	true_duplicate = read_data()

	model_duplicate = []
	dist_list = []

	print(len(test1))

	# For each pair of test sentences
	for i in range(len(test1)):
		# Get the vectors of both sentences
		q1 = test1[i]
		q2 = test2[i]

		vec1 = doc2vec(model, q1)
		vec2 = doc2vec(model, q2)

		# Compare their cosine similarity
		dist = distance.cosine(vec1, vec2)
		dist_list.append(dist)

		# Put into list
		if(dist > threshold):
			model_duplicate.append(1)
		else:
			model_duplicate.append(0)

	return(model_duplicate)

def assessment_measures(true_duplicate, model_duplicate):

	print("Assessing model.\n")

	for i in range(len(true_duplicate)):
		print(true_duplicate[i], model_duplicate[i], "\n")

	true_duplicate = np.array(true_duplicate)
	model_duplicate = np.array(model_duplicate)

	tp = sum(np.where((true_duplicate == 1) & (model_duplicate == 1), 1, 0))
	tn = sum(np.where((true_duplicate == 0) & (model_duplicate == 0), 1, 0))
	fp = sum(np.where((true_duplicate == 0) & (model_duplicate == 1), 1, 0))
	fn = sum(np.where((true_duplicate == 1) & (model_duplicate == 0), 1, 0))

	print("tp: ", tp, "\n")
	print("tn: ", tn, "\n")
	print("fp: ", fp, "\n")
	print("fn: ", fn, "\n")
	
	print("Accuracy test: ", (tp+tn)/len(true_duplicate))
	print("Precision test: ", tp/(tp+fp))
	print("Recall test: ", tp/(tp + fn))
	return


# Main function
if __name__ == "__main__":

	# Read in the data
	train_corpus, test, true_duplicate = read_data()
	train_data = list(read_corpus(train_corpus))

	# If a model already exists, we just need to load it.
	try:
		model = gensim.models.Doc2Vec.load("doc2vec.model")
		print("Found an existing model.\n")
	except:
		print("Could not find an existing model. Creating a new one.\n")
		model = doc2vec_model(train_data)
		model.save("doc2vec.model")

	model_dup = get_model_duplicates(model, test, threshold=0.8)
	assessment_measures(true_duplicate, model_dup)



	



