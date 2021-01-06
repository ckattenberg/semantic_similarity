import os
import gensim
import readdata
import preprocess

# Read in the data and preprocess it.
def read_data():
	data = readdata.read()
	train, test = preprocess.clean_split(data)
	train_corpus = list(train['question1']) + list(train['question2'])
	test_corpus = list(test['question1']) + list(test['question2'])
	true_duplicate = list(test['is_duplicate'])
	return(train_corpus, test_corpus, true_duplicate)

# Turn corpus into TaggedDocument (as input for model training)
def read_corpus(corpus, tokens_only=False):
	for i, line in enumerate(corpus):
		if(tokens_only):
			yield line
		else:
			yield gensim.models.doc2vec.TaggedDocument(line, [i])

def doc2vec_model(train_data):
	model = gensim.models.doc2vec.Doc2Vec(vector_size=50, 
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

def test(model_path, test_data, threshold=0.6):
	model = load_model("doc2vec.model")
	test1 = list(test['question1'])
	test2 = list(test['question2'])
	true_duplicate = read_data()

	model_duplicate = []
	dist_list = []

	# For each pair of test sentences
	for i in range(len(test1)):
		# Get the vectors of both sentences
		q1 = test1[i]
		q2 = test2[i]

		vec1 = model.doc2vec(q1)
		vec2 = model.doc2vec(q2)

		# Compare their cosine similarity
		dist = distance.cosine(vec1, vec2)
		dist_list.append(dist)

		# Put into list
		if(dist > threshold):
			model_duplicate.append(1)
		else:
			model_duplicate.append(0)


# Main function
if __name__ == "__main__":

	# Read in the data
	train_corpus, test_corpus, true_duplicate = read_data()
	train_data = list(read_corpus(train_corpus))
	test_data = list(read_corpus(test_corpus, tokens_only=True))

	model = doc2vec_model(train_data)
	model.save("doc2vec.model")

	



