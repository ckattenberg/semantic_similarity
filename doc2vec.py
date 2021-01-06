import os
import gensim
import readdata
import preprocess

# Read in the data and preprocess it.
def read_data():
	data = readdata.read()[:3]
	data = preprocess.clean_process(data)
	corpus = list(data['question1']) + list(data['question2'])
	return(corpus)

# Turn corpus into TaggedDocument (as input for model training)
def read_corpus(corpus):
	for i, line in enumerate(corpus):
		yield gensim.models.doc2vec.TaggedDocument(line, [i])

# Main function
if __name__ == "__main__":

	# Read in the data
	corpus = read_data()
	train_corpus = list(read_corpus(corpus))



