import os
import gensim
import readdata
import preprocess

def read_corpus(corpus):
	for i, line in enumerate(corpus):
		yield gensim.models.doc2vec.TaggedDocument(line, [i])


# Main function
if __name__ == "__main__":

	# Read in the data
	data = readdata.read()[:3]
	data = preprocess.clean_process(data)
	corpus = list(data['question1']) + list(data['question2'])
	tags = list(data['id'])

	print(corpus)


	train_corpus = list(read_corpus(corpus))

	print(train_corpus)


