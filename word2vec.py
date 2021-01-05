import gensim
import readdata
import preprocess
import nltk
import tempfile
from gensim.models import Word2Vec


# Trains the word2vec model
def word2vec_model(data):
	model = gensim.models.Word2Vec(data, min_count = 1, size = 100, 
                                             	window = 5, sg = 1)
	return(model)

# Compares 2 words using the model
def word2vec_similarity(word1, word2, model):
	simil = model.wv.similarity(word1, word2)
	print("Cosine similarity between " + word1 +
	      " and " + word2 + " - Skip Gram : ", simil)

	return(simil)

def get_vector(word, model):
	vec = model.wv[word]
	return(vec)

# Save model
def save_model(model):
	model.save("word2vec.model")
	return

def load_model(filepath):
	model = gensim.models.Word2Vec.load(filepath)
	return(model)


if __name__ == "__main__":
	# Read in the data
	data = readdata.read()
	data = preprocess.clean_process(data)
	corpus = list(data['question1']) + list(data['question2'])

	temp = []
	for sentence in corpus:
		for word in sentence:
			temp.append(word.lower())

	# Create Skip Gram model 
	model = word2vec_model([temp])

	# Test the model
	word2vec_similarity("cats", "dogs", model)

	get_vector("cats", model)