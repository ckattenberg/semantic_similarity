"""
The code in this document can be used to create a Doc2Vec model with the gensim package.
Use the function load_model(filepath) to load in a model and
doc2vec(model, sent) to get the vector embedding of a sentence sent.
"""
import gensim
from scipy.spatial import distance
import numpy as np
from progress.bar import Bar
from math import floor
import pickle

# Read in the data and preprocess it.
# def read_data():
# 	print("Reading in the data.")
# 	data = readdata.read()
# 	training_data = preprocess.clean_process(data)
# 	# In gensim, a "corpus" is a list of documents.
# 	training_corpus = list(training_data['question1']) + list(training_data['question2'])
# 	return(training_corpus)

def get_corpus(data):
	# In gensim, a "corpus" is a list of documents.
	training_corpus = list(data['question1']) + list(data['question2'])
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
		min_count=2)

	# Build a vocabulary
	model.build_vocab(train_data)

	# Train the doc2vec model on the training data
	model.train(train_data, total_examples=model.corpus_count, 
		epochs=model.epochs)

	return(model)

def vectorize_data_d2v(data, model):
    vector_list = []
    try:
        with open('data/d2v_vectors.p', 'rb') as f:
            vector_list = pickle.load(f)[:len(data.index)]
            print('Found pickle')
    except:
        with Bar('d2v_vectorizing', max=len(data)) as bar:
            for index, row in data.iterrows():
                text1 = row['question1']
                text2 = row['question2']
                vector_list.append(doc2vec(model, text1, text2))
                bar.next()
    return np.array(vector_list)

def create_d2v_pickle(data, vectors):
    vector_list = []
    with Bar('creating_pickle', max=len(data)) as bar:
        for index, row in data.iterrows():
            text1 = row['question1']
            text2 = row['question2']
            vector_list.append(doc2vec(vectors, text1, text2))
            bar.next()

    with open('data/d2v_vectors.p', 'wb') as f:
        pickle.dump(vector_list, f)

# Load a model from filepath
def load_model(filepath):
	model = gensim.models.Doc2Vec.load(filepath)
	return(model)

def doc2vec(model, sent1, sent2):
	vector_q1 = np.array(model.infer_vector(sent1))
	vector_q2 = np.array(model.infer_vector(sent2))
	
	return np.concatenate((vector_q1,vector_q2))

# Create a new model with the dataset and save it to disk
def create_model(data, filename):
	# Turn data into corpus
	training_corpus = get_corpus(data)
	# Tag the training corpus
	tagged_training_corpus = list(read_corpus(training_corpus))

	# Train the model and save it to disk
	model = doc2vec_model(tagged_training_corpus)
	model.save(filename)

# Main function
if __name__ == "__main__":

	# Read in the data
	training_corpus = read_data()
	tagged_training_corpus = list(read_corpus(training_corpus))

	# If a model already exists, we just need to load it.
	try:
		model = gensim.models.Doc2Vec.load("models/doc2vec.model")
		print("Found an existing model.")
	except:
		print("Could not find an existing model. Creating a new one. (Saved as doc2vec.model)")
		model = doc2vec_model(tagged_training_corpus)
		model.save("models/doc2vec.model")
