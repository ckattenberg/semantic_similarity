import gensim
from scipy.spatial import distance
import preprocess
import readdata

def load_model(filepath):
	model = gensim.models.Doc2Vec.load(filepath)
	return(model)

def read_data():
	data = readdata.read()
	train, test = preprocess.clean_split(data)
	test1 = list(test['question1'])
	test2 = list(test['question2'])
	true_duplicate = list(test['is_duplicate'])
	return(test1, test2, true_duplicate)

if __name__ == "__main__":
	# Get vector from sentence
	model = load_model("doc2vec.model")

	test1, test2, true_duplicate = read_data()

	print(len(test1), len(test2), len(true_duplicate))


	print(data.shape)
	print(clean_data.shape)

	# For each pair of test sentences
	for i in len(test1):
		model_duplicate = []
		# Get the vectors of both sentences
		q1 = test1[i]
		q2 = test2[i]

		vec1 = model.infer_vector(q1)
		vec2 = model.infer_vector(q2)

		# Compare their cosine similarity
		dist = distance.cosine(vec1, vec2)
		# Put into list
		model_duplicate.append(dist)


print(true_duplicate, model_duplicate)