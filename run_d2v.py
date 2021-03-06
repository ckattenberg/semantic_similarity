
from process import preprocess
from process import readdata
from embed import doc2vec as d2v
import os

if __name__ == "__main__":	

	# If a model already exists, we just need to load it.
	try:
		model = d2v.load_model("models/d2v/doc2vec.model")
		print("A d2v embedding model already exists.")
	except:
		print("Could not find an existing model. Creating one. \n (Will be saved in models/d2v/doc2vec.model)")
		print("Reading in data")
		data = readdata.read()
		print("Cleaning data")
		training_data = preprocess.clean_process(data)

		# Create a /models folder if it does not exist yet.
		if not os.path.exists('models'):
			os.makedirs('models')

		if not os.path.exists('models/d2v'):
			os.makedirs('models/d2v')

		d2v.create_model(training_data, "models/d2v/doc2vec.model")
		print('doc2vec model saved')
		
