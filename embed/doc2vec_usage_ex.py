from . import doc2vec

# Read in the data
data = readdata.read()
training_data = preprocess.clean_process(data)
doc2vec.create_model(training_data, "models/doc2vec.model")
model = doc2vec.load_model("models/doc2vec.model")
embedding = doc2vec.doc2vec(model, ["The quick brown fox jumps over the lazy dog."])
embedding2 = doc2vec.doc2vec(model, ["The quick brown fox jumps over the lazy dog."])

print(embedding)
