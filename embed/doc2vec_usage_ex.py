from . import doc2vec

# Read in the data
data = readdata.read()
training_data = preprocess.clean_process(data)
training_corpus = doc2vec.get_corpus(training_data)
doc2vec.create_model(training_corpus, "doc2vec.model")
model = doc2vec.load_model("doc2vec.model")
embedding = doc2vec.doc2vec(model, ["The quick brown fox jumps over the lazy dog."])

print(embedding)
