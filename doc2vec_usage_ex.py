import doc2vec

model = doc2vec.load_model("doc2vec.model")
embedding = doc2vec.doc2vec(model, ["The quick brown fox jumps over the lazy dog."])

print(embedding)