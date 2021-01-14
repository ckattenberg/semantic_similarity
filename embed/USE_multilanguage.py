import tensorflow_hub as hub
from scipy import spatial
import tensorflow_text

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

def encode(sentence):
    return embed([sentence])[0]

def cossim(arg1, arg2):
    return 1 - spatial.distance.cosine(arg1, arg2)

if __name__ == "__main__":
    print(cossim(encode("hello"), encode("hi")))

    
    
