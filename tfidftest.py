import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer
from readdata import read
from preprocess import process, clean_process, clean_split, untokenized_split
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity


stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

def stem_tokens(tokens):
    return [stemmer.stem(item) for item in tokens]

'''remove punctuation, lowercase, stem'''
def normalize(text):
    return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

vectorizer = TfidfVectorizer(tokenizer=normalize)

def cosine_sim(text1, text2):
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0,1]

if __name__ == "__main__":
    data = untokenized_split(read())
    # train = data[0]
    test = data[1]
    correct = list(test['is_duplicate'])
    result = []


    # Predict similarities of test set
    for id in test['id']:
        text1 = test['question1'][id]
        text2 = test['question2'][id]
        
        score = cosine_sim(text1, text2)

        # Threshold
        if score > 0.85:
            result.append(1)
        else:
            result.append(0)

    # Compare predictions of test set with actual correct answers
    print('Accuracy: ', accuracy_score(correct, result))
    print('Precision: ', precision_score(correct, result))
    print('Recall: ', recall_score(correct, result))
   
