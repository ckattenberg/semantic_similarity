from readdata import read
from preprocess import process, clean_process
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score

def cosine_sim(text1, text2):
   return ((tfidf*tfidf.T).A)[0,1]

vectorizer = TfidfVectorizer(min_df=1, stop_words='english')   

if __name__ == "__main__":
   data = read()[:5000]
   correct = list(data.is_duplicate)[:5000]
   result = []

   # Put all the sentences in the dataset in a large corpus list.
   corpus = list(data['question1']) + list(data['question2'])

   X = vectorizer.fit(corpus)

   for id in data['id']:
      text1 = data['question1'][id]
      text2 = data['question2'][id]
      tfidf = X.transform([text1, text2])
      score = cosine_sim(tfidf)

      # Threshold
      if score > 0.85:
         result.append(1)
      else:
         result.append(0)
   
print('Accuracy: ', accuracy_score(correct, result))
print('Precision: ', precision_score(correct, result))
print('Recall: ', recall_score(correct, result))

