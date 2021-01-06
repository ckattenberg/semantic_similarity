from readdata import read
from preprocess import process, clean_process
from sklearn.metrics import accuracy_score, precision_score, recall_score
import spacy

if __name__ == "__main__":
    # Om te kunnen runnen, run eerst deze command:
    # python -m spacy download en_core_web_md

    data = read()[:5000]
    correct = list(data.is_duplicate)[:5000]
    result = []
    nlp = spacy.load('en_core_web_md')

    for id in data['id']:
        text1 = nlp(data['question1'][id])
        text2 = nlp(data['question2'][id])
        score = text1.similarity(text2)

        if score > 0.96:
            result.append(1)
        else:
            result.append(0)
  
    print('Accuracy: ', accuracy_score(correct, result))
    print('Precision: ', precision_score(correct, result))
    print('Recall: ', recall_score(correct, result))
        