from readdata import read
from preprocess import process, clean_process, untokenized_split, clean_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import spacy

if __name__ == "__main__":
    # Om te kunnen runnen, run eerst deze command:
    # python -m spacy download en_core_web_md

    data = untokenized_split(read())
    train = data[0]
    test = data[1]
    correct = list(test['is_duplicate'])
    result = []

    nlp = spacy.load('en_core_web_md')

    for id in test['id']:
        text1 = nlp(test['question1'][id])
        text2 = nlp(test['question2'][id])

        score = text1.similarity(text2)

        if score > 0.96:
            result.append(1)
        else:
            result.append(0)
  
    print('Accuracy: ', accuracy_score(correct, result))
    print('Precision: ', precision_score(correct, result))
    print('Recall: ', recall_score(correct, result))
        