# Semantic and Intentional Textual Similarity

This is a Python library for dealing with semantic similarity tasks. It is being developed by students from the University of Amsterdam (UvA) for Open Online Research (https://openonlineresearch.com/) as part of the "Leren en Beslissen" course.

## Usage

To train a model on the data:
```bash
$ python run_METHOD.py
```
Where METHOD is the name of the embedding technique you want to use. 

- Word2Vec: w2v  
- Doc2Vec: d2v  
- Universal Sentence Encoder: use

To test the model after it has been trained:
```bash
$ python run_test_METHOD.py
```

#### Specific Example
```bash
$ python run_w2v.py
$ python run_test_w2v.py
```
Results of a test run are automatically saved in the results/ folder.

## Dependencies 

Packages required to use this library:
- gensim 3.8.3
- keras 2.4.3
- nltk 3.5
- numpy 1.19.4
- pandas 1.2.0
- scipy 1.6.0
- sklearn 0.23.2
- tensorflow 2.4.0
- tensorflow_hub

## Authors

Max Waser  
Ananda Pradhan  
Stefan Dijkstra  
Céline Kattenberg
