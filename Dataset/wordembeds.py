from wikihistory import WikiHistory as wh
from nltk.tokenize import word_tokenize as wt
from gensim.models import Word2Vec as w2v
import multiprocessing
import numpy as np
import json
'''
import os
import sklearn.manifold
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
'''

class WordEmbeds:

	@staticmethod
	def __init__():
		wh() # initialize dataset
		formatTest()
		formatDevelopment()
		formatTrain()
	
	@staticmethod
	def formatTest():
		# convert testing data into list of lists
		sentences = []
		senlis = wh.getFeatures("TESTING_DATA")
		for sentence in senlis:
			words = wt(sentence)
			sentences.append(words)

		model = w2v(sentences,
							sg = 1,
							seed = 1,
							workers = multiprocessing.cpu_count(),
							size = 300,
							min_count = 3,
							window = 7,
							sample = 1e-3,
							iter = 6)

		weights = model.syn0
		np.save(open("embeds.npy", 'wb'), weights)
		
		vocab = dict([(k, v.index) for k, v in model.vocab.items()])
		with open("TESTING_DATA_VOCAB.txt", 'w') as f:
			f.write(json.dumps(vocab))
		
		reverse_vocab = dict([v.index, k) for k, v in model.vocab.items()])
		with open("TESTING_DATA_REVERSE_VOCAB.txt", 'w') as f:
			f.write(json.dumps(reverse_vocab))
			
	@staticmethod
	def formatDevelopment():
		# convert testing data into list of lists
		sentences = []
		senlis = wh.getFeatures("DEVELOPMENT_DATA")
		for sentence in senlis:
			words = wt(sentence)
			sentences.append(words)

		model = w2v(sentences,
							sg = 1,
							seed = 1,
							workers = multiprocessing.cpu_count(),
							size = 300,
							min_count = 3,
							window = 7,
							sample = 1e-3,
							iter = 6)
		
		vocab = dict([(k, v.index) for k, v in model.vocab.items()])
		with open("DEVELOPMENT_DATA_VOCAB.txt", 'w') as f:
			f.write(json.dumps(vocab))
		
		reverse_vocab = dict([v.index, k) for k, v in model.vocab.items()])
		with open("DEVELOPMENT_DATA_REVERSE_VOCAB.txt", 'w') as f:
			f.write(json.dumps(reverse_vocab))
		
	@staticmethod
	def formatTrain():
		# convert testing data into list of lists
		sentences = []
		senlis = wh.getFeatures("DEVELOPMENT_DATA")
		for sentence in senlis:
			words = wt(sentence)
			sentences.append(words)

		model = w2v(sentences,
							sg = 1,
							seed = 1,
							workers = multiprocessing.cpu_count(),
							size = 300,
							min_count = 3,
							window = 7,
							sample = 1e-3,
							iter = 6)
		
		vocab = dict([(k, v.index) for k, v in model.vocab.items()])
		with open("TRAINING_DATA_VOCAB.txt", 'w') as f:
			f.write(json.dumps(vocab))
		
		reverse_vocab = dict([v.index, k) for k, v in model.vocab.items()])
		with open("TRAINING_DATA_REVERSE_VOCAB.txt", 'w') as f:
			f.write(json.dumps(reverse_vocab))

'''
tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
weights_2d = tsne.fit_transform(weights)

points = pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in [
            (word, weights_2d[model.vocab[word].index])
            for word in model.vocab
        ]
    ],
    columns=["word", "x", "y"]
)
sns.set_context("poster")
points.plot.scatter("x", "y", s=10, figsize=(20, 12))
plt.show()
'''
