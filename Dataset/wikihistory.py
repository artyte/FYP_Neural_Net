import csv
import json
import multiprocessing
import numpy as np
from nltk.tokenize import word_tokenize as wt
from gensim.models import Word2Vec as w2v

class WikiHistory:

	testData = []
	developData = []
	trainData = []
	model = 0

	@staticmethod
	def __init__():
		with open("wiki_history_write.csv", encoding = "utf8") as f0:
			read = csv.reader(f0)
			row_count = sum(1 for row in read)
			f0.seek(0)	#reset readline to start for other reading purposes
			WikiHistory.prepareTrainingData(read, row_count)
			WikiHistory.prepareDevelopmentData(read, row_count)
			WikiHistory.prepareTestData(read, row_count)
		
		WikiHistory.produceWeights()
		convertTrainingData(WikiHistory.formatTest())
		convertDevelopmentData(WikiHistory.formatDevelopment())
		convertTestData(WikiHistory.formatTrain())
			
	@staticmethod
	def convertTrainingData(sentences):
		with open("TESTING_DATA_VOCAB.txt", encoding = "utf8") as f:
			word_dict = json.load(f)
			sentences_tmp = []
			for sentence in sentences:
				sentence_tmp = []
				for word in sentence:
					sentence_tmp.append(int(word_dict[word]))
				sentences_tmp.append(sentence_tmp)
			testData = np.array(sentences_tmp)
	
	@staticmethod
	def convertDevelopmentData(sentences):
		with open("DEVELOPMENT_DATA_VOCAB.txt", encoding = "utf8") as f:
			word_dict = json.load(f)
			sentences_tmp = []
			for sentence in sentences:
				sentence_tmp = []
				for word in sentence:
					sentence_tmp.append(int(word_dict[word]))
				sentences_tmp.append(sentence_tmp)
			developData = np.array(sentences_tmp)
	
	@staticmethod
	def convertTestData(sentences):
		with open("TRAINING_DATA_VOCAB.txt", encoding = "utf8") as f:
			word_dict = json.load(f)
			sentences_tmp = []
			for sentence in sentences:
				sentence_tmp = []
				for word in sentence:
					sentence_tmp.append(int(word_dict[word]))
				sentences_tmp.append(sentence_tmp)
			trainData = np.array(sentences_tmp)
			

	@staticmethod
	def prepareTrainingData(input, row_count):
		for index, row in enumerate(input):
			if (index < int(row_count * 0.8)):
				WikiHistory.trainData.append([row[2], row[3]])
			else:
				WikiHistory.developData.append([row[2], row[3]])
				break

	@staticmethod
	def prepareDevelopmentData(input, row_count):
		for index, row in enumerate(input):
			if (index < int(row_count * 0.1) - 1):
				WikiHistory.developData.append([row[2], row[3]])
			else:
				WikiHistory.testData.append([row[2], row[3]])
				break

	@staticmethod
	def prepareTestData(input, row_count):
		for index, row in enumerate(input):
			WikiHistory.testData.append([row[2], row[3]])

	@staticmethod
	def getLabels(string):
		lis = []
		if (string == "TESTING_DATA"):
			for row in WikiHistory.testData:
				lis.append(row[1])
			return lis
		elif (string == "DEVELOPMENT_DATA"):
			for row in WikiHistory.developData:
				lis.append(row[1])
			return lis
		elif (string == "TRAINING_DATA"):
			for row in WikiHistory.trainData:
				lis.append(row[1])
			return lis

	@staticmethod
	def getFeatures(string):
		lis = []
		if (string == "TESTING_DATA"):
			for row in WikiHistory.testData:
				lis.append(row[0])
			return lis
		elif (string == "DEVELOPMENT_DATA"):
			for row in WikiHistory.developData:
				lis.append(row[0])
			return lis
		elif (string == "TRAINING_DATA"):
			for row in WikiHistory.trainData:
				lis.append(row[0])
			return lis
		elif (string == "EVERYTHING"):
			for row in WikiHistory.testData:
				lis.append(row[0])
			for row in WikiHistory.developData:
				lis.append(row[0])
			for row in WikiHistory.trainData:
				lis.append(row[0])
			return lis
			
	@staticmethod
	def getList(string):
		lis = []
		if (string == "TESTING_DATA"):
			return WikiHistory.testData
		elif (string == "DEVELOPMENT_DATA"):
			return WikiHistory.developData
		elif (string == "TRAINING_DATA"):
			return WikiHistory.trainData
			
	@staticmethod
	def produceWeights():
	# convert testing data into list of lists
		sentences = []
		senlis = WikiHistory.getFeatures("EVERYTHING")
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
	
	@staticmethod
	def formatTest():
		# convert testing data into list of lists
		sentences = []
		senlis = WikiHistory.getFeatures("TESTING_DATA")
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
		with open("TESTING_DATA_VOCAB.json", 'w') as f:
			f.write(json.dumps(vocab))
			
		return sentences
		
		'''
		reverse_vocab = dict([v.index, k) for k, v in model.vocab.items()])
		with open("TESTING_DATA_REVERSE_VOCAB.txt", 'w') as f:
			f.write(json.dumps(reverse_vocab))
		'''
			
	@staticmethod
	def formatDevelopment():
		# convert testing data into list of lists
		sentences = []
		senlis = WikiHistory.getFeatures("DEVELOPMENT_DATA")
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
		with open("DEVELOPMENT_DATA_VOCAB.json", 'w') as f:
			f.write(json.dumps(vocab))
			
		return sentences
		
		'''
		reverse_vocab = dict([v.index, k) for k, v in model.vocab.items()])
		with open("DEVELOPMENT_DATA_REVERSE_VOCAB.txt", 'w') as f:
			f.write(json.dumps(reverse_vocab))
		'''
		
	@staticmethod
	def formatTrain():
		# convert testing data into list of lists
		sentences = []
		senlis = WikiHistory.getFeatures("TRAINING_DATA")
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
		with open("TRAINING_DATA_VOCAB.json", 'w') as f:
			f.write(json.dumps(vocab))
			
		return sentences
