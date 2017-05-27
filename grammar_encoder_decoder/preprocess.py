from bs4 import BeautifulSoup as bs
import sys
import codecs
reload(sys)
sys.setdefaultencoding('utf8')

def calculateShift(shift_influ, word_len, start_index, end_index):
	shift = word_len - (end_index - start_index)
	accu = 0
	i = 0
	for obj in shift_influ:
		if start_index < obj[0]:
			shift_influ.insert(i, [start_index, shift])
			break
		else:
			i += 1
			accu += obj[1]

	if i == len(shift_influ): shift_influ.insert(i, [start_index, shift])
	return accu + 1

def efficientNucle():
	import pickle
	sys.setrecursionlimit(10000000)
	soup = bs(open("nucle3.2.sgml"), "lxml")
	f = open('nucle3.2.p', 'w')
	pickle.dump(soup, f)
	f.close()

def editParagraphs(soup):
	import nltk
	import re
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

	paragraphs = []
	original = []
	for p in soup.textword.findChildren('p'): paragraphs += p
	for p in paragraphs:
		p = re.sub('\n', '', p)
		original.append(str(p))

	mistakelis = []
	for mistake in soup.annotation.findChildren('mistake'):
		mistakelis.append([mistake.correction.text, int(mistake['start_par']) - 1,
						int(mistake['start_off']), int(mistake['end_off'])])

	shift_influ = []
	prev_par = 0
	for m in mistakelis:
		if m[1] != prev_par: shift_influ = []
		prev_par = m[1]
		p = paragraphs[m[1]]
		shift = calculateShift(shift_influ, len(m[0]), m[2], m[3])
		p = p[:shift + m[2]] + m[0] + p[shift + m[3]:]
		paragraphs[m[1]] = p

	sentences = []
	for p in paragraphs:
		p = re.sub(' +', ' ', p)
		p = re.sub(' \.', '.', p)
		p = re.sub('\n', '', p)
		sentences.append(str(p))

	zipped = zip(original, sentences)
	return zipped

def makeCSV():
	import csv
	import pickle
	f = open('nucle3.2.p', 'r')
	soup = pickle.load(f)
	f.close()

	senlis = []
	for doc in soup.findChildren('doc'): senlis += editParagraphs(doc)

	f = open('nucle3.2.csv', 'w')
	write = csv.writer(f)
	for row in senlis: write.writerow(row)
	f.close()

def tokenizeAll(data):
	from nltk.tokenize import word_tokenize as wt

	original = []
	edited = []
	for sentence in data:
		original.append(wt(sentence[0]))
		edited.append(wt(sentence[1]))

	return original, edited

def produceDataFiles(sentences):
	from gensim.models import Word2Vec as w2v
	import multiprocessing as mp
	import numpy as np
	import json

	model = w2v(sentences, sg = 1, seed = 1, workers = mp.cpu_count(),
				size = 500, min_count = 0, window = 7, iter = 6)
	weights = model.syn0
	np.save(open("embeds.npy", 'wb'), weights)
	vocab = dict([(k, v.index) for k, v in model.vocab.items()])
	reverse_vocab = dict([(v.index, k) for k, v in model.vocab.items()])
	for k, v in model.vocab.items():
		print v
		print v.index
	f = open("index.json", 'w')
	f.write(json.dumps(vocab))
	f.close()
	f = open("reverse_index.json", 'w')
	f.write(json.dumps(reverse_vocab))
	f.close()

def prepareInput():
	import csv
	f = open('nucle3.2.csv', 'r')
	read = csv.reader(f)
	row_count = sum(1 for row in read)
	f.seek(0)

	train_data = []
	test_data = []
	for index, row in enumerate(read):
		if(index < int(row_count * 0.8)): train_data.append(row)
		else: test_data.append(row)
	f.close()

	sentences_train_input, sentences_train_output = tokenizeAll(train_data)
	sentences_test_input, sentences_test_output = tokenizeAll(test_data)
	produceDataFiles(sentences_train_input + sentences_train_output + sentences_test_input + sentences_test_output)

prepareInput()
