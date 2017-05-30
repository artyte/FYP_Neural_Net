from bs4 import BeautifulSoup as bs
import sys
import codecs
reload(sys)
sys.setdefaultencoding('utf8')

def calculate_shift(shift_influ, word_len, start_index, end_index):
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

def efficient_nucle():
	import pickle
	sys.setrecursionlimit(10000000)
	soup = bs(open("nucle3.2.sgml"), "lxml")
	f = open('nucle3.2.p', 'w')
	pickle.dump(soup, f)
	f.close()

def edit_paragraphs(soup):
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
		shift = calculate_shift(shift_influ, len(m[0]), m[2], m[3])
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

def make_csv():
	import csv
	import pickle
	f = open('nucle3.2.p', 'r')
	soup = pickle.load(f)
	f.close()

	senlis = []
	for doc in soup.findChildren('doc'): senlis += edit_paragraphs(doc)

	f = open('nucle3.2.csv', 'w')
	write = csv.writer(f)
	for row in senlis: write.writerow(row)
	f.close()

def tokenize_all(data):
	from nltk.tokenize import word_tokenize as wt

	original = []
	edited = []
	for sentence in data:
		original.append(wt(sentence[0]))
		edited.append(wt(sentence[1]))

	return original, edited

def index_all(sentences, word_dict, filename):
	sentences_tmp = []
	for sentence in sentences:
		sentence_tmp = []
		for word in sentence:
			sentence_tmp.append(int(word_dict[word]))
		sentences_tmp.append(sentence_tmp)

	import pickle
	f = open(filename, 'w')
	pickle.dump(sentences_tmp, f)
	f.close()

def produce_data_files(train_input, train_output, test_input, test_output):
	from gensim.models import Word2Vec as w2v
	import multiprocessing as mp
	import numpy as np
	import pickle

	model = w2v(train_input + train_output + test_input + test_output, sg = 1, seed = 1,
		workers = mp.cpu_count(), size = 300, min_count = 0, window = 7, iter = 4)
	weights = model.syn0
	np.save(open("embeds.npy", 'wb'), weights)
	vocab = dict([(k, v.index) for k, v in model.vocab.items()])
	index_all(train_input, vocab, 'training_input_vectors.p')
	index_all(train_output, vocab, 'training_output_vectors.p')
	index_all(test_input, vocab, 'testing_input_vectors.p')
	index_all(test_output, vocab, 'testing_output_vectors.p')
	vocab = dict([(v.index, k) for k, v in model.vocab.items()])
	f = open("reverse_index.p", 'w')
	pickle.dump(vocab, f)
	f.close()

def prepare_input():
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

	train_input, train_output = tokenize_all(train_data)
	test_input, test_output = tokenize_all(test_data)
	produce_data_files(train_input, train_output, test_input, test_output)

prepare_input()
