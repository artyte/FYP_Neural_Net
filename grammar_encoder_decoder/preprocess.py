from bs4 import BeautifulSoup as bs
import sys
import codecs
reload(sys)
sys.setdefaultencoding('utf8')

def pickle_dump(filename, data):
	import pickle
	f = open(filename, 'w')
	pickle.dump(data, f)
	f.close()

def efficient_nucle():
	sys.setrecursionlimit(10000000)
	soup = bs(open("nucle3.2.sgml"), "lxml")
	pickle_dump('nucle3.2.p', soup)

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

def edit_paragraphs(soup):
	import re

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

	'''
	tmp = []
	for sen in senlis:
		tmp.append(len(sen[0].split()))
	import matplotlib.pyplot as plt
	plt.hist(tmp, 100)
	plt.show()
	'''

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
			word = word.lower()
			if word not in word_dict: sentence_tmp.append(0)
			else: sentence_tmp.append(word_dict[word][1])
		sentences_tmp.append(sentence_tmp)

	pickle_dump(filename, sentences_tmp)

def produce_data_files(train_input, train_output, test_input, test_output):
	import numpy as np

	vocab = {}
	reverse_vocab = {}
	f = open('glove.6B.200d.txt', 'r')
	vocab['##NULL##'] = [0,0]
	reverse_vocab[0] = '##NULL##'

	index = 1
	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		vocab[word] = [coefs, index]
		reverse_vocab[index] = word
		index += 1
	f.close()

	embedding_matrix = np.zeros((len(vocab), 200))
	for word, index in vocab.items():
		embedding_matrix[index[1]] = index[0]

	'''
	print embedding_matrix.shape[0]
	print embedding_matrix.shape[1]
	return
	'''

	np.save(open("embeds.npy", 'wb'), embedding_matrix)
	#pickle_dump('embeds.p', embedding_matrix)
	index_all(train_input, vocab, 'training_input_vectors.p')
	index_all(train_output, vocab, 'training_output_vectors.p')
	index_all(test_input, vocab, 'testing_input_vectors.p')
	index_all(test_output, vocab, 'testing_output_vectors.p')
	pickle_dump('index.p', vocab)
	pickle_dump('reverse_index.p', reverse_vocab)

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

#efficient_nucle()
#make_csv()
prepare_input()
