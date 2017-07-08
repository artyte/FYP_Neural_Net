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

def index_all(sentences, word_dict):
	sentences_tmp = []
	for sentence in sentences:
		sentence_tmp = []
		for word in sentence:
			word = word.lower()
			if word not in word_dict: sentence_tmp.append(0)
			else: sentence_tmp.append(word_dict[word][1])
		sentences_tmp.append(sentence_tmp)

	return sentences_tmp

def get_unique(train_input, train_output, test_input, test_output, threshold=5):
	a = train_input + train_output + test_input + test_output
	dic = {}
	for lis in a:
	    for element in lis:
	        element = element.lower()
	        if element in dic: dic[element] += 1
	        else: dic[element] = 1

	'''import matplotlib.pyplot as plt
	tmp = [dic[key] for key in dic.keys()]
	plt.hist(tmp, 10000, facecolor='red')
	plt.show()'''


	for key in dic.keys():
		if dic[key] <= threshold: del dic[key]
	dic = [list(elem) for elem in dic.items()]

	from operator import itemgetter
	dic = sorted(dic, key=itemgetter(0))
	dic = sorted(dic, key=itemgetter(1), reverse=True)
	lis = dic

	dic = {}
	reverse_dic = {}
	index = 1
	for array in lis:
	    dic[array[0]] = [array[1], index]
	    reverse_dic[index] = array[0]
	    index += 1

	return dic, reverse_dic

def trim(vocab, reverse_vocab):
	for item in vocab.keys():
	    if type(vocab[item][0]) == int:
			del reverse_vocab[vocab[item][1]]
			del vocab[item]

	vocab = [[key, vocab[key][0], vocab[key][1]] for key in vocab.keys()]
	reverse_vocab = [[key, reverse_vocab[key]] for key in reverse_vocab.keys()]

	from operator import itemgetter
	vocab = sorted(vocab, key=itemgetter(2))
	reverse_vocab = sorted(reverse_vocab, key=itemgetter(0))

	new_vocab = {}
	new_reverse_vocab = {}
	index = 1
	for array in vocab:
		new_vocab[array[0]] = [array[1], index]
		index += 1

	index = 0
	for array in reverse_vocab:
		new_reverse_vocab[index] = array[1]
		index += 1

	return new_vocab, new_reverse_vocab

def produce_data_files(train_input, train_output, test_input, test_output):

	vocab, reverse_vocab = get_unique(train_input, train_output, test_input, test_output)
	reverse_vocab[0] = '#null#'

	import numpy as np
	f = open('glove.6B.200d.txt', 'r')
	for line in f:
		values = line.split()
		word = values[0].lower()
		if word not in vocab: continue
		coefs = np.asarray(values[1:], dtype='float32')
		vocab[word] = [coefs, vocab[word][1]]
	f.close()

	vocab, reverse_vocab = trim(vocab, reverse_vocab) # some unique tokens don't have an embeding
	print len(vocab)+1

	embedding_matrix = np.zeros((len(vocab)+1, 200))
	for word, array in vocab.items(): embedding_matrix[array[1]] = array[0]

	'''
	print embedding_matrix.shape[0]
	print embedding_matrix.shape[1]
	return
	'''

	np.save(open("embeds.npy", 'wb'), embedding_matrix)
	x = index_all(train_input, vocab)
	y = index_all(train_output, vocab)
	pickle_dump('training_vectors.p', zip(x,y))
	x = index_all(test_input, vocab)
	y = index_all(test_output, vocab)
	pickle_dump('testing_vectors.p', zip(x,y))
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
	threshold = int(row_count * 0.8)
	for index, row in enumerate(read):
		if(index < threshold): train_data.append(row)
		else: test_data.append(row)
	f.close()

	train_input, train_output = tokenize_all(train_data)
	test_input, test_output = tokenize_all(test_data)
	produce_data_files(train_input, train_output, test_input, test_output)

#efficient_nucle()
#make_csv()
prepare_input()
