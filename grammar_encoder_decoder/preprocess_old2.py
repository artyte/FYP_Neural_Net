import sys
import codecs
reload(sys)
sys.setdefaultencoding('utf8')

def make_csv():
    f = open('official-preprocessed.txt', 'r')
    lines = f.readlines()

    sentence_index = 0
    shift_accu = 0
    prev_index = 0
    sentences = []
    edited = []
    for line in lines:
        if len(line) > 1: label, sentence = nucle_splitter(line)
        else:
            s = edited[sentence_index]
            edited[sentence_index] = " ".join(s)
            sentence_index += 1
            shift_accu = 0
            prev_index = 0
            continue

        if label == 'S':
            sentences.append(sentence)
            edited.append(sentence.split(" "))

        if label == 'A':
            start_index, end_index, edit = correction_splitter(sentence)
            s = edited[sentence_index] # sentence to edit
            s, shift_accu, prev_index = shift_edit(s, edit, shift_accu, start_index, end_index, prev_index)
            edited[sentence_index] = s

    import re
    for i in range(len(edited)): edited[i] = re.sub(' +', ' ', edited[i])

    import csv
    senlis = zip(sentences, edited)
    f = open('nucle3.2_new.csv', 'w')
    write = csv.writer(f)
    for row in senlis: write.writerow(row)
    f.close()

    # number of sentences vs number of words
    tmp = []
    for sen in senlis: tmp.append(len(sen[0].split()))
    import matplotlib.pyplot as plt
    plt.hist(tmp, 100)
    plt.show()

def shift_edit(s, edit, shift, start_index, end_index, prev_index):
    if prev_index > start_index: start_index = prev_index
    s = s[:shift + start_index] + [edit] + s[shift + end_index:]
    shift += start_index - end_index + 1
    return s, shift, end_index

def correction_splitter(correction):
    correction = correction.split('|||')
    indexes = correction[0].split(" ")

    start_index = int(indexes[0])
    end_index = int(indexes[1])
    edit = correction[2]

    return start_index, end_index, edit

def nucle_splitter(line):
    label = line[0]
    sentence = line[2:]
    return label, sentence

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

def get_unique(train_input, train_output, test_input, test_output, threshold=0):
	a = train_input + train_output + test_input + test_output
	dic = {}
	for lis in a:
	    for element in lis:
	        element = element.lower()
	        if element in dic: dic[element] += 1
	        else: dic[element] = 1

    # number of words with that frequency vs frequency of words
	import matplotlib.pyplot as plt
	tmp = [dic[key] for key in dic.keys()]
	plt.hist(tmp, 500, facecolor='red')
	plt.show()


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
	f = open('glove.6B.100d.txt', 'r')
	for line in f:
		values = line.split()
		word = values[0].lower()
		if word not in vocab: continue
		coefs = np.asarray(values[1:], dtype='float32')
		vocab[word] = [coefs, vocab[word][1]]
	f.close()

	vocab, reverse_vocab = trim(vocab, reverse_vocab) # some unique tokens don't have an embeding
	print len(vocab)+1

	embedding_matrix = np.zeros((len(vocab)+1, 100))
	for word, array in vocab.items(): embedding_matrix[array[1]] = array[0]

	np.save(open("embeds.npy", 'wb'), embedding_matrix)
	x = index_all(train_input, vocab)
	y = index_all(train_output, vocab)
	pickle_dump('training_vectors.p', zip(x,y))
	x = index_all(test_input, vocab)
	y = index_all(test_output, vocab)
	pickle_dump('testing_vectors.p', zip(x,y))
	pickle_dump('index.p', vocab)
	pickle_dump('reverse_index.p', reverse_vocab)

def pickle_dump(filename, data):
	import pickle
	f = open(filename, 'w')
	pickle.dump(data, f)
	f.close()

def prepare_input():
	import csv
	f = open('nucle3.2_new.csv', 'r')
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

make_csv()
prepare_input()
