import sys
import codecs
reload(sys)
sys.setdefaultencoding('utf8')

def rmv_particle(sentence, choice, tokens, corrective_set):
    import random
    choice = random.choice(choice)
    random.shuffle(tokens[choice]) # ensure that a batch of tokens aren't always read in the same order
    token = tokens[choice]

    from nltk.tokenize import word_tokenize as wt
    sentence = wt(sentence)
    for word in sentence:
        tmp = word.lower()
        if tmp in token:
            if tmp not in corrective_set: corrective_set.append(tmp)
            sentence.remove(word)
            break

    return " ".join(sentence)

def get_tokens():
    choice = []
    tokens = []
    with open('tokens.txt') as f:
        i = 0
        for line in f:
            import re
            tmp = line.split(' ')
            tmp[-1] = re.sub('\n', '', tmp[-1]) # because last token always contain return carraige
            tokens.append(tmp)
            choice.append(i)
            i += 1

    return choice, tokens

def make_csv(value=1.0, choice=1):
    original = []
    edited = []
    threshold = int(len(open('entries.train').readlines())) * value if choice == 1 else value
    with open('entries.train') as f:
        for index, line in enumerate(f):
            if index < threshold:
                if len(line) == 1: continue
                line = line.split('\t')
                if int(line[0]) == 0: edited.append(line[4])
                else: edited.append(line[5])

    import re
    from numpy.random import uniform as rand
    corrective_set = [] # removed tokens used in perturbation of sentences
    choice, tokens = get_tokens()
    for i, sentence in enumerate(edited):
        edited[i] = re.sub(' +', ' ', sentence)
        edited[i] = re.sub('\n', '', edited[i])
        if rand(0,1) <= 0.75: original.append(rmv_particle(edited[i], choice, tokens, corrective_set))
        else: original.append(edited[i])

    pickle_dump('corrective_set.p', corrective_set)

    import csv
    senlis = zip(original,edited)
    f = open('nucle3.2_lang.csv', 'w')
    write = csv.writer(f)
    for row in senlis: write.writerow(row)
    f.close()

    # number of sentences vs number of words
    '''
    tmp = []
    for sen in senlis: tmp.append(len(sen[0].split()))
    import matplotlib.pyplot as plt
    plt.hist(tmp, 100)
    plt.show()'''

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
	'''
    import matplotlib.pyplot as plt
	tmp = [dic[key] for key in dic.keys()]
	plt.hist(tmp, 450, facecolor='red')
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
	f = open('glove.6B.100d.txt', 'r')
	for line in f:
		values = line.split()
		word = values[0].lower()
		if word not in vocab: continue
		coefs = np.asarray(values[1:], dtype='float32')
		vocab[word] = [coefs, vocab[word][1]]
	f.close()

	vocab, reverse_vocab = trim(vocab, reverse_vocab) # some unique tokens don't have an embeding
	pickle_dump('output_size.p', (len(vocab)+1))
	print len(vocab) + 1

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
	pickle_dump('indexed_corrective_set.p', index_all([pickle_return('corrective_set.p')], vocab)[0])

def pickle_return(filename):
	import pickle
	f = open(filename, 'r')
	data = pickle.load(f)
	f.close()
	return data

def pickle_dump(filename, data):
	import pickle
	f = open(filename, 'w')
	pickle.dump(data, f)
	f.close()

def prepare_input():
	import csv
	f = open('nucle3.2_lang.csv', 'r')
	read = csv.reader(f)

	train_data = []
	test_data = []
	#threshold = int(row_count * value) if choice == 1 else value
	for row in read: train_data.append(row)
	f.close()
	train_input, train_output = tokenize_all(train_data)
	test_input, test_output = tokenize_all(test_data)
	produce_data_files(train_input, train_output, test_input, test_output)

choice = int(raw_input("Enter an option (%s), (%s), (%s): " % ("1. Process CSV", "2. Prepare input", "3. Both")))
if choice == 1:
    choice = int(raw_input("Sample by Proportion (1)/ Absolute (Any other number): "))
    value = float(raw_input("Type the value: "))
    make_csv(value, choice)
elif choice == 2:
    prepare_input()
elif choice == 3:
    choice = int(raw_input("Sample by Proportion (1)/ Absolute (Any other number): "))
    value = float(raw_input("Type the value: "))
    make_csv(value, choice)
    prepare_input()
