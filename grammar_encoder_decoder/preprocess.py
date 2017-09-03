from os.path import join
from convenient_pickle import pickle_return

path = "data"
log_short = pickle_return(join(path, "log_short.p"))
log_long = pickle_return(join(path, "log_long.p"))

def perturb_and_label(correct, corrects, wrongs, labels, particles, sentence_mode):
	import re
	import random
	tokens = open(join(path, "particles.txt")).readlines()
	tokens = [re.sub('\n', '', token) for token in tokens]
	random.shuffle(tokens)

	# perturbation only possible at word level
	wrong = correct.split(" ")
	index = range(len(wrong))
	random.shuffle(index)
	for i in index:
		flag = False # used to break out of this loop
		for token in tokens:
			if wrong[i].lower() == token:
				if token not in particles: particles.append(token)
				wrong.remove(wrong[i]) # remove using token might not match case
				flag = True
				break

		if flag == True: break

	corrects, wrongs, labels = label(correct, corrects, " ".join(wrong), wrongs, labels, sentence_mode)

	return corrects, wrongs, labels, particles

def label(correct, corrects, wrong, wrongs, labels, sentence_mode):
	# split by character/word level
	correct = list(correct) if sentence_mode == "character" else correct.split(" ")
	wrong = list(wrong) if sentence_mode == "character" else wrong.split(" ")
	wrongs.append(wrong)
	corrects.append(correct)

	wrong2 = wrong[:] # make a copy of wrong so that wrongs won't be mutated by original wrong
	# labels: 0->inexistent 1->same 2->change
	for index, word in enumerate(correct):
		try:
			if word == wrong2[index]: wrong2[index] = 1
			else: wrong2.insert(index, 2)
		except IndexError:
			wrong2.insert(index, 2)
	labels.append(wrong2)

	return corrects, wrongs, labels


'''
Produce a clean list of lists of selected sentences with appropriate labels.
Labelling samples here allows for easy labelling.

label -> whether or not word in sentence is evaluated to the same or not'''
def prepare_first_data(sample_by, sample_val, sentence_mode):
	# load all lines from data for use
	lines = open(join(path, "entries.train")).readlines()

	# number of lines sampled
	threshold = int(len(lines)) * sample_val if sample_by == "proportion" else sample_val

	# grab ONLY correct sentences from samples, labels will be made later with sentence perturbation
	corrects = [] # to store correct sentences
	import re
	for index, line in enumerate(lines):
		line = re.sub('\n', '', line) # remove trailing \n in every sample
		if index < threshold:
			if len(line) == 0: continue
			line = line.split('\t')
			if int(line[0]) == 0: corrects.append(line[4])
			else: corrects.append(line[5])

	from numpy.random import uniform as rand
	rights = [] # to store splitted correct sentences
	wrongs = [] # to store perturbed sentences
	labels = []
	particles = [] # to store used particles for perturbation
	# perturb samples by deleting grammatical particles
	# also apply labels
	for correct in corrects:
		if rand(0,1) <= 0.75: rights, wrongs, labels, particles = perturb_and_label(correct, rights, wrongs, labels, particles, sentence_mode)
		else: rights, wrongs, labels = label(correct, rights, correct, wrongs, labels, sentence_mode)

	# add custom sentences
	lines = open(join(path, 'custom_sentences.txt')).readlines()
	for line in lines:
		line = re.sub('\n', '', line)
		rights, wrongs, labels, particles = perturb_and_label(line, rights, wrongs, labels, particles, sentence_mode)

	if log_long:
		for i in zip(rights, wrongs, labels):
			print i, "\n"
		print particles

	if log_short:
		# number of sentences vs number of words in that sentence
		import matplotlib.pyplot as plt
		plt.hist([len(right) for right in rights], 100)
		plt.show()

	return zip(wrongs, rights, labels), particles

'''
Since right have at least all the words in wrong, only right is assumed to be used
'''
def create_index_maps(data, threshold=5):
	index_map = {}
	for _, right, _ in data:
		for word in right:
			word = word.lower()
			if word not in index_map: index_map[word] = 1
			else: index_map[word] += 1

	if log_long:
		# number of words with that frequency vs frequency of words
		import matplotlib.pyplot as plt
		plt.hist([index_map[word] for word in index_map.keys()], 450, facecolor='red')
		plt.show()

	# remove words that appear below or equals to threshold times
	for word in index_map.keys():
		if index_map[word] <= threshold: del index_map[word]

	# sorting of words so as to give a proper index
	# result: descending order of occurences of words in ascending order of words
	# e.g. [(ab,1),(aa,1),(bb,3)] -> [(bb,3),(aa,1),(ab,1)]
	from operator import itemgetter
	index_map = [list(record) for record in index_map.items()]
	index_map = sorted(index_map, key=itemgetter(0)) # sort by word in ascending order
	index_map = sorted(index_map, key=itemgetter(1), reverse=True) # sort by word's number of occurence in descending order
	lis = index_map[:]

	# creating actual index maps
	index_map = {}
	reverse_index = {}
	index = 1 # index number to be given to a word, 0 reserved for unknown
	for array in lis:
		index_map[array[0]] = index
		reverse_index[index] = array[0]
		index += 1 # change to next index number

	return index_map, reverse_index

def index_words(index_map, sentence):
	for index, word in enumerate(sentence):
		word = word.lower()
		if word not in index_map: sentence[index] = 0
		else: sentence[index] = index_map[word]

	return sentence

def produce_data_files(*args):
	import re
	from convenient_pickle import pickle_dump
	filenames = open(join(path, "filenames.txt")).readlines()
	filenames = [re.sub('\n', '', filename) for filename in filenames]

	for index, filename in enumerate(filenames): pickle_dump(join(path, filename), args[index])

def prepare_input(data, particles):
	train_data = []
	train_label = []
	test_data = []
	test_label = []

	index_map, reverse_index = create_index_maps(data)

	import random
	from numpy.random import uniform as rand
	random.shuffle(data)
	for wrong, right, label in data:
		wrong = index_words(index_map, wrong)
		right = index_words(index_map, right)
		if rand(0,1) <= 0.75:
			train_data.append([wrong, right])
			train_label.append([wrong, label])
		else:
			test_data.append([wrong, right])
			test_label.append([wrong, label])

	if log_short: print "Vocabulary size:", len(index_map) + 1
	produce_data_files(train_data, train_label, test_data, test_label, particles, index_map, reverse_index, len(index_map)+1)

def main(preprocessor):
	data, particles = prepare_first_data(preprocessor["sample_by"], int(preprocessor[preprocessor["sample_by"]]), preprocessor["sentence_mode"])

	prepare_input(data, particles)
