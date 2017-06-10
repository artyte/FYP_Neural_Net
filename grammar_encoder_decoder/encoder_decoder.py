import sys
import codecs
reload(sys)
sys.setdefaultencoding('utf8')

def pickle_return(filename):
	import pickle
	f = open(filename, 'r')
	data = pickle.load(f)
	f.close()
	return data

def numpy_return(filename):
	import numpy as np
	return np.load(open(filename, 'rb'))

def extract_data(num):
	from keras.preprocessing.sequence import pad_sequences as ps

	max_len = 300
	weights = numpy_return('embeds.npy')
	X_train = ps(numpy_return('training_input_vectors.npy'), maxlen=max_len)
	y_train = ps(numpy_return('training_output_vectors.npy'), maxlen=max_len)
	y_train = y_train.astype('float')/22355
	X_test = ps(numpy_return('testing_input_vectors.npy'), maxlen=max_len)
	y_test = ps(numpy_return('testing_output_vectors.npy'), maxlen=max_len)
	y_test = y_test.astype('float')/22355
	index_map = pickle_return('index.p')
	reverse_index = pickle_return('reverse_index.p')
	return weights, X_train, y_train, X_test, y_test, index_map, reverse_index

def train_model(weights, X_train, y_train):
	from keras.models import Sequential
	from keras.layers import Flatten, Embedding, Dense, RepeatVector, GRU, LSTM, TimeDistributed
	from keras.optimizers import Adam

	model = Sequential([
		Embedding(input_dim=weights.shape[0],
							output_dim=weights.shape[1],
							weights=[weights],
							input_length=300,
							trainable=False),
		LSTM(50, return_sequences=False),
		RepeatVector(300),
		LSTM(10, return_sequences=True),
		LSTM(1, return_sequences=True),
		TimeDistributed(Dense(1)),
		Flatten()
	])
	adam = Adam(lr=0.01)
	model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
	model.fit(X_train, y_train, nb_epoch=10, batch_size=595)
	model.save('model')

def test_default_data(model, X_test, y_test):
	scores = model.evaluate(X_test, y_test, verbose=0)
	print("Accuracy: %.2f%%" % (scores[1]*100))

def test_custom_data(model, index_map, reverse_index_map, num):
	data = raw_input("Enter a sentence: ")
	from nltk.tokenize import word_tokenize as wt
	sentence = wt(data)
	sentence_tmp = []
	for word in sentence:
		word = word.lower()
		if word not in index_map: sentence_tmp.append(0)
		else: sentence_tmp.append(int(index_map[word][1]))

	from keras.preprocessing.sequence import pad_sequences as ps
	print ps([sentence_tmp], maxlen=300)

	import numpy as np

	result = model.predict(ps(np.asarray([sentence_tmp]), maxlen=300), verbose=1)
	print result*num
	predict = []
	for num in result[0]:
		if int(round(num)) not in reverse_index_map: predict += '##NULL##'
		else: predict += reverse_index_map[int(round(num))]
	print " ".join(predict)

weights, X_train, y_train, X_test, y_test, index_map, reverse_index = extract_data(400000)
train_model(weights, X_train, y_train)
from keras.models import load_model as lm
test_default_data(lm('model'), X_test, y_test)
test_custom_data(lm('model'), index_map, reverse_index, 400000)
