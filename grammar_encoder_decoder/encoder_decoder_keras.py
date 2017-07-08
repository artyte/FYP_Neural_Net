import sys
import codecs
reload(sys)
sys.setdefaultencoding('utf8')

max_len = 100
word_dim = 5026

def pickle_return(filename):
	import pickle
	f = open(filename, 'r')
	data = pickle.load(f)
	f.close()
	return data

def numpy_return(filename):
	import numpy as np
	return np.load(open(filename, 'rb'))

def extract_data():
	from keras.preprocessing.sequence import pad_sequences as ps
	from keras.utils import to_categorical as tc

	weights = numpy_return('embeds.npy')
	X_train = ps(numpy_return('training_input_vectors.npy'), maxlen=max_len)
	y_train = tc(ps(numpy_return('training_output_vectors.npy'), maxlen=max_len), num_classes=word_dim)
	y_train = np.reshape(y_train, (-1, max_len, word_dim))
	X_test = ps(numpy_return('testing_input_vectors.npy'), maxlen=max_len)
	y_test = tc(ps(numpy_return('testing_output_vectors.npy'), maxlen=max_len), num_classes=word_dim)
	y_test = np.reshape(y_test, (-1, max_len, word_dim))
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
				input_length=max_len,
				trainable=False),
		LSTM(100, return_sequences=False),
		RepeatVector(max_len),
		LSTM(100, return_sequences=True),
		TimeDistributed(Dense(word_dim, activation="sigmoid"))
	])
	print model.summary()
	adam = Adam(lr=0.01)
	model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
	model.fit(X_train, y_train, nb_epoch=10, batch_size=1)
	model.save('model')

def test_default_data(model, X_test, y_test):
	scores = model.evaluate(X_test, y_test, verbose=0)
	print("Accuracy: %.2f%%" % (scores[1]*100))

def test_custom_data(model, index_map, reverse_index_map):
	data = raw_input("Enter a sentence: ")
	from nltk.tokenize import word_tokenize as wt
	sentence = wt(data)
	sentence_tmp = []
	for word in sentence:
		word = word.lower()
		if word not in index_map: sentence_tmp.append(0)
		else: sentence_tmp.append(int(index_map[word][1]))

	from keras.preprocessing.sequence import pad_sequences as ps
	print ps([sentence_tmp], maxlen=max_len)

	import numpy as np

	result = model.predict(ps(np.asarray([sentence_tmp]), maxlen=max_len), verbose=1)
	print result
	predict = []
	for num in result[0]:
		if int(round(num)) not in reverse_index_map: predict += '##NULL##'
		else: predict += reverse_index_map[int(round(num))]
	print " ".join(predict)

#weights, X_train, y_train, X_test, y_test, index_map, reverse_index = extract_data()

from keras.preprocessing.sequence import pad_sequences as ps
from keras.utils import to_categorical as tc
import numpy as np
'''
train_model(numpy_return('embeds.npy'),
			ps(numpy_return('testing_input_vectors.npy'),
				maxlen=max_len),
			np.reshape(tc(ps(numpy_return('testing_output_vectors.npy'),
							maxlen=max_len),
						  num_classes=word_dim),
						(-1, max_len, word_dim)))
'''
from keras.models import load_model as lm
test_default_data(lm('model'),
				ps(numpy_return('testing_input_vectors.npy'),
					maxlen=max_len),
				np.reshape(tc(ps(numpy_return('testing_output_vectors.npy'),
								maxlen=max_len),
							  num_classes=word_dim),
							(-1, max_len, word_dim)))
#test_custom_data(lm('model'), index_map, reverse_index)'''
