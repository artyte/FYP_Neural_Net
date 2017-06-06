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

def extract_data():
	from keras.preprocessing.sequence import pad_sequences as ps
	from keras.utils.np_utils import to_categorical as tc
	import numpy as np

	max_len = 200
	weights = np.load(open("embeds.npy", 'rb'))
	X_train = np.asarray(ps(pickle_return('training_input_vectors.p'), maxlen=max_len))
	y_train = np.asarray(ps(pickle_return('training_output_vectors.p'), maxlen=max_len))
	X_test = np.asarray(ps(pickle_return('testing_input_vectors.p'), maxlen=max_len))
	y_test = np.asarray(ps(pickle_return('testing_output_vectors.p'), maxlen=max_len))
	'''
	np.reshape(X_train, (X_train.shape[1], X_train.shape[2], 1))
	np.reshape(y_train, (y_train.shape[1], y_train.shape[2], 1))
	np.reshape(X_test, (X_test.shape[1], X_test.shape[2], 1))
	np.reshape(y_test, (y_test.shape[1], y_test.shape[2], 1))
	'''
	index_map = pickle_return('index.p')
	reverse_index = pickle_return('reverse_index.p')
	return weights, X_train, y_train, X_test, y_test, index_map, reverse_index

def train_model(weights, X_train, y_train):
	from keras.models import Sequential
	from keras.layers import Flatten, Embedding, Dense, RepeatVector, GRU, LSTM
	from keras.optimizers import Adam

	model = Sequential([
		Embedding(input_dim=weights.shape[0],
							output_dim=weights.shape[1],
							weights=[weights],
							input_length=200,
							trainable=False),
		LSTM(150, return_sequences=True),
		LSTM(100, return_sequences=False),
		RepeatVector(200),
		GRU(80, return_sequences=True),
		GRU(50, return_sequences=True),
		GRU(20, return_sequences=True),
		Flatten(),
		Dense(200, activation = 'sigmoid')
	])
	print model.summary()
	adam = Adam(lr=0.01, beta_1=0.90, beta_2=0.999, epsilon=1e-06, decay=0.00)
	model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
	model.fit(X_train, y_train, nb_epoch=20, batch_size=50)
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
		if index_map[word] == None: sentence_tmp.append(0)
		else: sentence_tmp.append(int(index_map[word]))

	result = model.predict(data, verbose=1)

	predict = []
	for num in result:
		if reverse_index_map[int(round(num))] == None: predict += '##NULL##'
		else: predict += reverse_index_map[int(round(num))]
	print " ".join(predict)

weights, X_train, y_train, X_test, y_test, index_map, reverse_index = extract_data()
train_model(weights, X_train, y_train)
