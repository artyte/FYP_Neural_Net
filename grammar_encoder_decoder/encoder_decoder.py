import sys
import codecs
reload(sys)
sys.setdefaultencoding('utf8')

output_dim = [50, 20, 1]
repeat = 200
max_len = 200
epochs = 5
batch = 32
grad_desc = 'adam'
error_calc = 'categorical_crossentropy'

def pickle_return(filename):
	import pickle
	f = open(filename, 'r')
	data = pickle.load(f)
	f.close()
	return data

def extract_data():
	from keras.preprocessing.sequence import pad_sequences as ps
	import numpy as np
	weights = np.load(open("embeds.npy", 'rb'))
	X_train = ps(pickle_return('training_input_vectors.p'), maxlen=max_len)
	y_train = ps(pickle_return('training_output_vectors.p'), maxlen=max_len)
	X_test = ps(pickle_return('testing_input_vectors.p'), maxlen=max_len)
	y_test = ps(pickle_return('testing_output_vectors.p'), maxlen=max_len)
	index_map = pickle_return('index.p')
	reverse_index = pickle_return('reverse_index.p')
	return weights, X_train, y_train, X_test, y_test, index_map, reverse_index

def train_model(weights, X_train, y_train):
	from keras.models import Sequential
	from keras.layers import Embedding, Dense, RepeatVector, TimeDistributed
	from recurrentshop import RecurrentSequential, LSTMCell, GRUCell

	model = Sequential()
	model.add(Embedding(input_dim=weights.shape[0],
						output_dim=output_dim[0],
						weights=[weights],
						input_length=max_len,
						trainable=False))
	encoder = RecurrentSequential(state_initializer='random_normal',
								state_sync=True,
								teacher_force=True,
								return_sequences=False)
	encoder.add(LSTMCell(output_dim[1]))
	model.add(encoder.get_cell())
	model.add(RepeatVector(repeat))
	decoder = RecurrentSequential(state_initializer='random_normal',
								readout='add',
								state_sync=True,
								teacher_force=True,
								return_sequences=True)
	decoder.add(GRUCell(output_dim[2]))
	model.add(decoder.get_cell())
	model.add(TimeDistributed(Dense(output_dim[2], activation = 'softmax'))
	model.compile(optimizer=grad_desc, loss=error_calc, metrics='rmsprop')
	model.fit(X_train, y_train, nb_epoch=epochs, batch_size=batch)
	return model

def save_model(model):
	import pickle
	f = open('model.p', 'w')
	pickle.dump(model, f)
	f.close()

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
save_model(train_model(weights, X_train, y_train))
