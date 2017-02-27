from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import numpy as np
import time

# fix random seed for reproducibility
np.random.seed(time.time())

# load the dataset in a tuple
weights = np.load(open("embeds.npy", 'rb'))
(X_train, y_train), (X_test, y_test) = 

# pad input_length
max_sentence_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_sentence_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_sentence_length)

# create the model
word_length = 300
model = Sequential([
	Embedding(input_dim=weights.shape[0],
					output_dim=weights.shape[1],
					weights=[weights],
					input_length=max_sentence_length),
	LSTM(300, return_sequences=False),
	Dense(2),
	Activation('sigmoid')
])
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# train model
model.fit(X_train, y_train, nb_epoch=10, batch_size=64)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score)
print('Test accuracy:', acc)
