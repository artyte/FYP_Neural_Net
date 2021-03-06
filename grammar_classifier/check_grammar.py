from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.recurrent import LSTM, GRU
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.regularizers import l2
from wikihistory import WikiHistory as wh
import numpy as np

# fix random seed for reproducibility and initialise dataset
np.random.seed(1)
wh()

# load the dataset and weights
weights = np.load(open("embeds.npy", 'rb'))
X_train = wh.train_X
y_train = wh.train_Y
X_test = wh.test_X
y_test = wh.test_Y

# pad input_length
max_sentence_length = 300
X_train = sequence.pad_sequences(X_train, maxlen=max_sentence_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_sentence_length)

# create the model
model = Sequential([
	Embedding(input_dim=weights.shape[0],
					output_dim=weights.shape[1],
					weights=[weights],
					input_length=max_sentence_length),
	MaxPooling1D(pool_length=5),
	GRU(weights.shape[1],
			W_regularizer=l2(0.05),
			dropout_W=0.2,
			return_sequences=True),
	MaxPooling1D(pool_length=2),
	GRU(int(weights.shape[1]/2),
			W_regularizer=l2(0.05),
			dropout_W=0.1,
			return_sequences=False),
	Dense(1, activation = "sigmoid")
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# train model
model.fit(X_train, y_train, nb_epoch=2, batch_size=5)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
