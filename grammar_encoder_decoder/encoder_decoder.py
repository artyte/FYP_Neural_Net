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
    import numpy as np
    weights = np.load(open("embeds.npy", 'rb'))
	X_train = pickle_return('training_input_vectors.p')
	y_train = pickle_return('training_output_vectors.p')
    X_test = pickle_return('testing_input_vectors.p')
    y_test = pickle_return('testing_output_vectors.p')
    reverse_index = pickle_return('reverse_index.p')
    return weights, X_train, y_train, X_test, y_test, reverse_index

def train_model(weights, X_train, y_train):
    from keras.models import Sequential
    from keras.layers.embeddings import Embedding
    from seq2seq.models import Seq2Seq
    model = Sequential()
    mode.add(Embedding(weights=[weights]))
    model.add(Seq2Seq(peek=True))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='rmsprop')
    model.fit(X_train, y_train, nb_epoch=5, batch_size=5)
    return model

def save_model(model):
    import pickle
    f = open('model.p', 'w')
	pickle.dump(model, f)
	f.close()

def test_default_data(model, X_test, y_test):
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

def test_custom_data(model, index, reverse_index):
    data = raw_input("Enter a sentence: ")
    # do index conversion here
    result = model.predict(data, verbose=1)

    sentence = []
    for num in result:
        sentence += reverse_index[num]
