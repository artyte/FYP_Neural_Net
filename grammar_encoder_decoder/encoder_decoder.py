import sys
import codecs
reload(sys)
sys.setdefaultencoding('utf8')

max_len = 300

def pickle_return(filename):
    import pickle
    f = open(filename, 'r')
	data = pickle.load(f)
	f.close()
    return data

def extract_data():
    from keras.preprocessing.sequence import pad_sequences as ps
    weights = pickle_return('embeds.p')
    X_train = ps(pickle_return('training_input_vectors.p'), maxlen=max_len)
    y_train = ps(pickle_return('training_output_vectors.p'), maxlen=max_len)
    X_test = ps(pickle_return('testing_input_vectors.p'), maxlen=max_len)
    y_test = ps(pickle_return('testing_output_vectors.p'), maxlen=max_len)
    index_map = pickle_return('index.p')
    reverse_index = pickle_return('reverse_index.p')
    return weights, X_train, y_train, X_test, y_test, index_map, reverse_index

def train_model(weights, X_train, y_train):
    from keras.models import Sequential
    from keras.layers.embeddings import Embedding
    from seq2seq.models import Seq2Seq
    model = Sequential([
        Embedding(input_dim=weights.shape[0],
                    output_dim=weights.shape[1],
                    weights=[weights],
                    input_length=max_len,
                    trainable=False),
        Seq2Seq(hidden_dim=100,
                output_length=max_len,
                output_dim=1,
                depth=[300,300],
                dropout=0.1,
                peek=True)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics='rmsprop')
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
        if reverse_index_map[num] == None: predict += 0
        else: predict += reverse_index_map[num]
    print " ".join(predict)

weights, X_train, y_train, X_test, y_test, index_map, reverse_index = extract_data()
save_model(train_model(weights, X_train, y_train))
