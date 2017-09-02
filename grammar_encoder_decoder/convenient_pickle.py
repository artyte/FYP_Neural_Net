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
