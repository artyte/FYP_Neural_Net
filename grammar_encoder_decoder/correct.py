from os.path import join
from convenient_pickle import pickle_return
log_short = pickle_return(join("data", "log_short.p"))
log_long = pickle_return(join("data", "log_long.p"))

def random_batch(params, data=None):
	import random
	from keras.preprocessing.sequence import pad_sequences as ps

	final_input = []
	final_output = []
	epoch_finished = False
	for i in range(int(params["batch_size"])):
		if len(data) == 0:
			epoch_finished = True
			break
		# use python list here instead of numpy array because numpy array doesn't have append
		instance = random.choice(data) # don't use shuffle because batch is quite huge
		data.remove(instance) # remove so that data won't appear again
		final_input.append(instance[0])
		final_output.append(instance[1])

	seq_len = int(params["seq_len"])
	from torch.autograd import Variable
	import torch
	import numpy as np
	# reverse to pad from end -> pad -> reverse -> convert to pytorch tensor
	x = ps([i[::-1] for i in final_input], maxlen=seq_len).tolist()
	final_input = Variable(torch.from_numpy(np.array([i[::-1] for i in x])).long())
	y = ps([i[::-1] for i in final_output], maxlen=seq_len).tolist()
	final_output = Variable(torch.from_numpy(np.array([i[::-1] for i in y])).long())

	return final_input, final_output, epoch_finished, data

def iterate(seq2seq, input, target, optimizer, criterion):
	# for each training cycle, zero the gradients out otherwise gradients will accumulate
	seq2seq.zero_grad()

	# pass data through these 2 layers
	output = seq2seq(input)
	output = output.transpose(0,1) # transposed axis : B x S x D (batch as first axis so as to iterate easily)
	target = target.cuda()

	# target.size(0) is the batch axis
	loss = sum([criterion(output[i], target[i]) for i in range(target.size(0))])
	'''for i in range(target.size(0)): loss += criterion(output[i], target[i])'''

	loss.backward()

	import torch
	torch.nn.utils.clip_grad_norm(seq2seq.parameters(), 5) # do this otherwise gradients will explode
	optimizer.step()

	return loss.data[0]

def training_loop(model, optimizer, criterion, option, params):
	# use time to print evaluate_rate duration and training duration
	from time import time
	total_time = time()

	loss = 0.0 # total loss per evaluate rate iteration
	epoch = 0
	loss_list = [] # store history of loss per epoch
	with open("continue_epoch.txt", "w") as f: f.write("start")
	while bool(open("continue_epoch.txt").readline()): # whether to continue running
		total_loss = 0.0 # total loss per epoch

		start_iterate = time()

		# reset temporary batch data for memory efficiency
		data = pickle_return(join("data",'train_label.p')) if option == "train label" else pickle_return(join("data",'train_data.p'))

		epoch_finished = False
		num_of_iterations = 0
		while not epoch_finished:
			input, output, epoch_finished, data = random_batch(params, data=data)
			input = input.cuda()
			loss += iterate(model, input, output, optimizer, criterion)

			num_of_iterations += 1

			if num_of_iterations % int(params["evaluate_rate"]) == 0:
				diff = (time() - start_iterate) / 60.0
				print 'epoch: %d\titeration: %d\tloss: %f\tduration: %f' %  (epoch + 1, num_of_iterations, loss, diff)
				total_loss += loss # accumulate loss
				loss = 0.0
				start_iterate = time()

		loss_list.append(total_loss)
		epoch += 1

	data = []
	data.append(range(epoch))
	data.append(loss_list)

	# get total time spent on training
	from datetime import datetime, timedelta
	total_time = datetime(1,1,1) + timedelta(seconds=time() - total_time)

	if log_short: print "total time spent:", total_time

	history = {}
	history["loss_list"] = data
	history["epoch"] = epoch
	history["total_time"] = total_time

	return model, params, history

def save_net(model, params, history, option):
	import re
	variables = open(join("data", "hyperparam_defaults.txt")).readlines()
	variables = [re.sub('\n', '', variable) for variable in variables] # get name of hyperparameters
	variables = [str(params[variable]) for variable in variables] # get value of respective hyperparameters

	variables2 = variables[:] # to be used later

	if log_short:
		print "saving model..."
		print "Variables:", variables
		print "Variables2:", variables2

	for index, variable in enumerate(variables):
		if index % 2 == 1: variables.insert(index, '_')

	name = "".join(variables)

	import torch
	from convenient_pickle import pickle_dump
	if option == "train label": torch.save(model, join("models", name + ".label")) # save label net
	elif option == "train model": torch.save(model, join("models", name + ".model")) # save model net
	pickle_dump(join("models", name + ".history"), history) # save model running histories

	# use readlines to retain \n
	param_data = []
	param_data = open(join("data", "param_format.txt")).readlines()
	param_data = [i.split("%") for i in param_data]
	param_data = [k for i in param_data for k in i]

	if log_short: print "param_data", param_data, "\n"

	variables2.append("nn.CrossEntropyLoss().cuda()")
	variables2.append(str(params["output_size"]))
	variables2 = variables2[::-1]

	for index, item in enumerate(param_data):
		if item == "": param_data[index] = variables2.pop()

	with open(join("models", name + ".param"), 'w') as f: f.write("".join(param_data))

def train(params, option):
	# net initilizations
	from models.seq2seq import Seq2Seq
	seq2seq = Seq2Seq(int(params["embed_hidden_size"]), int(params["encoder_hidden_size"]), int(params["decoder_hidden_size"]), int(params["output_size"]), int(params["seq_len"]),  params["seq2seq_type"], int(params["index_dim"]), option).cuda()

	# initilize optimizers & loss functions
	# don't initilize in a separate train function because the net can't keep track if these variables are deallocated
	from torch import optim
	to_train = filter(lambda p: p.requires_grad, seq2seq.parameters())
	optimizer = optim.SGD(to_train, lr=float(params["lr"]), momentum=0.9, weight_decay=1e-4)
	criterion = params["loss_function"]

	seq2seq, params, history = training_loop(seq2seq, optimizer, criterion, option, params)
	save_net(seq2seq, params, history, option)

def correct_sentence(params, label, model):
	input = raw_input("Enter a sentence: ")

	from nltk.tokenize import word_tokenize as wt
	sentence = wt(input)
	sentence_tmp = []

	from convenient_pickle import pickle_return
	index_map = pickle_return(join("data",'index_map.p'))
	OoV = [] # out of vocabulary
	for word in sentence:
		tmp = word # in case of upper case
		word = word.lower()
		if word not in index_map:
			sentence_tmp.append(0)
			OoV.append(tmp)
		else: sentence_tmp.append(int(index_map[word][1]))

	# make sentence_tmp list of list to fit keras api
	# reverse to pad from end -> pad -> reverse -> convert to pytorch tensor
	from keras.preprocessing.sequence import pad_sequences as ps
	x = ps([i[::-1] for i in [sentence_tmp]], maxlen=seq_len).tolist()
	input = Variable(torch.from_numpy(np.array([i[::-1] for i in x])).long())
	input = input.cuda()

	particles = pickle_return('particles.p')
	for i in sentence_tmp:
		if i not in particles: particles.append(i)

	mask = Variable(torch.zeros(output_size)).cuda()
	for i in particles: mask[i] = 1.0

	# get output from model
	output = model(input)
	output = output.transpose(0,1) # batch must be on the first axis
	for i in range(seq_len):
		output[0,i] = output[0,i] * mask
	values, indices = output.max(2)

	# convert into list for reverse mapping
	indices = indices.cpu()
	indices = indices.transpose(1,2).squeeze(0).squeeze(0).data.numpy().tolist()

	# get label
	labels = label(input, label.get_hidden(input.size(0))).cpu().data.numpy().tolist()
	print labels

	'''labels = 0 if labels < 0.5 else 1

	if labels == 0:
		indices = []
		print "Your sentence is correct"
	else:
		indices = indices[:len(sentence_tmp) + 1]
		print "Sentence seems wrong."

		# reverse mapping
		reverse_index = pickle_return('reverse_index.p')
		predict = []
		for num in indices:
			if num == 0 and OoV: predict.append(OoV.pop(0))
			if num != 0: predict.append(reverse_index[num])
		sentence = " ".join(predict)
		sentence = sentence[0].upper() + sentence[1:]
		print "Suggested sentence: %s" % (sentence)'''

'''def evaluate():'''

def main(name, option=None):
	path = "models"
	if option == "train":
		train(exec(open(join(path, name + ".param"))), "train label")
		train(exec(open(join(path, name + ".param"))), "train model")
	elif option == "correct sentence":
		correct_sentence(exec(join(path, name + ".param")), torch.load(join(path, name + ".label")), torch.load(join(path, name + ".model")))
	'''elif option == "evaluate test data":
		evaluate()'''
