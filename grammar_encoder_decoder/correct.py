from os.path import join
from convenient_pickle import pickle_return
log_short = pickle_return(join("data", "log_short.p"))
log_long = pickle_return(join("data", "log_long.p"))

def make_batch(params, data):
	batch_size = params["batch_size"]

	from random import shuffle
	shuffle(data) # randomize order of data

	data_batch = []
	for i in range(0, len(data), batch_size):
		input_batch = []
		output_batch = []
		for item in data[i:i + batch_size]:
			input_batch.append(item[0])
			output_batch.append(item[1])
		data_batch.append((input_batch, output_batch))

	return data_batch

def pad_tensor(params, final_input, final_output):
	seq_len = int(params["seq_len"])

	from keras.preprocessing.sequence import pad_sequences as ps
	from torch.autograd import Variable
	import torch
	import numpy as np
	# reverse to pad from end -> pad -> reverse -> convert to pytorch tensor
	x = ps([i[::-1] for i in final_input], maxlen=seq_len).tolist()
	final_input = Variable(torch.from_numpy(np.array([i[::-1] for i in x])).long())
	y = ps([i[::-1] for i in final_output], maxlen=seq_len).tolist()
	final_output = Variable(torch.from_numpy(np.array([i[::-1] for i in y])).long())

	return final_input, final_output

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
	evaluate_rate = int(raw_input("Enter evaluate rate: "))

	# use time to print evaluate_rate duration and training duration
	from time import time
	total_time = time()

	loss = 0.0 # total loss per evaluate rate iteration
	epoch = 0
	loss_list = [] # store history of loss per epoch
	with open("continue_epoch.txt", "w") as f: f.write("start")
	while bool(open("continue_epoch.txt").readline()): # whether to continue running
		total_loss = 0.0 # total loss per epoch

		# reset temporary batch data for memory efficiency
		data = pickle_return(join("data",'train_label.p')) if option == "train label" else pickle_return(join("data",'train_data.p'))

		num_of_iterations = 0
		for input, output in make_batch(params, data):
			start_iterate = time()

			input, output = pad_tensor(params, input, output)
			input = input.cuda()
			loss += iterate(model, input, output, optimizer, criterion)

			num_of_iterations += 1

			if num_of_iterations % evaluate_rate == 0:
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

def save_net(model, name, history, option):
	import torch
	from convenient_pickle import pickle_dump
	if option == "train label":
		torch.save(model, join("models", name + ".label")) # save label net
		pickle_dump(join("models", name + ".labelhistory"), history) # save model running histories
	elif option == "train model":
		torch.save(model, join("models", name + ".model")) # save model net
		pickle_dump(join("models", name + ".modelhistory"), history) # save model running histories

def train(name, option):
	hyperparameters = {}
	exec(open(join("models", name + ".param")))

	params = hyperparameters

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
	save_net(seq2seq, name, history, option)

def correct_sentence(name, mode):
	import torch
	path = "models"

	hyperparameters = {}
	exec(open(join(path, name + ".param")))

	label = torch.load(join(path, name + ".label"))
	model = torch.load(join(path, name + ".model"))

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
		else: sentence_tmp.append(index_map[word])

	# make sentence_tmp list of list to fit keras api
	# reverse to pad from end -> pad -> reverse -> convert to pytorch tensor
	from keras.preprocessing.sequence import pad_sequences as ps
	from torch.autograd import Variable
	import numpy as np
	x = ps([i[::-1] for i in [sentence_tmp]], maxlen=hyperparameters["seq_len"]).tolist()
	input = Variable(torch.from_numpy(np.array([i[::-1] for i in x])).long())
	input = input.cuda()

	particles = pickle_return(join("data",'particles.p'))
	particles = [index_map[particle] for particle in particles]
	for i in sentence_tmp:
		if i not in particles: particles.append(i)

	mask = Variable(torch.zeros(hyperparameters["output_size"])).cuda()
	for i in particles: mask[i] = 1.0

	output = get_output([i[::-1] for i in x], input, label, model, mask, hyperparameters, mode)[0]

	# reverse mapping
	reverse_index = pickle_return(join("data",'reverse_index.p'))
	predict = []
	for num in output:
		if num == 0 and OoV: predict.append(OoV.pop(0))
		if num != 0: predict.append(reverse_index[num])
	sentence = " ".join(predict)
	sentence = sentence[0].upper() + sentence[1:]
	print "Suggested sentence: %s" % (sentence)

def get_output(list_input, input, label, model, mask, hyperparameters, mode="evaluate"):
	import numpy as np
	# get output from model
	output = model(input)
	output = output.transpose(0,1) # batch must be on the first axis
	if mode == "correct":
		for i in range(hyperparameters["seq_len"]):
			output[0,i] = output[0,i] * mask # 0 because batch size is 1
	elif mode == "evaluate":
		for i in range(hyperparameters["seq_len"]):
			for k in range(output.size(0)):
				output[k,i] = output[k,i] * mask[k]
	values, indices = output.max(2)

	# convert into list for reverse mapping
	indices = indices.cpu()
	indices = np.asarray(indices.transpose(1,2).squeeze(0).data.numpy()) if mode == "correct" else np.asarray(indices.squeeze(2).data.numpy())

	# get label
	labels = label(input).cpu().squeeze(1)
	values, indices_masks = labels.max(1) if mode == "correct" else labels.max(2)
	indices_masks = indices_masks.squeeze(2) if mode == "evaluate" else indices_masks
	indices_masks = indices_masks.data.numpy()
	for i, indices_mask in enumerate(indices_masks):
		for k, item in enumerate(indices_mask):
			if item == 2: indices_masks[i,k] = -1 # -1 for easy detection of words to change into indices
	indices_masks = np.transpose(indices_masks)
	indices_masks *= np.asarray(list_input)

	# get actual output
	for i, indices_mask in enumerate(indices_masks):
		for k, item in enumerate(indices_mask):
			if item < 0: indices_masks[i,k] = indices[i,k] # if negative, change to indices

	return indices_masks.tolist()

def one_batch_evaluation(name, input, candidates, params, mode):
	import torch
	path = "models"

	label = torch.load(join(path, name + ".label"))
	model = torch.load(join(path, name + ".model"))

	sentence_tmp = []

	from convenient_pickle import pickle_return
	index_map = pickle_return(join("data",'index_map.p'))
	particle_list = pickle_return(join("data",'particles.p'))
	particle_list = [index_map[particle] for particle in particle_list]
	particles = [] # a list of list, each list contains a particle_list + that sentence's words
	list_input = input.cpu().data.numpy().tolist()

	for i in list_input:
		tmp = particle_list[:] # start off with only particles
		for k in i:
			if k not in tmp: tmp.append(k) # add sentence's words in one particle list

		particles.append(tmp)

	from torch.autograd import Variable
	mask = Variable(torch.zeros(int(params["batch_size"]), params["output_size"])).cuda()
	for i, particle in enumerate(particles):
		for k, item in enumerate(particle):
			mask[i, item] = 1.0

	references = get_output([i[::-1] for i in list_input], input, label, model, mask, params, mode)

	from nltk.metrics.scores import precision as p
	from nltk.metrics.scores import recall as r
	from nltk.metrics.scores import f_measure as f
	from nltk.translate.bleu_score import sentence_bleu as bleu

	precision = 0.0
	recall = 0.0
	f_measure = 0.0
	blu = 0.0

	candidates = candidates.cpu().data.numpy().tolist()
	for reference, candidate in zip(references, candidates):
		precision += p(set(reference), set(candidate))
		recall += r(set(reference), set(candidate))
		f_measure += f(set(reference), set(candidate))
		blu += bleu([str(i) for i in reference], [str(i) for i in candidate])

	return precision, recall, f_measure, blu, len(candidates)

def evaluate(name, mode):
	hyperparameters = {}
	exec(open(join("models", name + ".param")))
	params = hyperparameters

	total_loss = [0.0, 0.0, 0.0, 0.0, 0] # total loss per epoch

	# reset temporary batch data for memory efficiency
	from convenient_pickle import pickle_return
	data = pickle_return(join("data",'test_data.p'))

	epoch_finished = False
	params["batch_size"] = 73 # in case previously large models use too much memory
	for input, output in make_batch(params, data):
		input, output = pad_tensor(params, input, output)
		input = input.cuda()
		loss = one_batch_evaluation(name, input, output, params, mode)
		total_loss = [num+loss[i] for i, num in enumerate(total_loss)]

	avg_loss = [i/total_loss[4] for i in total_loss]

	print "Precision:", avg_loss[0]
	print "Recall:", avg_loss[1]
	print "F_measure:", avg_loss[2]
	print "Bleu:", avg_loss[3]

def show_history(name):
	label_history = pickle_return(join("models", name + ".labelhistory"))
	model_history = pickle_return(join("models", name + ".modelhistory"))

	import matplotlib.pyplot as plt

	print "This is the labeller stats -------------------------------->"
	print "Time spent on training:", label_history["total_time"].day - 1, "day(s)", label_history["total_time"].hour, "hour(s)", label_history["total_time"].minute, "minute(s)", label_history["total_time"].second, "second(s)"
	print "Number of epochs used for training:", label_history["epoch"]
	plt.plot(label_history["loss_list"][0], label_history["loss_list"][1])
	plt.ylabel('loss')
	plt.xlabel('epochs')
	plt.show()

	print ""

	print "This is the corrector stats ------------------------------->"
	print "Time spent on training:", model_history["total_time"].day - 1, "day(s)", model_history["total_time"].hour, "hour(s)", model_history["total_time"].minute, "minute(s)", model_history["total_time"].second, "second(s)"
	print "Number of epochs used for training:", model_history["epoch"]
	plt.plot(model_history["loss_list"][0], model_history["loss_list"][1])
	plt.ylabel('loss')
	plt.xlabel('epochs')
	plt.show()

	print ""

def main(name, option):
	if option == "train":
		train(name, "train model")
		train(name, "train label")
	elif option == "correct sentence":
		correct_sentence(name, "correct")
	elif option == "evaluate test data":
		evaluate(name, "evaluate")
	elif option == "show history":
		show_history(name)
