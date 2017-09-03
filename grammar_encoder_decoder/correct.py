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

	loss = 0.0
	for i in range(target.size(0)): # target.size(0) is the batch axis
		loss += criterion(output[i], target[i])

	loss.backward()
	torch.nn.utils.clip_grad_norm(seq2seq.parameters(), 5) # do this otherwise gradients will explode
	optimizer.step()

	return loss.data[0]

def training_loop(model, optimizer, criterion, option, params):
	# use time to print evaluate_rate duration and training duration
	from time import time
	total_time = time()

	loss = 0.0 # total loss per evaluate rate iteration
	epoch = 0
	epoch_list = [] # store history of epoch run number
	loss_list = [] # store history of loss per epoch
	while bool(open("continue_epoch.txt").readline()): # whether to continue running
		total_loss = 0.0 # total loss per epoch

		start_iterate = time()

		# reset temporary batch data for memory efficiency
		data = pickle_return('train_label.p') if option == "train label" else pickle_return('train_data.p')

		epoch_finished = False
		num_of_iterations = 0
		while not epoch_finished:
			input, output, epoch_finished, data = random_batch(params, data=data)
			input = input.cuda()
			loss += iterate(model, input, output, optimizer, criterion)

			num_of_iterations += 1

			if num_of_iterations % params["evaluate_rate"] == 0:
				diff = (time() - start_iterate) / 60.0
				print 'epoch: %d\titeration: %d\tloss: %f\tduration: %f' %  (epoch + 1, num_of_iterations, loss, diff)
				total_loss += loss # accumulate loss
				loss = 0.0
				start_iterate = time()

		loss_list.append(total_loss)
		epoch += 1
		epoch_list.append(epoch)

	data = []
	data.append(epoch_list)
	data.append(loss_list)

	# get total time spent on training
	from datetime import datetime, timedelta
	total_time = datetime(1,1,1) + timedelta(seconds=time() - total_time)
	total_time.day -= 1

	history = {}
	history["loss_list"] = loss_list
	history["epoch_list"] = epoch_list
	history["total_time"] = total_time

	return model, params, history

def save_model(model, params, history):
	import re
	variables = open(join("data", "hyperparam_defaults.txt")).readlines()
	variables = [re.sub('\n', '', variable) for variable in variables] # get name of hyperparameters
	variables = [params[variable] for variable in variables] # get value of respective hyperparameters

	variables2 = variables[:] # to be used later

	if log_short:
		print "saving model..."
		print "Variables:", variables
		print "Variables2:", variables2

	for index, variable in enumerate(variables):
		if index % 2 == 1: variables.insert(index, '_')

	name = "".join(variables)

	torch.save(seq2seq, join("models", name + ".model")) # save model
	from convenient_pickle import pickle_dump
	pickle_dump(join("models", name + ".history"), history) # save model running histories

	# use readline to retain \n
	param_data = []
	with open(join("data", "param_format.txt")) as f: param_data.append(f.readline())
	param_data = [i.split("%") for i in param_data]
	param_data = [k for i in param_data for k in i]

	if log_short: print "param_data", param_data, "\n"

	variables2.append("nn.CrossEntropyLoss().cuda()")
	variables2.append(params["output_size"])
	variables2 = variables2[::-1]

	for index, item in enumerate(param_data):
		if item == "": param_data.insert(index, variables2.pop())

	with open(join("models", name + ".param"), 'w') as f: f.write("".join(param_data))

def train(params, option):
	# net initilizations
	from models.seq2seq import Seq2Seq
	from convenient_pickle import pickle_return
	output_size = 3 if option == "train label" else pickle_return('output_size.p')
	params["output_size"] = str(output_size)
	seq2seq = Seq2Seq(int(params["embed_hidden_size"]), int(params["encoder_hidden_size"]), int(params["decoder hidden size"]), output_size, int(params["seq_len"]), params["seq2seq_type"]).cuda()

	# initilize optimizers & loss functions
	# don't initilize in a separate train function because the net can't keep track if these variables are deallocated
	to_train = filter(lambda p: p.requires_grad, seq2seq.parameters())
	optimizer = optim.SGD(to_train, lr=int(params["lr"]), momentum=0.9, weight_decay=1e-4)
	criterion = params["loss_function"]

	model, params, history = training_loop(seq2seq, optimizer, criterion, option, params)
	save_net(model, params, history)

def correct_sentence():

def evaluate():

def main(params, option=None):
	if option == "train label" or option == "train model":
		train(params, option)
	elif option == "correct sentence":
		correct_sentence()
	elif option == "evaluate test data":
		evaluate()
