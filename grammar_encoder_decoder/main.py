# set up logging
from convenient_pickle import pickle_dump
from os.path import join
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-ls', '--log-short', dest='log_short', action='store_true', help='show short logging messages')
parser.add_argument('-ll', '--log-long', dest='log_long', action='store_true', help='show long logging messages')
log_short = parser.parse_args().log_short
log_long = parser.parse_args().log_long
pickle_dump(join("data", "log_short.p"), log_short)
pickle_dump(join("data", "log_long.p"), log_long)

def loop_line(lines, *args):
	choice = None
	data = []

	import re
	for line in lines:
		if choice == "0": break

		line = re.sub('\n', '', line) # all lines have a trailing \n

		# don't ask for input
		if line[0] == "!":
			line = line[1:]
			if len(args) == 2 and args[0] == "2":
				line = line.split("\t")
				if args[1] == None: line.insert(1, "None")
				else: line.insert(1, args[1])
				line = " ".join(line)
			print line
			continue

		# ask for choice input
		if line[1] != "-":
			choice = raw_input(line)
			continue

		# ask for data input of corresponding choice
		line = line.split("-")
		if re.search('\d', line[0]):
			if int(line[0]) == int(choice):
				data.append(raw_input(line[1]))
		elif line[0] == "A":
			data.append(raw_input(line[1]))

	return choice, data

def prepare_hyperparam(data):
	# use readlines to retain \n
	param_data = []
	param_data = open(join("data", "param_format.txt")).readlines()
	param_data = [i.split("%") for i in param_data]
	param_data = [k for i in param_data for k in i]

	data.append("nn.CrossEntropyLoss().cuda()")
	data.append(pickle_return(join("data","output_size.p")))
	data = data[::-1]

	for index, item in enumerate(param_data):
		if item == "": param_data[index] = data.pop()

	with open(join("models", name + ".param"), 'w') as f: f.write("".join(param_data))

def main():
	choice = "N"
	data = None
	path = "ui"

	file_format = ".txt"
	file_num = "1"

	while True:
		with open(join(path, file_num + file_format)) as f:
			choice, _ = loop_line(f.readlines())
		if choice == "0": break

		sub_choice = None
		model_name = None
		while True:
			print "\n"

			# use with because variables have been initialized
			with open(join(path, file_num + "_" + choice + file_format)) as f:
				sub_choice, data = loop_line(f.readlines(), choice, model_name)
				if sub_choice == "1" and choice == "2": model_name = prepare_hyperparam(data)
			if sub_choice == "0": break

			# don't use with because variables created in exec statement may need to be retained
			f = open(join(path, file_num + "_" + choice + "c" + sub_choice + file_format))
			exec(f)
			f.close()

		print "\n\n\n"

main()
