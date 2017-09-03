import torch
import torch.nn as nn
from torch.autograd import Variable
from os.path import join
from convenient_pickle import pickle_return
log_short = pickle_return(join("data", "log_short.p"))
log_long = pickle_return(join("data", "log_long.p"))

class Decoder(nn.Module):
	def __init__(self, hidden_size, output_size, seq_len):
		super(Decoder, self).__init__()

		self.seq_len = seq_len
		self.output_size = output_size
		self.hidden_size = hidden_size

		self.shrink = nn.Linear(self.output_size, self.hidden_size)

	def get_hidden(self, batch_size):
		return Variable(torch.zeros(batch_size, self.hidden_size)).cuda()

	def get_output(self, batch_size):
		return Variable(torch.zeros(batch_size, self.output_size)).cuda()
