import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from decoder import Decoder
from os.path import join
from convenient_pickle import pickle_return
log_short = pickle_return(join("data", "log_short.p"))
log_long = pickle_return(join("data", "log_long.p"))

class Vanilla(Decoder):
	def __init__(self, hidden_size, output_size, seq_len):
		super(Vanilla, self).__init__(hidden_size, output_size, seq_len)

		self.shrink = nn.Linear(2 * self.hidden_size, self.hidden_size/self.seq_len)
		self.gru = nn.GRUCell(self.hidden_size, self.output_size)

	def forward(self, encoder_output, hidden, decoder_output):
		batch_size = encoder_output.size(1)

		# create placeholder for net's output
		final_output = Variable(torch.zeros(self.seq_len, batch_size, self.output_size)).cuda()

		context = self.shrink(encoder_output.contiguous().view(-1, encoder_output.size(-1)))
		dim = context.size(-1)
		context = context.view(self.seq_len, batch_size, -1)
		context = context.transpose(0,2).transpose(0,1).transpose(1,2)
		context = context.contiguous().view(-1, self.seq_len * dim)

		for i in range(self.seq_len):
			decoder_output = self.gru(context, decoder_output)
			final_output[i] = decoder_output

		return F.softmax(final_output)
