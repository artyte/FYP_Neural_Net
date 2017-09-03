import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from decoder import Decoder
from os.path import join
from convenient_pickle import pickle_return
log_short = pickle_return(join("data", "log_short.p"))
log_long = pickle_return(join("data", "log_long.p"))

class Attention(Decoder):
	def __init__(self, hidden_size, output_size, seq_len):
		super(Attention, self).__init__(hidden_size, output_size, seq_len)

		self.attn = nn.Linear(self.hidden_size * 3, self.hidden_size)
		self.v = nn.Linear(self.hidden_size, 1)
		self.shrink = nn.Linear(self.output_size, self.hidden_size)
		self.gru = nn.GRUCell(self.hidden_size * 3, self.hidden_size)
		self.out = nn.Linear(self.hidden_size * 4, self.output_size)

	def forward(self, encoder_output, hidden, decoder_output):
		batch_size = encoder_output.size(1)

		# create placeholder for net's output
		final_output = Variable(torch.zeros(self.seq_len, batch_size, self.output_size)).cuda()

		for i in range(self.seq_len):
			# mimic keras's timedistributeddense for computing efficiency
			vector = self.attn(torch.cat((hidden.repeat(self.seq_len,1), encoder_output.contiguous().view(-1, encoder_output.size(-1))), 1))
			attn_energy = F.softmax(self.v(F.tanh(vector)).contiguous().view(-1,batch_size).transpose(0,1))

			# encoder_output axis: S x B x D -> B x S x D (to match attn_energy's B x 1 x S)
			context = torch.bmm(attn_energy.unsqueeze(1), encoder_output.transpose(0,1)).cuda()

			# context axis: B x 1 x S -> B x S (suitable for GRUCell's api definition)
			context = context.squeeze(1)

			decoder_output = self.shrink(decoder_output)

			# concat at S of B x S
			hidden = F.tanh(self.gru(torch.cat((decoder_output, context), 1), hidden))

			# concat at S of B x S
			decoder_output = self.out(torch.cat((hidden, decoder_output, context), 1))

			final_output[i] = decoder_output

		return F.softmax(final_output)
