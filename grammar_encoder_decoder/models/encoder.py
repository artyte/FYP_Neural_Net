import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from os.path import join
from convenient_pickle import pickle_return
log_short = pickle_return(join("data", "log_short.p"))
log_long = pickle_return(join("data", "log_long.p"))

class Encoder(nn.Module):
	def __init__(self, embedding_size, hidden_size, output_size):
		super(Encoder, self).__init__()

		self.hidden_size = hidden_size
		self.embedding = nn.Embedding(output_size, embedding_size)
		self.embedding.weight.requires_grad = True
		self.gru = nn.GRU(embedding_size, self.hidden_size, bidirectional=True)

	def get_hidden(self, batch_size):
		return Variable(torch.zeros(2, batch_size, self.hidden_size)).cuda()

	def forward(self, input, hidden):
		embed = self.embedding(input)
		embed = embed.transpose(0,1)
		embed = embed.float() # without float, net cannot multiply properly later
		output, _ = self.gru(embed, hidden)

		return F.tanh(output)
