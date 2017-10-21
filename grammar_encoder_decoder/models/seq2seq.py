import torch.nn as nn
from encoder import Encoder
from os.path import join
from convenient_pickle import pickle_return
log_short = pickle_return(join("data", "log_short.p"))
log_long = pickle_return(join("data", "log_long.p"))

class Seq2Seq(nn.Module):
	def __init__(self, embedding_size, encoder_hidden_size, decoder_hidden_size, output_size, seq_len, seq2seq_type, index_dim, option):
		super(Seq2Seq, self).__init__()

		self.seq2seq_type = seq2seq_type
		self.encoder = Encoder(embedding_size, encoder_hidden_size, output_size)
		if self.seq2seq_type == "attention":
			from decoder.attention import Attention
			self.decoder = Attention(decoder_hidden_size, index_dim, seq_len) if option == "train label" else Attention(decoder_hidden_size, output_size, seq_len)
		elif self.seq2seq_type == "vanilla":
			from decoder.vanilla import Vanilla
			self.decoder = Vanilla(decoder_hidden_size, index_dim, seq_len) if option == "train label" else Vanilla(decoder_hidden_size, output_size, seq_len)

	def forward(self, input):
		output = self.encoder(input, self.encoder.get_hidden(input.size(0)))
		if self.seq2seq_type == "vanilla":
			output = self.decoder(output, self.decoder.get_hidden(output.size(1)), self.decoder.get_output(output.size(1)))
			return output
		else:
			output, attention = self.decoder(output, self.decoder.get_hidden(output.size(1)), self.decoder.get_output(output.size(1)))
			return output, attention
