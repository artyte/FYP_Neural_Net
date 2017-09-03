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

		self.shrink = nn.Linear(self.seq_len * 2 * self.hidden_size, self.hidden_size)
		self.gru = nn.GRU(self.hidden_size, self.output_size)

	def forward(self, encoder_output, hidden, decoder_output):
		encoder_output = encoder_output.transpose(0,1).contiguous()
		encoder_output = encoder_output.view(encoder_output.size(0),-1)
		context = self.shrink(encoder_output)
		final_output, _ = self.gru(context.repeat(self.seq_len,1,1), decoder_output.unsqueeze(0))

		return F.softmax(final_output)
