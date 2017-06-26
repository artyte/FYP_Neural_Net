import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as packpad, pad_packed_sequence as padpack

class Encoder(nn.Module):
    def __init__(self, embeddings, hidden_size, num_layers=1, dropout=0.1):
        super(Encoder, self).__init__()

        embed_size = embeddings.size(1)
        self.embedding = nn.Embedding(embeddings.size(0), embed_size)
        self.embedding.weight = nn.Parameter(embeddings)
        self.embedding.weight.requires_grad = False
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, dropout=dropout, bidirectional=True)

    def forward(self, input, input_length, hidden=None):
        embed = self.embedding(input)
        packed = packpad(embedded, input_length)
        output, hidden = self.gru(packed, hidden)
        output, _ = padpack(output)

        return output, hidden

class Attention(nn.Module):
    def __init__():

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=1, dropout=0.1):
        super(Decoder, self).__init__()

        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size * 3, output_size)

    def forward(self, prev_output, hidden, encoder_output):
        # Calculate attention weights and apply to encoder outputs
        attn_weight = self.attn(torch.cat((hidden[-1], encoder_output), 1))
        context = torch.bmm(attn_weight.unsqueeze(0), encoder_output.unsqueeze(0))

        # Get current hidden state
        output, hidden = self.gru(torch.cat((prev_output, context), 1), prev_hidden)

        # Obtain output
        output = self.out(torch.cat((prev_output, hidden, context),1))
        output = F.log_softmax(output)

        # Get the index of the final output
        final, index = output.max(0)

        # Return final output, hidden state, and attention weights (for visualization)
        return final, index
