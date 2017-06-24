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

class Deocder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=1, dropout=0.1):
        super(Deocder, self).__init__()

        # Define layers
        self.attn = Attn('concat', hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, last_hidden, encoder_outputs):
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attn(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x 1 x N
        context = context.transpose(0, 1) # 1 x B x N

        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((word_embedded, context), 2)
        output, hidden = self.gru(rnn_input, last_hidden)

        # Final output layer
        output = output.squeeze(0) # B x N
        output = F.log_softmax(self.out(torch.cat((output, context), 1)))

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights
