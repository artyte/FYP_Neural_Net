import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as packpad, pad_packed_sequence as padpack

class Encoder(nn.Module):
    def __init__(self, embeddings, hidden_size):
        super(Encoder, self).__init__()

        embed_size = embeddings.size(1)
        self.embedding = nn.Embedding(embeddings.size(0), embed_size)
        self.embedding.weight = nn.Parameter(embeddings)
        self.embedding.weight.requires_grad = False
        self.gru = nn.GRU(embed_size, hidden_size, bidirectional=True)

    def forward(self, input, input_length, hidden):
        embed = self.embedding(input)
        packed = packpad(embedded, input_length)
        output, hidden = self.gru(packed, hidden)
        output, _ = padpack(output)

        return output, hidden

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size=1):
        super(Decoder, self).__init__()

        self.output_size = output_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))
        self.gru = nn.GRUCell(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size * 3, hidden_size)
        self.final = nn.Linear(hidden_size, self.output_size)

    def forward(self, encoder_output, hidden, decoder_output):
        batch_size = encoder_outputs.size(1)
        seq_len = encoder_outputs.size(2)

        # create placeholder for net's output
        final_output = Variable(torch.zeros(seq_len, batch_size, self.output_size))

        for i in range(seq_len):
            # initialize attention weights' parameters
            attn_energy = Variable(torch.zeros(batch_size, seq_len))

            # hidden and encoder_output axis: S -> 1 x S (suitable for Linear's api definition)
            # concat axis : 1 x 2*S
            for b in range(batch_size):
                for l in range(seq_len):
                    # v tensor used to reduce attention output to 1 dim
                    attn_energy[b, l] = self.v.dot(self.attn(torch.cat((hidden[b,:].unsqueeze(0), encoder_output[l,b].unsqueeze(0)), 1)))
            attn_energy = F.softmax(attn_energy)

            # encoder_output axis: S x B x D -> B x S x D (to match attn_energy's B x 1 x S)
            context = torch.bmm(attn_energy.unsqueeze(1), encoder_output.transpose(0,1))

            # context axis: B x 1 x S -> B x S (suitable for GRUCell's api definition)
            context = context.squeeze(1)

            # concat axis: B x 2*S
            hidden = self.gru(torch.cat((decoder_output, context), 1), hidden)

            # concat axis : B x 3*S
            decoder_output = self.out(torch.cat((hidden, decoder_output, context), 1))
            decoder_output = F.softmax(decoder_output)

            # used different variable name so as to maintain decoder_output's shape for next iteration
            output = self.final(decoder_output)
            output = F.softmax(output)

            final_output[i] = output

        return final_output
