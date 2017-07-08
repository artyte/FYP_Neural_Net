import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

class Encoder(nn.Module):
    def __init__(self, embeddings, hidden_size):
        super(Encoder, self).__init__()

        embed_size = embeddings.size(1)
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(embeddings.size(0), embed_size)
        self.embedding.weight = nn.Parameter(torch.from_numpy(embeddings).double())
        self.embedding.weight.requires_grad = False
        self.gru = nn.GRU(embed_size, self.hidden_size, bidirectional=True)

    def forward(self, input, input_length, hidden):
        from torch.nn.utils.rnn import pack_padded_sequence as packpad, pad_packed_sequence as padpack
        embed = self.embedding(input)
        packed = packpad(embedded, input_length)
        output, hidden = self.gru(packed, hidden)
        output, _ = padpack(output)

        return output, hidden

    def set_weights():
        return Variable(torch.zeros(1, 1, self.hidden_size)).cuda()

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()

        self.output_size = output_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.v = nn.Parameter(torch.FloatTensor(1, self.hidden_size))
        self.gru = nn.GRUCell(self.hidden_size * 2, self.hidden_size)
        self.out = nn.Linear(self.hidden_size * 3, self.output_size)

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

            final_output[i] = decoder_output

        return final_output

    def set_weights():
        return Variable(torch.zeros(1, 1, self.hidden_size)).cuda()

def pickle_return(filename):
	import pickle
	f = open(filename, 'r')
	data = pickle.load(f)
	f.close()
	return data

def pickle_dump(filename, data):
	import pickle
	f = open(filename, 'w')
	pickle.dump(data, f)
	f.close()

def recache():
    batch = pickle_return('training_vectors.p')
    pickle_dump('training_vectors_cache.p', batch)

def random_batch(num_of_batch=50, max_len=300, word_dim=10237):
    import numpy as np
    import random
    from keras.preprocessing.sequence import pad_sequences as ps
    from keras.utils import to_categorical as tc

    batch = pickle_return('training_vectors_cache.p')
    final_input = []
    final_output = []
    for i in range(num_of_batch):
        if len(batch) == 0: break
        # use python list here instead of numpy array because numpy array doesn't have append
        instance = random.choice(batch)
        batch.remove(instance)
        final_input.append(instance[0])
        final_output.append(instance[1])

    # do not repeat same possibly same random on next batch call
    pickle_dump('training_vectors_cache.p', batch)

    # pad for consistent length
    final_input = torch.from_numpy(ps(final_input, maxlen=max_len))

    # 3 ops: pad for consistent length -> turn into one hot for easy evaluation -> reshape since keras's to_categorical doesn't
    final_output = torch.from_numpy(np.reshape(tc(ps(final_output, maxlen=max_len), num_classes=word_dim), (max_len,-1,word_dim)))
    return final_input, final_output
