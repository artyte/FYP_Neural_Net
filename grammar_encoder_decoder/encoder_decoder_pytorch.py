import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from torch.autograd import Variable

# enter hyperparameters here
encoder_hidden_size = 100
decoder_hidden_size = 100
output_size = 7163
learning_rate = 0.01
epochs = 2
batch_size = 50
seq_len = 40
word_dim = output_size
evaluate_rate = 5 # print error per 'evaluate_rate' number of iterations

class Encoder(nn.Module):
    def __init__(self, embeddings, hidden_size):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(embeddings.size(0), embeddings.size(1))
        self.embedding.weight = nn.Parameter(embeddings.double())
        self.embedding.weight.requires_grad = False
        self.gru = nn.GRU(embeddings.size(1), self.hidden_size, bidirectional=True)

    def forward(self, input, hidden):
        # from torch.nn.utils.rnn import pack_padded_sequence as packpad, pad_packed_sequence as padpack
        embed = self.embedding(input)
        embed = embed.transpose(0,1)
        embed = embed.float()
        # packed = packpad(embedded, input_length)
        output, hidden = self.gru(embed, hidden)
        # output, _ = padpack(output)

        return output, hidden

    def get_hidden(self, batch_size, seq_len):
        return Variable(torch.zeros(2, batch_size, self.hidden_size)).cuda()

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()

        self.output_size = output_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.v = nn.Parameter(torch.FloatTensor(1, self.hidden_size))
        self.gru = nn.GRUCell(self.hidden_size * 2 + self.output_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size * 3 + self.output_size, self.output_size)

    def forward(self, encoder_output, hidden, decoder_output):
        batch_size = encoder_output.size(1)
        seq_len = encoder_output.size(0)

        # create placeholder for net's output
        final_output = Variable(torch.zeros(seq_len, batch_size, self.output_size)).cuda()

        for i in range(seq_len):
            # initialize attention weights' parameters
            attn_energy = Variable(torch.zeros(batch_size, seq_len)).cuda()

            # hidden and encoder_output axis: S -> 1 x S (suitable for Linear's api definition)
            # concat axis : 1 x 2*S
            for b in range(batch_size):
                for l in range(seq_len):
                    # v tensor used to reduce attention output to 1 dim
                    attn_energy[b, l] = self.v.dot(self.attn(torch.cat((hidden[b,:].unsqueeze(0), encoder_output[l,b].unsqueeze(0)), 1)))
            attn_energy = F.softmax(attn_energy)

            # encoder_output axis: S x B x D -> B x S x D (to match attn_energy's B x 1 x S)
            context = torch.bmm(attn_energy.unsqueeze(1), encoder_output.transpose(0,1)).cuda()

            # context axis: B x 1 x S -> B x S (suitable for GRUCell's api definition)
            context = context.squeeze(1)

            # concat axis: B x 2*S
            hidden = self.gru(torch.cat((decoder_output, context), 1), hidden)

            # concat axis : B x 3*S
            decoder_output = self.out(torch.cat((hidden, decoder_output, context), 1))
            decoder_output = F.softmax(decoder_output)

            final_output[i] = decoder_output

        return final_output

    def get_hidden(self, batch_size):
        return Variable(torch.zeros(batch_size, self.hidden_size)).cuda()

    def get_output(self, batch_size):
        return Variable(torch.zeros(batch_size, self.output_size)).cuda()

class Seq2Seq(nn.Module):
    def __init__(self, embeddings, encoder_hidden_size, decoder_hidden_size, output_size):
        super(Seq2Seq, self).__init__()

        self.encoder = Encoder(embeddings, encoder_hidden_size)
        self.decoder = Decoder(decoder_hidden_size, output_size)

    def forward(self, input):
        # pass data through these 2 layers
        output, _ = self.encoder(input, self.encoder.get_hidden(input.size(0), input.size(1)))
        output = self.decoder(output, self.decoder.get_hidden(output.size(1)), self.decoder.get_output(output.size(1)))

        return output

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

def random_batch(batch_size=batch_size, seq_len=seq_len, word_dim=word_dim):
    import random
    from keras.preprocessing.sequence import pad_sequences as ps
    # from keras.utils import to_categorical as tc

    batch = pickle_return('training_vectors_cache.p')
    final_input = []
    final_output = []
    epoch_finished = False
    for i in range(batch_size):
        if len(batch) == 0:
            epoch_finished = True
            break
        # use python list here instead of numpy array because numpy array doesn't have append
        instance = random.choice(batch)
        batch.remove(instance)
        final_input.append(instance[0])
        final_output.append(instance[1])

    # do not repeat same possibly same random on next batch call
    pickle_dump('training_vectors_cache.p', batch)

    # pad for consistent length
    final_input = Variable(torch.from_numpy(ps(final_input, maxlen=seq_len)).long())

    # pad for consistent length
    final_output = Variable(torch.from_numpy(ps(final_output, maxlen=seq_len)).long())
    return final_input, final_output, epoch_finished

def train(seq2seq, input, target, seq2seq_optimizer, criterion):
    # for each training cycle, zero the gradients out otherwise gradients will accumulate
    seq2seq.zero_grad()

    # pass data through these 2 layers
    output = seq2seq(input)
    output = output.transpose(0,1) # transposed axis : B x S x D (batch as first axis so as to iterate easily)
    target = target.cuda()

    loss = 0.0
    for i in range(target.size(0)): # target.size(0) is the batch axis
        loss += criterion(output[i], target[i])

    loss.backward()

    seq2seq_optimizer.step()

    return loss.data[0]

def evaluate(model, model_optimizer, criterion):
    import time
    loss = 0.0
    for epoch in range(epochs):
        recache() # reset temporary batch file for memory efficiency
        epoch_finished = False
        start = time.time()
        num_of_iterations = 0
        while not epoch_finished:
            input, output, epoch_finished = random_batch()
            input = input.cuda()
            loss += train(model, input, output, model_optimizer, criterion)

            num_of_iterations += 1

            if num_of_iterations % evaluate_rate == 0:
                diff = (time.time() - start) / 60.0
                print 'epoch: %d iteration: %d loss: %f duration: %f' %  (epoch + 1, num_of_iterations, loss, diff)
                loss = 0.0
                start = time.time()

    return model

def make_model():
    # net initilizations
    embeddings = torch.from_numpy(np.load(open('embeds.npy', 'rb')))
    seq2seq = Seq2Seq(embeddings, encoder_hidden_size, decoder_hidden_size, output_size)
    seq2seq.cuda()

    # initilize optimizers & loss functions here
    # don't initilize in the train function because the net can't keep track if these variables are deallocated
    seq2seq_optimizer = optim.Adam(filter(lambda p: p.requires_grad, seq2seq.parameters()), lr=learning_rate)
    criterion = nn.CrossEntropyLoss().cuda()

    seq2seq = evaluate(seq2seq, seq2seq_optimizer, criterion)
    torch.save(seq2seq, "model.model")

def predict():
    input = raw_input("Enter a sentence: ")

    from nltk.tokenize import word_tokenize as wt
    sentence = wt(data)
    sentence_tmp = []

    index_map = pickle_return('index.p')
    for word in sentence:
        word = word.lower()
        if word not in index_map: sentence_tmp.append(0)
        else: sentence_tmp.append(int(index_map[word][1]))

    # make sentence_tmp list of list to fit keras api
    from keras.preprocessing.sequence import pad_sequences as ps
    input = Variable(torch.from_numpy(ps([sentence_tmp], maxlen=seq_len)))
    input = input.cuda()

    # get output from model
    model = torch.load("model.model")
    output = model(input)
    output = output.transpose(0,1) # batch must be on the first axis
    values, indices = output.max(2)

    # convert into list for reverse mapping
    indices = indices.transpose(1,2).squeeze(0).squeeze(0).data.numpy().tolist()
    reverse_index = pickle_return('reverse_index.p')
    predict = []
    for num in indices:
        if num not in reverse_index_map: predict += '##NULL##'
        else: predict += reverse_index_map[num]
    print " ".join(predict)

make_model()
#predict()
