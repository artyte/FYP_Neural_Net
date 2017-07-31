import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from torch.autograd import Variable

# enter hyperparameters heret
encoder_hidden_size = 100
decoder_hidden_size = 100
output_size = 30796
learning_rate = 0.004
momentum = 0.95
epochs = 2
batch_size = 60
seq_len = 30
word_dim = output_size
loss_function = nn.NLLLoss().cuda()
evaluate_rate = 10 # print error per 'evaluate_rate' number of iterations

class Encoder(nn.Module):
    def __init__(self, embeddings, hidden_size, batch_size):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(embeddings.size(0), embeddings.size(1))
        self.embedding.weight = nn.Parameter(embeddings.double())
        self.embedding.weight.requires_grad = True
        self.gru = nn.GRU(embeddings.size(1), self.hidden_size, bidirectional=True)

        self.hidden = Variable(torch.zeros(2, batch_size, self.hidden_size)).cuda()

    def forward(self, input):
        hidden = self.hidden
        embed = self.embedding(input)
        embed = embed.transpose(0,1)
        embed = embed.float()
        output, _ = self.gru(embed, hidden)

        return F.tanh(output)

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, batch_size):
        super(Decoder, self).__init__()

        self.output_size = output_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.v = nn.Linear(self.hidden_size, 1)
        self.shrink = nn.Linear(self.output_size, self.hidden_size)
        self.gru = nn.GRUCell(self.hidden_size * 3, self.hidden_size)
        self.out = nn.Linear(self.hidden_size * 4, self.output_size)

        self.hidden = Variable(torch.zeros(batch_size, self.hidden_size)).cuda()
        self.decoder_output = Variable(torch.zeros(batch_size, self.output_size)).cuda()

    def forward(self, encoder_output):
        hidden = self.hidden
        decoder_output = self.decoder_output

        batch_size = encoder_output.size(1)
        seq_len = encoder_output.size(0)

        # create placeholder for net's output
        final_output = Variable(torch.zeros(seq_len, batch_size, self.output_size)).cuda()

        for i in range(seq_len):
            # initialize attention weights' parameters
            attn_energy = Variable(torch.zeros(batch_size, seq_len)).cuda()

            # mimic keras's timedistributeddense
            vector = self.attn(torch.cat((hidden.repeat(seq_len,1), encoder_output.contiguous().view(-1, encoder_output.size(-1))), 1))
            attn_energy = F.softmax(self.v(F.tanh(vector)).contiguous().view(-1,batch_size).transpose(0,1))

            # encoder_output axis: S x B x D -> B x S x D (to match attn_energy's B x 1 x S)
            context = torch.bmm(attn_energy.unsqueeze(1), encoder_output.transpose(0,1)).cuda()

            # context axis: B x 1 x S -> B x S (suitable for GRUCell's api definition)
            context = context.squeeze(1)

            decoder_output = self.shrink(decoder_output)

            # concat axis: B x 2*S
            hidden = F.tanh(self.gru(torch.cat((decoder_output, context), 1), hidden))

            # concat axis : B x 3*S
            decoder_output = self.out(torch.cat((hidden, decoder_output, context), 1))
            decoder_output = decoder_output

            final_output[i] = decoder_output

        return F.log_softmax(final_output)

class Seq2Seq(nn.Module):
    def __init__(self, embeddings, encoder_hidden_size, decoder_hidden_size, output_size, batch_size):
        super(Seq2Seq, self).__init__()

        self.encoder = Encoder(embeddings, encoder_hidden_size, batch_size)
        self.decoder = Decoder(decoder_hidden_size, output_size, batch_size)

    def forward(self, input):
        # pass data through these 2 layers
        output = self.encoder(input)
        output = self.decoder(output)

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

def random_batch(batch_size=batch_size, seq_len=seq_len, word_dim=word_dim, batch=None):
    import random
    from keras.preprocessing.sequence import pad_sequences as ps

    # batch = pickle_return('training_vectors_cache.p')
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


    # reverse to pad from end -> pad -> reverse -> convert to pytorch tensor
    x = ps([i[::-1] for i in final_input], maxlen=seq_len).tolist()
    final_input = Variable(torch.from_numpy(np.array([i[::-1] for i in x])).long())
    y = ps([i[::-1] for i in final_output], maxlen=seq_len).tolist()
    final_output = Variable(torch.from_numpy(np.array([i[::-1] for i in y])).long())

    # do not repeat same possibly same random on next batch call
    # pickle_dump('training_vectors_cache.p', batch)

    return final_input, final_output, epoch_finished, batch

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
    torch.nn.utils.clip_grad_norm(seq2seq.parameters(), 5)
    seq2seq_optimizer.step()

    return loss.data[0]

def evaluate(model, model_optimizer, criterion):
    import time
    loss = 0.0
    for epoch in range(epochs):
        start = time.time()

        data = pickle_return('training_vectors.p') # reset temporary batch data for memory efficiency
        epoch_finished = False
        num_of_iterations = 0
        while not epoch_finished:
            input, output, epoch_finished, data = random_batch(batch=data)
            input = input.cuda()
            loss += train(model, input, output, model_optimizer, criterion)

            num_of_iterations += 1

            if num_of_iterations % evaluate_rate == 0:
                diff = (time.time() - start) / 60.0
                print 'epoch: %d\titeration: %d\tloss: %f\tduration: %f' %  (epoch + 1, num_of_iterations, loss, diff)
                loss = 0.0
                start = time.time()

    return model

def make_model():
    # net initilizations
    embeddings = torch.from_numpy(np.load(open('embeds.npy', 'rb')))
    seq2seq = Seq2Seq(embeddings, encoder_hidden_size, decoder_hidden_size, output_size, batch_size)
    seq2seq.cuda()

    # initilize optimizers & loss functions here
    # don't initilize in the train function because the net can't keep track if these variables are deallocated
    seq2seq_optimizer = optim.SGD(filter(lambda p: p.requires_grad, seq2seq.parameters()), lr=learning_rate, momentum=momentum)
    criterion = loss_function

    seq2seq = evaluate(seq2seq, seq2seq_optimizer, criterion)
    torch.save(seq2seq, "model.model")

def predict():
    input = raw_input("Enter a sentence: ")

    from nltk.tokenize import word_tokenize as wt
    sentence = wt(input)
    sentence_tmp = []

    index_map = pickle_return('index.p')
    for word in sentence:
        word = word.lower()
        if word not in index_map: sentence_tmp.append(0)
        else: sentence_tmp.append(int(index_map[word][1]))

    # make sentence_tmp list of list to fit keras api
    from keras.preprocessing.sequence import pad_sequences as ps
    input = Variable(torch.from_numpy(ps([sentence_tmp], maxlen=seq_len)).long())
    input = input.cuda()

    # get output from model
    model = torch.load("model.model")
    output = model(input)
    output = output.transpose(0,1) # batch must be on the first axis
    values, indices = output.max(2)

    # convert into list for reverse mapping
    indices = indices.cpu()
    indices = indices.transpose(1,2).squeeze(0).squeeze(0).data.numpy().tolist()
    reverse_index = pickle_return('reverse_index.p')
    predict = []
    for num in indices:
        if num != 0: predict.append(reverse_index[num])
    print "Corrected sentence is %s" % (" ".join(predict))

choice = raw_input("Enter an option:\n%s\n%s\n" % ("1. Train model", "2. Correct sentence"))
if choice == "1": make_model()
elif choice == "2": predict()
