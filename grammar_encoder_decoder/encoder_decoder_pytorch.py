import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from torch.autograd import Variable

def pickle_return(filename):
    import pickle
    f = open(filename, 'r')
    data = pickle.load(f)
    f.close()
    return data

# enter hyperparameters here
encoder_hidden_size = 200
decoder_hidden_size = 200
embedding_size = 200
output_size = pickle_return('output_size.p')
learning_rate = 0.001
momentum = 0.9
weight_decay = 1e-4
batch_size = 53 # use a prime if possible
seq_len = 30
word_dim = output_size
loss_function = nn.CrossEntropyLoss().cuda()
evaluate_rate = 1 # print error per 'evaluate_rate' number of iterations

class Encoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, embedding_size)
        #self.embedding.weight = nn.Parameter(embeddings.double())
        self.embedding.weight.requires_grad = True
        self.gru = nn.GRU(embedding_size, self.hidden_size, bidirectional=True)

    def get_hidden(self, batch_size):
        return Variable(torch.zeros(2, batch_size, self.hidden_size)).cuda()

    def forward(self, input, hidden):
        embed = self.embedding(input)
        embed = embed.transpose(0,1)
        embed = embed.float()
        output, _ = self.gru(embed, hidden)

        return F.tanh(output)

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()

        self.output_size = output_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.v = nn.Linear(self.hidden_size, 1)
        self.shrink = nn.Linear(self.output_size, self.hidden_size)
        self.gru = nn.GRUCell(self.hidden_size * 3, self.hidden_size)
        self.out = nn.Linear(self.hidden_size * 4, self.output_size)

    def get_hidden(self, batch_size):
        return Variable(torch.zeros(batch_size, self.hidden_size)).cuda()

    def get_output(self, batch_size):
        return Variable(torch.zeros(batch_size, self.output_size)).cuda()

    def forward(self, encoder_output, hidden, decoder_output):
        batch_size = encoder_output.size(1)
        seq_len = encoder_output.size(0)

        # create placeholder for net's output
        final_output = Variable(torch.zeros(seq_len, batch_size, self.output_size)).cuda()

        for i in range(seq_len):
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

        return F.log_softmax(final_output) * -1

class Seq2Seq(nn.Module):
    def __init__(self, embedding_size, encoder_hidden_size, decoder_hidden_size, output_size):
        super(Seq2Seq, self).__init__()

        self.encoder = Encoder(embedding_size, encoder_hidden_size, output_size)
        self.decoder = Decoder(decoder_hidden_size, output_size)

    def forward(self, input):
        # pass data through these 2 layers
        output = self.encoder(input, self.encoder.get_hidden(input.size(0)))
        output = self.decoder(output, self.decoder.get_hidden(output.size(1)), self.decoder.get_output(output.size(1)))

        return output

def pickle_dump(filename, data):
    import pickle
    f = open(filename, 'w')
    pickle.dump(data, f)
    f.close()

def random_batch(batch_size=batch_size, seq_len=seq_len, word_dim=word_dim, batch=None):
    import random
    from keras.preprocessing.sequence import pad_sequences as ps

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
    epoch = 0
    while bool(open("continue_epoch.txt").readline()):
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

        epoch += 1

    return model

def make_model():
    # net initilizations
    # embeddings = torch.from_numpy(np.load(open('embeds.npy', 'rb')))
    seq2seq = Seq2Seq(embedding_size, encoder_hidden_size, decoder_hidden_size, output_size)
    seq2seq.cuda()

    # initilize optimizers & loss functions here
    # don't initilize in the train function because the net can't keep track if these variables are deallocated
    seq2seq_optimizer = optim.SGD(filter(lambda p: p.requires_grad, seq2seq.parameters()), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    criterion = loss_function

    seq2seq = evaluate(seq2seq, seq2seq_optimizer, criterion)
    torch.save(seq2seq, "model.model")

def predict():
    input = raw_input("Enter a sentence: ")

    from nltk.tokenize import word_tokenize as wt
    sentence = wt(input)
    sentence_tmp = []

    index_map = pickle_return('index.p')
    OoV = [] # out of vocabulary
    for word in sentence:
        tmp = word # in case of upper case
        word = word.lower()
        if word not in index_map:
            sentence_tmp.append(0)
            OoV.append(tmp)
        else: sentence_tmp.append(int(index_map[word][1]))

    # make sentence_tmp list of list to fit keras api
    # reverse to pad from end -> pad -> reverse -> convert to pytorch tensor
    from keras.preprocessing.sequence import pad_sequences as ps
    x = ps([i[::-1] for i in [sentence_tmp]], maxlen=seq_len).tolist()
    input = Variable(torch.from_numpy(np.array([i[::-1] for i in x])).long())
    input = input.cuda()

    corrective_set = pickle_return('indexed_corrective_set.p')
    for i in sentence_tmp:
        if i not in corrective_set: corrective_set.append(i)

    mask = Variable(torch.zeros(output_size)).cuda()
    for i in corrective_set: mask[i] = 1.0 # because log_softmax is the final activation

    # get output from model
    model = torch.load("model.model")
    output = model(input)
    output = output.transpose(0,1) # batch must be on the first axis
    for i in range(seq_len):
        output[0,i] = output[0,i] * mask
    values, indices = output.max(2)

    # convert into list for reverse mapping
    indices = indices.cpu()
    indices = indices.transpose(1,2).squeeze(0).squeeze(0).data.numpy().tolist()

    '''# assume that indices that happen more than three times are useless
    # note: this is a bit gimmicky
    count = {}
    for i in indices:
        if i not in count: count[i] = 1
        else: count[i] += 1
    count_list = [(value, key) for key, value in count.items()]
    del_key = [i[1] for i in count_list if i[0] > 3 and i[1] != 0]
    for key in del_key: indices = filter(lambda a: a != key, indices)'''

    # reverse mapping
    reverse_index = pickle_return('reverse_index.p')
    predict = []
    for num in indices:
        if num == 0 and OoV: predict.append(OoV.pop(0))
        if num != 0: predict.append(reverse_index[num])
    sentence = " ".join(predict)
    sentence = sentence[0].upper() + sentence[1:]
    print "Corrected sentence is: %s" % (sentence)


choice = raw_input("Enter an option (%s), (%s): " % ("1. Train model", "2. Correct sentence"))
if choice == "1": make_model()
elif choice == "2": predict()
