import gin
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@gin.configurable
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        self.embed = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=0.1, bidirectional=bidirectional)
        self.o2p = nn.Linear(hidden_size, output_size * 2)
        
    def sample(self, mu, logvar):
        eps = Variable(torch.randn(mu.size())).to(device)
        std = torch.exp(logvar / 2.0)
        return mu + eps * std

    def forward(self, input):
        embedded = self.embed(input)# .unsqueeze(1)

        output, hidden = self.gru(embedded, None)
        # mean loses positional info?
        output = output[-1]
        if self.bidirectional:
            output = output[:, :self.hidden_size] + output[: ,self.hidden_size:] # Sum bidirectional outputs
        else:
            output = output[:, :self.hidden_size]

        ps = self.o2p(output)
        mu, logvar = torch.chunk(ps, 2, dim=1)
        z = self.sample(mu, logvar)
        return mu, logvar, z


@gin.configurable
class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, sos_token='<s>', max_sample=True):
        super(DecoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.SOS_TOKEN = sos_token
        self.MAX_SAMPLE = max_sample

        self.embed = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size + input_size, hidden_size, n_layers)
        self.z2h = nn.Linear(input_size, hidden_size)
        self.i2h = nn.Linear(hidden_size + input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size + input_size, output_size)

    def sample(self, output, temperature):
        if self.MAX_SAMPLE:
            # Sample top value only
            top_i = output.data.topk(1)[1].squeeze(1)
        else:
            # Sample from the network as a multinomial distribution
            output_dist = output.data.view(-1).div(temperature).exp()
            top_i = torch.multinomial(output_dist, 1)[0]

        input = Variable(top_i).to(device)
        return input, top_i

    def forward(self, z, inputs, temperature):
        n_steps = inputs.size(0)
        outputs = Variable(torch.zeros(n_steps, z.size(0), self.output_size)).to(device)

        input = Variable(torch.LongTensor([self.SOS_TOKEN])).to(device)
        input = input.repeat(z.size(0))
        hidden = self.z2h(z).unsqueeze(0).repeat(self.n_layers, 1, 1)

        for i in range(n_steps):
            output, hidden = self.step(i, z, input, hidden)
            outputs[i] = output
            use_teacher_forcing = np.random.randn() < temperature
            if use_teacher_forcing:
                input = inputs[i]
            else:
                input, top_i = self.sample(output, temperature)

        return outputs.transpose(0,1)

    def generate(self, z, n_steps, temperature):
        outputs = Variable(torch.zeros(n_steps, z.size(0), self.output_size)).to(device)

        input = Variable(torch.LongTensor([self.SOS_TOKEN])).to(device)
        input = input.repeat(z.size(0))
        hidden = self.z2h(z).unsqueeze(0).repeat(self.n_layers, 1, 1)

        for i in range(n_steps):
            output, hidden = self.step(i, z, input, hidden)
            outputs[i] = output
            input, top_i = self.sample(output, temperature)
            
        return outputs.transpose(0,1)

    def step(self, s, z, input, hidden):
        input = F.relu(self.embed(input))
        input = torch.cat((input, z), 1)
        input = input.unsqueeze(0)
        output, hidden = self.gru(input, hidden)
        output = output.squeeze(0)
        output = torch.cat((output, z), 1)
        output = self.out(output)
        return output, hidden


@gin.configurable
class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.steps_seen = 0

    def encode(self, inputs):
        m, l, z = self.encoder(inputs)
        return m, l, z

    def forward(self, inputs, temperature=1.0):
        m, l, z = self.encoder(inputs)
        decoded = self.decoder(z, inputs, temperature)
        return m, l, z, decoded
    