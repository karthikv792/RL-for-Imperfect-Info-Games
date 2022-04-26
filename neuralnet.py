import numpy as np
from tqdm import tqdm
import sys
import os
sys.path.append('../../')

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

class AverageMeter(object):
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class NNetSequence(nn.Module):
    def __init__(self, lr=0.001,dropout=0.3, batch_size=64, epochs=10, momentum=0.99):
        self.lr = lr
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.momentum = momentum
        self.state_size = 700
        self.action_size = 300

        super(NNetSequence, self).__init__()
        self.fc1 = nn.Linear(self.state_size, 512)
        self.fc2 = nn.Linear(512, 400)
        self.fc3 = nn.Linear(400,self.action_size)
        self.fc4 = nn.Linear(400,1)

    def forward(self, x):
        x = x.view(-1, self.state_size)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout)
        pi = self.fc3(x)
        v = self.fc4(x)
        return F.log_softmax(pi), torch.tanh(v)
class NeuralNet():
    def __init__(self):
        self.nnet = NNetSequence()
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.nnet.cuda()

    def train(self, examples):
        print("Training", len(examples))
        optimizer = optim.Adam(self.nnet.parameters(), lr=self.nnet.lr)

        for epoch in range(self.nnet.epochs):
            # print('Epoch: {}'.format(epoch))
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            batch_count = int(len(examples) / self.nnet.batch_size)
            # print('Batch count: {}'.format(batch_count))
            t = tqdm(range(batch_count), desc='Training Net')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=self.nnet.batch_size)
                states,pis,vs = list(zip(*[examples[j] for j in sample_ids]))
                states = torch.FloatTensor(np.array(states).astype(np.float32))
                pis = torch.FloatTensor(pis)
                vs = torch.FloatTensor(np.array(vs).astype(np.float32))

                if self.cuda:
                    states, pis, vs = states.cuda(), pis.cuda(), vs.cuda()

                out_pi, out_v = self.nnet(states)
                l_pi = -torch.sum(pis* out_pi) / pis.size()[0]
                l_v = torch.sum((vs - out_v.squeeze())**2) / vs.size()[0]
                total_loss = l_pi + l_v

                pi_losses.update(l_pi.item(), pis.size()[0])
                v_losses.update(l_v.item(), vs.size()[0])
                t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    def predict(self, state):
        state = torch.FloatTensor(state.astype(np.float32))
        if self.cuda:
            state = state.cuda()
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(state)

        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def save(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print("Checkpoint Directory does not exist! Making directory {}".format(folder))
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save({
            'state_dict': self.nnet.state_dict(),
        }, filepath)

    def load(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise("No model in path {}".format(filepath))
        map_location = None if self.cuda else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])