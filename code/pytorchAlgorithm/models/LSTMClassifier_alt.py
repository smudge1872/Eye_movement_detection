import torch
from torch.nn import Module
from torch.nn import LSTM, Linear
from torch.nn import functional as F
from torch.autograd import Variable
import pdb


class LSTMClassifier_alt(Module):

    def __init__(self, num_features, batch_size, seq_len, num_layers=1, layer_size=128, num_labels=2, device='cpu'):
        super(LSTMClassifier_alt, self).__init__()
        self.device = device        
        self.layer_size = layer_size
        self.num_labels = num_labels
        self.num_layers = num_layers        
        
        self.lstm = LSTM(num_features, layer_size, num_layers, batch_first=True)
        self.fc1 = Linear(layer_size * seq_len, 256)
        #self.fc1 = Linear(layer_size, 32)        
        self.fc2 = Linear(256, num_labels)
        self.relu = torch.nn.ReLU() 
        self.sig = torch.nn.Sigmoid()
        self.smax = torch.nn.Softmax(dim=1)
        self.ddtype = float
        
        # self._hidden_cell = None
        # self.clear_hidden_cell(batch_size)


    def forward(self, x):
        # self.clear_hidden_cell(x.shape[0])
        h_0 = torch.zeros((self.num_layers, x.size(0), self.layer_size), dtype=self.ddtype, device=self.device).float()     
        c_0 = torch.zeros((self.num_layers, x.size(0), self.layer_size), dtype=self.ddtype, device=self.device).float()
        hidden = (h_0, c_0)   
        
        output, (h_0, c_0) = self.lstm(x, hidden)
        
        
        #pdb.set_trace()
        
        #output = output.contiguous().view(-1, self.layer_size)
        output = output.contiguous().view(x.size(0), -1)
        #print(output.shape)
        
        output = self.relu(output)
        output = self.fc1(output)
        #print(output.shape)
        output = self.relu(output)
        output = self.fc2(output)
        
        output = F.softmax(output, dim=1)
        # output = self.sig(output)
        #print(output.shape)
        #print(output)
        
        #output = output.view(x.size(0), -1)
        #print(output.shape)
        
        #output = output[:, -1]
        #print(output.shape)
        
        
        #quit()
        
        # h_0 = h_0.contiguous().view(x.shape[0], -1)
        # #out = self.relu(h_0.float())
        # #output = output.view(-1, self.layer_size)
        # #out = self.relu(output.float())
        # out = self.fc1(out.float())
        # #out = self.relu(out.float())
        # #out = self.fc2(out.float())
        # #out = F.sigmoid(out.float(), dim=1)
        # out = self.smax(out.float())
        # #out = self.sig(out.float())
        return output
        
        
    # def clear_hidden_cell(self, batch_sz):
    #    self._hidden_cell = (torch.zeros(self.num_layers, batch_sz, self.layer_size).to(self.device),
    #                         torch.zeros(self.num_layers, batch_sz, self.layer_size).to(self.device))
