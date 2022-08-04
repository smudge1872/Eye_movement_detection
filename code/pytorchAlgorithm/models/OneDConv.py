import torch
from torch.nn import Module
from torch.nn import  Linear, Conv1d, MaxPool1d, Dropout
from torch.nn import functional as F
from torch.autograd import Variable
import pdb


class OneDConv(Module):

    def __init__(self, num_features, seq_len, num_labels=2,device='cpu'):
        super(OneDConv, self).__init__()
        self.device = device

        #architecture from https://www.mdpi.com/1660-4601/16/4/599
        # A Deep Learning Model for Automated Sleep StagesClassification Using PSG Signals by Yildririm et.al.
        self.layers = torch.nn.Sequential(
            Conv1d(num_features, 64, 5, stride=3),
            torch.nn.ReLU(),
            Conv1d(64, 128, 5, stride=1),
            torch.nn.ReLU(),
            MaxPool1d(2, stride=2),
            Dropout(p=0.2),
            Conv1d(128, 128, 13, stride=1),
            torch.nn.ReLU(),
            Conv1d(128, 256, 7, stride=1),
            torch.nn.ReLU(),
            MaxPool1d(2, stride=2),
            Conv1d(256, 256, 7, stride=1,padding=1), #adding padding  for 200 sequence
            torch.nn.ReLU(),
            Conv1d(256,64,4, stride=1,padding=2 ),  #adding padding for 200 sequence
            torch.nn.ReLU(),
            MaxPool1d(2, stride=2),
            Conv1d(64,32,3,stride=1, padding=1), #adding padding for 200 sequence
            torch.nn.ReLU(),
            Conv1d(32,64,6,stride=1,padding=3), #adding padding for 200 sequence
            torch.nn.ReLU(),
            MaxPool1d(2,stride=2),
            Conv1d(64,8,5,stride=1,padding=2), #adding padding for 200 sequence
            torch.nn.ReLU(),
            Conv1d(8,8,2,stride=1, padding=1), #adding padding for 200 sequence
            torch.nn.ReLU(),
            MaxPool1d(2, stride=2),
            torch.nn.Flatten(),
            Linear(8,64),
            torch.nn.ReLU(),
            Dropout(p=0.2),
            Linear(64,num_labels),
            torch.nn.Softmax(dim=1)


        )

    def forward(self, x):

        permuted = x.permute((0,2,1))
        return self.layers(permuted)

def main():
    con1d = OneDConv(9, 200,2)
    input = torch.randn(1,200,9)
    #input = input.permute( (0,2,1))
    test = con1d.forward(input)
    print('Hello')
    print(test)
    print(test.size())

if __name__ == "__main__":
    main()