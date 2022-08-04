import torch
from torch.nn import Module
from torch.nn import LSTM, Linear
from torch.nn import functional as F


class LSTMClassifierCombinedOutputs(Module):

    def __init__(self, num_features, seq_len, hidden_layer_size, num_labels, device):
        super(LSTMClassifierCombinedOutputs, self).__init__()
        self._seq_len = seq_len
        self._hidden_layer_size = hidden_layer_size
        self._num_labels = num_labels
        self._device = device
        self._lstm = LSTM(num_features, hidden_layer_size)

        # Use 1 linear (fully connected / dense) layer with num_labels # of classes
        self._linear = Linear(hidden_layer_size, self._num_labels)

        self._hidden_cell = None
        self.clear_hidden_cell()


    def forward(self, x):
        lstm_out, self._hidden_cell = self._lstm(x, self._hidden_cell)
        pred = F.softmax(self._linear(lstm_out), dim=2)
        return pred


    def clear_hidden_cell(self):
        self._hidden_cell = (torch.zeros(1, self._seq_len, self._hidden_layer_size).to(self._device),
                             torch.zeros(1, self._seq_len, self._hidden_layer_size).to(self._device))
