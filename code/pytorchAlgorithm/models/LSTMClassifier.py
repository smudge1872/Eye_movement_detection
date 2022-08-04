import torch
from torch.nn import Module
from torch.nn import LSTM, Linear
from torch.nn import functional as F


class LSTMClassifier(Module):

    def __init__(self, num_features, seq_len, output_sizes, hidden_layer_size, device):
        super(LSTMClassifier, self).__init__()
        self._seq_len = seq_len
        self._hidden_layer_size = hidden_layer_size
        self._device = device
        self._lstm = LSTM(num_features, hidden_layer_size)

        # Use 3 linear (fully connected / dense) layers for each output (2, 3, 3 sizes for coma data)
        self._linear_movement = Linear(hidden_layer_size, output_sizes[0])
        self._linear_direction = Linear(hidden_layer_size, output_sizes[1])
        self._linear_speed = Linear(hidden_layer_size, output_sizes[2])

        self._hidden_cell = None
        self.clear_hidden_cell()


    def forward(self, x):
        lstm_out, self._hidden_cell = self._lstm(x, self._hidden_cell)
        pred_movement = F.softmax(self._linear_movement(lstm_out), dim=2)
        pred_direction = F.softmax(self._linear_direction(lstm_out), dim=2)
        pred_speed = F.softmax(self._linear_speed(lstm_out), dim=2)

        return pred_movement, pred_direction, pred_speed


    def clear_hidden_cell(self):
        self._hidden_cell = (torch.zeros(1, self._seq_len, self._hidden_layer_size).to(self._device),
                             torch.zeros(1, self._seq_len, self._hidden_layer_size).to(self._device))
