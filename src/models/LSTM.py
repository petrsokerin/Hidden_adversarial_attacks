import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, hidden_dim, n_layers, x_dim=1, output_dim=1, dropout=0.2):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.rnn = nn.LSTM(
            x_dim, 
            hidden_dim, 
            num_layers=n_layers, 
            dropout=dropout,
            batch_first=True
        )
        
        self.fc1 = nn.Linear(hidden_dim * n_layers, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        
        
    def forward(self, data, use_sigmoid=True, use_tanh=False):
        
        output, (hidden, cell) = self.rnn(data)

        # hidden have shape n_layers, n_onjects, hidden_dim
        n_objects = hidden.shape[1]

        hidden = hidden.transpose(1, 0)  # shape n_onjects, n_layers, hidden_dim

        hidden = hidden.reshape(
            n_objects, 
            self.n_layers * self.hidden_dim,
        )  # shape n_onjects, n_layers * hidden_dim
        
        hidden = self.dropout(hidden)
        output = self.relu(self.fc1(hidden))
        output = self.fc2(self.dropout(output))

        if use_sigmoid:
            output = torch.sigmoid(output)
        elif use_tanh:
            output = self.tanh(output)
            
        return output