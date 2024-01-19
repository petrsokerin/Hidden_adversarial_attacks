import torch
from torch import nn


class LSTM_net(nn.Module):
    def __init__(self, hidden_dim, n_layers, output_dim=1, dropout=0.2):
        super().__init__()
        self.rnn = nn.LSTM(1, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           dropout=dropout,
                           batch_first=True)
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        
        
    def forward(self, data, use_sigmoid=True, use_tanh=False):
        
        packed_output, (hidden, cell) = self.rnn(data)
        hidden = hidden.reshape(hidden.shape[1], hidden.shape[2])
        
        hidden = self.dropout(hidden)
        output = self.relu(self.fc1(hidden))
        output = self.fc2(self.dropout(output))

        if use_sigmoid:
            output = torch.sigmoid(output)
        elif use_tanh:
            output = self.tanh(output)
            
        return output
    
def build_head(emb_size, out_size, n_layers=3, dropout='None'):

    if n_layers not in [1, 2, 3]:
        raise ValueError('n layers should be in [3, 2, 1]')
    if n_layers == 3:
        if dropout != 'None':
            classifier = nn.Sequential(
                nn.Linear(emb_size, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(dropout),
                nn.Linear(128, 32), nn.ReLU(),nn.BatchNorm1d(32),nn.Dropout(dropout),
                nn.Linear(32, out_size)
            )
        else:
            classifier = nn.Sequential(
                nn.Linear(emb_size, 128),nn.ReLU(),nn.BatchNorm1d(128),
                nn.Linear(128, 32), nn.ReLU(),nn.BatchNorm1d(32),
                nn.Linear(32, out_size)
            )
    elif n_layers == 2:
        if dropout != 'None':
            classifier = nn.Sequential(
                nn.Linear(emb_size, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(dropout),
                nn.Linear(64, out_size)
            )
        else:
            classifier = nn.Sequential(
                nn.Linear(emb_size, 64), nn.ReLU(), nn.BatchNorm1d(64),
                nn.Linear(64, out_size)
            )

    elif n_layers == 1:
        if dropout != 'None':
            classifier = nn.Sequential(
                nn.ReLU(), nn.BatchNorm1d(emb_size), nn.Dropout(dropout), nn.Linear(emb_size, out_size)
            )
        else:
            classifier = nn.Sequential(
                nn.ReLU(), nn.BatchNorm1d(emb_size), nn.Linear(emb_size, out_size)
            )
    return classifier
    

class HeadClassifier(nn.Module):
    def __init__(self, emb_size, out_size, n_layers=3, dropout='None'):
        super().__init__()

        self.classifier = build_head(emb_size, out_size, n_layers, dropout)
        self.sigm = nn.Sigmoid()
        
    def forward(self, x):
        out = self.classifier(x)
        return self.sigm(out)