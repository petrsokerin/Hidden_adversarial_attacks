import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from utils.TS2Vec.ts2vec import TS2Vec


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
    
    
    
class TS2VecClassifier(nn.Module):
    def __init__(self, emb_size=320, input_dim=1, n_layers=3,  n_classes=2, emb_batch_size=16, dropout='None', dropout_ts2vec=0.1, device='cpu'):
        super().__init__()
        
        if n_classes == 2:
            output_size = 1
        else:
            output_size = n_classes

        self.ts2vec = TS2Vec(
        input_dims=input_dim,
        dropout=dropout_ts2vec,
        device=device,
        output_dims=emb_size,
        batch_size=emb_batch_size
        )
        
        self.emd_model = self.ts2vec.net

        self.classifier = HeadClassifier(emb_size, output_size, n_layers=n_layers, dropout='None')
    
    def train_embedding(self, X_train, verbose=False):
        self.ts2vec.fit(X_train, verbose=verbose)
        self.emd_model = self.ts2vec.net
        
    def forward(self, X, mask=None):
        
            emb_out = self.emd_model(X, mask)
            emb_out = F.max_pool1d(emb_out.transpose(1, 2), kernel_size = emb_out.size(1))
            emb_out = emb_out.transpose(1, 2).squeeze(1)
            out = self.classifier(emb_out)
            
            return out
        
    def load_old(self, path_emb, path_head):
        self.emd_model.load_state_dict(torch.load(path_emb))
        self.classifier.load_state_dict(torch.load(path_head))

    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path))


    