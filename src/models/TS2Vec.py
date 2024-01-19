import torch
from torch import nn
import torch.nn.functional as F

from src.models.TS2Vec_src.ts2vec import TS2Vec
from src.models.head import HeadClassifier

class TS2VecClassifier(nn.Module):
    def __init__(
        self,
        emb_size = 320, 
        input_dim = 1,
        n_layers = 3,
        n_classes = 2,
        emb_batch_size = 16,
        fine_tune = False,
        dropout = 'None', 
        dropout_ts2vec = 0.1, 
        device='cpu',
    ):
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

        if not fine_tune:
            for param in self.emd_model.parameters():
                param.requires_grad = False

        self.classifier = HeadClassifier(emb_size, output_size, n_layers=n_layers, dropout=dropout
    
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
