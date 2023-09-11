import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sktime.datasets import load_from_tsfile

def load_Ford_A(path_train: str ="data/Ford_A/FordA_TRAIN.ts", 
                path_test: str ="data/Ford_A/FordA_TEST.ts"):
    
    root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"
    X_train, y_train = readucr(root_url + "FordA_TRAIN.tsv")
    X_test, y_test = readucr(root_url + "FordA_TEST.tsv")
    return X_train, X_test, y_train, y_test

def readucr(filename):
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)

def transform_data(X_train, X_test, y_train, y_test, slice_data=True, window=50):
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])

    if slice_data:
        len_seq = X_train.shape[1]
        n_patches = len_seq//window

        X_train = np.vstack([X_train[:, i:i+window] for i in range(n_patches)])
        X_test = np.vstack([X_test[:, i:i+window] for i in range(n_patches)])

        y_train = np.array([(int(y)+1) // 2 for y in y_train])
        y_test = np.array([(int(y)+1) // 2 for y in y_test])

        y_train = np.vstack([y_train.reshape(-1, 1) for i in range(n_patches)])
        y_test = np.vstack([y_test.reshape(-1, 1) for i in range(n_patches)])

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor =torch.tensor(X_test, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train, dtype=torch.int32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.int32)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

def build_dataloaders(X_train, X_test, y_train, y_test, batch_size=64):

    print(X_train.shape)
    train_loader = DataLoader(MyDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(MyDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

class MyDataset(Dataset):
    def __init__(self, X, y, window=50):
        super().__init__()
        self.X = X
        self.y = y
        self.window=window
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        X = torch.tensor(self.X[idx], dtype=torch.float32)

        X = X.reshape([-1, 1])
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return X, y
