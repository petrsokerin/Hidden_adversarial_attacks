import os
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset, DataLoader
from tsai.data.core import TSTensor


def load_data(dataset: str = 'Ford_A_LSTM'):
    if dataset == 'Ford_A_LSTM':
        X_train, X_test, y_train, y_test = load_Ford_A()
    else:
        X_train, y_train, X_test, y_test = load_UCR(dataset)


def load_data(dataset:str = 'FordA'):
    X_train, y_train, X_test, y_test = load_UCR(dataset)
    return X_train, y_train, X_test, y_test


def load_UCR(dataset):
    train_file = os.path.join('data/TS2Vec/UCR', dataset, dataset + "_TRAIN.tsv")
    test_file = os.path.join('data/TS2Vec/UCR', dataset, dataset + "_TEST.tsv")
    train_df = pd.read_csv(train_file, sep='\t', header=None)
    test_df = pd.read_csv(test_file, sep='\t', header=None)
    train_array = np.array(train_df)
    test_array = np.array(test_df)

    # Move the labels to {0, ..., L-1}
    labels = np.unique(train_array[:, 0])
    transform = {}
    for i, l in enumerate(labels):
        transform[l] = i

    train = train_array[:, 1:].astype(np.float64)
    train_labels = np.vectorize(transform.get)(train_array[:, 0])
    test = test_array[:, 1:].astype(np.float64)
    test_labels = np.vectorize(transform.get)(test_array[:, 0])

    # Normalization for non-normalized datasets
    # To keep the amplitude information, we do not normalize values over
    # individual time series, but on the whole dataset
    if dataset not in [
        'AllGestureWiimoteX',
        'AllGestureWiimoteY',
        'AllGestureWiimoteZ',
        'BME',
        'Chinatown',
        'Crop',
        'EOGHorizontalSignal',
        'EOGVerticalSignal',
        'Fungi',
        'GestureMidAirD1',
        'GestureMidAirD2',
        'GestureMidAirD3',
        'GesturePebbleZ1',
        'GesturePebbleZ2',
        'GunPointAgeSpan',
        'GunPointMaleVersusFemale',
        'GunPointOldVersusYoung',
        'HouseTwenty',
        'InsectEPGRegularTrain',
        'InsectEPGSmallTrain',
        'MelbournePedestrian',
        'PickupGestureWiimoteZ',
        'PigAirwayPressure',
        'PigArtPressure',
        'PigCVP',
        'PLAID',
        'PowerCons',
        'Rock',
        'SemgHandGenderCh2',
        'SemgHandMovementCh2',
        'SemgHandSubjectCh2',
        'ShakeGestureWiimoteZ',
        'SmoothSubspace',
        'UMD'
    ]:
        return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels

    mean = np.nanmean(train)
    std = np.nanstd(train)
    train = (train - mean) / std
    test = (test - mean) / std
    return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels


def load_UEA(dataset):
    train_data = loadarff(f'data/TS2Vec/UEA/{dataset}/{dataset}_TRAIN.arff')[0]
    test_data = loadarff(f'data/TS2Vec/UEA/{dataset}/{dataset}_TEST.arff')[0]

    def extract_data(data):
        res_data = []
        res_labels = []
        for t_data, t_label in data:
            t_data = np.array([d.tolist() for d in t_data])
            t_label = t_label.decode("utf-8")
            res_data.append(t_data)
            res_labels.append(t_label)
        return np.array(res_data).swapaxes(1, 2), np.array(res_labels)

    train_X, train_y = extract_data(train_data)
    test_X, test_y = extract_data(test_data)

    scaler = StandardScaler()
    scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    return train_X, train_y, test_X, test_y


def load_Ford_A(
        path_train: str = "data/Ford_A/FordA_TRAIN.ts",
        path_test: str = "data/Ford_A/FordA_TEST.ts"
):
    root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"
    X_train, y_train = readucr(root_url + "FordA_TRAIN.tsv")
    X_test, y_test = readucr(root_url + "FordA_TEST.tsv")
    return X_train, X_test, y_train, y_test


def readucr(filename):
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)


def transform_data(
        X_train,
        X_test,
        y_train,
        y_test,
        slice_data = True,
        window = 50,
    ):
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1])

    # transform from -1,1 labels to 0,1.
    if len(np.unique(y_train)) == 2 and np.sum(y_train == -1) > 0:
        y_train = (y_train + 1) // 2 
        y_test = (y_test + 1) // 2 

    if slice_data:
        len_seq = X_train.shape[1]
        n_patches = len_seq // window

        X_train = np.vstack([X_train[:, i:i + window] for i in range(n_patches)])
        X_test = np.vstack([X_test[:, i:i + window] for i in range(n_patches)])

        y_train = np.concatenate([y_train for _ in range(n_patches)])
        y_test = np.concatenate([y_test for _ in range(n_patches)])

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train, dtype=torch.int32).unsqueeze(dim=1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.int32).unsqueeze(dim=1)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor


def build_dataloaders(X_train, X_test, y_train, y_test, batch_size=64):
    train_loader = DataLoader(MyDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(MyDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

class Augmentator:
    def __init__(self, methods):
        if isinstance(methods, Iterable):
            self.methods = methods
        else:
            self.methods = [methods]

    def __call__(self, x):
        for method in self.methods:
            x = TSTensor(x.unsqueeze(0).transpose(1, 2))
            x = method.encodes(x).data.transpose(1, 2).squeeze(0)
        return x

class MyDataset(Dataset):
    def __init__(self, X, y, window=50, transform=None):
        super().__init__()
        self.X = X
        self.y = y
        self.window=window
        self.transform = transform
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X = torch.tensor(self.X[idx], dtype=torch.float32)

        X = X.reshape([-1, 1])
        y = torch.tensor(self.y[idx], dtype=torch.float32)

        if self.transform:
            X = self.transform(X)

        return X, y
