import h5py
import torch
from torch.utils.data import DataLoader, TensorDataset

def get_data():
    train_dataset = h5py.File('datasets/train_signs.h5', 'r')
    X_train = torch.tensor(train_dataset['train_set_x'][:], dtype=torch.float) / 255
    X_train = X_train.permute(0, 3, 1, 2)
    y_train = torch.tensor(train_dataset["train_set_y"][:])
    
    test_dataset = h5py.File('datasets/test_signs.h5', 'r')
    X_test = torch.tensor(test_dataset['test_set_x'][:], dtype=torch.float) / 255
    X_test = X_test.permute(0, 3, 1, 2)                                                                               
    y_test = torch.tensor(test_dataset["test_set_y"][:])
    return X_train, X_test, y_train, y_test