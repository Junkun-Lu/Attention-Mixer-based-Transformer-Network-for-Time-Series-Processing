import numpy as np 
import os
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split




def load_y_data(y_path):
    y = np.loadtxt(y_path, dtype=np.int32).reshape(-1,1)
    # change labels range from 1-6 t 0-5, this enables a sparse_categorical_crossentropy loss function
    return y - 1


def load_X_data(X_path):
    X_signal_paths = [X_path + file for file in os.listdir(X_path)]
    X_signals = [np.loadtxt(path, dtype=np.float32) for path in X_signal_paths]
    return np.transpose(np.array(X_signals), (1, 2, 0))


def UCI_HAR_load(PATH, split_rate):
# load X data
    X_train = load_X_data(os.path.join(PATH + r'/train/Inertial Signals/'))
    X_test = load_X_data(os.path.join(PATH + r'/test/Inertial Signals/'))
    # load y label
    y_train = load_y_data(os.path.join(PATH + r'/train/y_train.txt'))
    y_test = load_y_data(os.path.join(PATH + r'/test/y_test.txt'))

    X_train, X_vali, y_train, y_vali = train_test_split(X_train, y_train, test_size=split_rate, random_state=42)

    print("useful information:")
    print(f"shapes (n_samples, n_steps, n_signals) of X_train: {X_train.shape} ,X_validation: {X_vali.shape} and X_test: {X_test.shape}")

    return X_train, X_vali, X_test, y_train, y_vali, y_test


class UCI_HAR_Dataset(Dataset):
    def __init__(self, data_x, data_y ):
        self.data_x = data_x
        self.data_y = data_y
        self.data_channel = data_x.shape[2]
    
    def __getitem__(self, index):
        sample_x = self.data_x[index]
        sample_y = self.data_y[index]

        return sample_x,sample_y
    
    def __len__(self):
        return len(self.data_x)