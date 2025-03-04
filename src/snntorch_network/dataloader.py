from typing import Any
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

class WisdmEncodedDatasetParser():
    def __init__(self, file_name):
        self.file_name = file_name
        (x_train, x_val, x_test, y_train, y_val, y_test) = self.load_wisdm2_data(file_name)
        
        self.train_dataset = (x_train, y_train)
        self.val_dataset = (x_val,y_val)
        self.test_dataset = (x_test,y_test)

        
        print(f'num classes train dataset: {self.train_dataset[1].max()+1} occurrences of each class:{np.bincount(self.train_dataset[1])}')
        print(f'num classes eval dataset: {self.val_dataset[1].max()+1} occurrences of each class:{np.bincount(self.val_dataset[1])}')
        print(f'num classes test dataset: {self.test_dataset[1].max()+1} occurrences of each class:{np.bincount(self.test_dataset[1])}')

    def get_training_set(self, subset=None, shuffle=True):
        
        if subset:
            N = self.test_dataset[0].shape[0]

            if shuffle:
                ids = np.array(range(0, N))
                np.random.shuffle(ids)
                ids = ids[:subset]

            else:
                ids = np.array(range(0, subset))
                
            return np.array(self.train_dataset[0][ids]), np.array(self.train_dataset[1][ids])
        return self.train_dataset

    def get_validation_set(self, subset=None, shuffle=True):
        
        if subset:
            N = self.test_dataset[0].shape[0]

            if shuffle:
                ids = np.array(range(0, N))
                np.random.shuffle(ids)
                ids = ids[:subset]

            else:
                ids = np.array(range(0, subset))

            return np.array(self.val_dataset[0][ids]), np.array(self.val_dataset[1][ids])
        
        return self.val_dataset

    def get_test_set(self, subset=None, shuffle=True):
        
        if subset:
            N = self.test_dataset[0].shape[0]

            if shuffle:
                ids = np.array(range(0, N))
                np.random.shuffle(ids)
                ids = ids[:subset]

            else:
                ids = np.array(range(0, subset))

            return np.array(self.test_dataset[0][ids]), np.array(self.test_dataset[1][ids])
        
        return self.test_dataset
    
    def load_wisdm2_data(self,file_path):
        filepath = os.path.join(file_path)
        data = np.load(filepath)
        return (data['x_train'], data['x_val'], data['x_test'], data['y_train'], data['y_val'], data['y_test'])


class WisdmDatasetParser():
    def __init__(self, file_name, norm="std", class_sublset = None, subset_list = None):
        self.file_name = file_name
        (x_train, x_val, x_test, y_train, y_val, y_test) = self.load_wisdm2_data(file_name)
        self.class_sublset = class_sublset
        self.norm = norm
        self.mean = np.mean(x_train, axis=(0,1))
        self.std = np.std(x_train, axis=(0,1))
        print(self.mean.shape)
        print(self.std.shape)
        if self.norm == "std":
            x_train = x_train - self.mean
            x_train = x_train/self.std

            x_val = x_val - self.mean
            x_val = x_val/self.std

            x_test = x_test - self.mean
            x_test = x_test/self.std
        
        elif self.norm == "custom":
            x_train = x_train/self.std
            x_val = x_val/self.std
            x_test = x_test/self.std
        
        elif self.norm == None:
            pass
        
        print(f'ytrain shape {y_train.shape}')
        print(f'yval shape {y_val.shape}')
        print(f'ytest shape {y_test.shape}')
        
        x_train = np.transpose(x_train,axes=(0,2,1))
        x_val = np.transpose(x_val,axes=(0,2,1))
        x_test = np.transpose(x_test,axes=(0,2,1))
        
        if self.class_sublset is not None:
            if self.class_sublset == '7BC':
                selected_classes =  [1,6,7,8,13,14,17]
            elif self.class_sublset == '7WC':
                selected_classes = [11,12,13,14,15,16,17]
            elif self.class_sublset == 'subset_2':
                selected_classes = [6, 7, 8, 9, 10, 11, 12]
            elif self.class_sublset == 'custom':
                selected_classes = subset_list
            
            x_train, y_train = filter_dataset(x_train, y_train, selected_classes)
            x_val, y_val = filter_dataset(x_val, y_val, selected_classes)
            x_test, y_test = filter_dataset(x_test, y_test, selected_classes)

        if len(y_test.shape) > 1:
            self.train_dataset = (x_train, np.argmax(y_train, axis=-1))
            self.val_dataset = (x_val, np.argmax(y_val, axis=-1))
            self.test_dataset = (x_test, np.argmax(y_test, axis=-1))
        else:
            self.train_dataset = (x_train, y_train)
            self.val_dataset = (x_val,y_val)
            self.test_dataset = (x_test,y_test)
        
        print(f'num classes train dataset: {self.train_dataset[1].max()+1} occurrences of each class:{np.bincount(self.train_dataset[1])}')
        print(f'num classes eval dataset: {self.val_dataset[1].max()+1} occurrences of each class:{np.bincount(self.val_dataset[1])}')
        print(f'num classes test dataset: {self.test_dataset[1].max()+1} occurrences of each class:{np.bincount(self.test_dataset[1])}')

    def get_training_set(self, subset=None, shuffle=True):
        
        if subset:
            N = self.test_dataset[0].shape[0]

            if shuffle:
                ids = np.array(range(0, N))
                np.random.shuffle(ids)
                ids = ids[:subset]

            else:
                ids = np.array(range(0, subset))
                
            return np.array(self.train_dataset[0][ids]), np.array(self.train_dataset[1][ids])
        return self.train_dataset

    def get_validation_set(self, subset=None, shuffle=True):
        
        if subset:
            N = self.test_dataset[0].shape[0]

            if shuffle:
                ids = np.array(range(0, N))
                np.random.shuffle(ids)
                ids = ids[:subset]

            else:
                ids = np.array(range(0, subset))

            return np.array(self.val_dataset[0][ids]), np.array(self.val_dataset[1][ids])
        
        return self.val_dataset

    def get_test_set(self, subset=None, shuffle=True):
        
        if subset:
            N = self.test_dataset[0].shape[0]

            if shuffle:
                ids = np.array(range(0, N))
                np.random.shuffle(ids)
                ids = ids[:subset]

            else:
                ids = np.array(range(0, subset))

            return np.array(self.test_dataset[0][ids]), np.array(self.test_dataset[1][ids])
        
        return self.test_dataset

    def de_std(self, data):
        if self.norm == "norm":
            data= data * self.std
            data= data + self.mean
        if self.norm == "custom":
            data= data * self.std

    def do_std(self, data):
        data= data - self.mean
        data= data / self.std
        
    @staticmethod
    def load_wisdm2_data(file_path):
        filepath = os.path.join(file_path)
        data = np.load(filepath)
        return (data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3'], data['arr_4'], data['arr_5'])

def load_wisdm2_data(file_path):
        filepath = os.path.join(file_path)
        data = np.load(filepath)
        return (data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3'], data['arr_4'], data['arr_5'])


def filter_dataset(x_train, y_train, selected_classes):
    # Create a mapping dictionary for the selected classes
    class_mapping = {original: new for new, original in enumerate(selected_classes)}
    
    # Convert selected_classes to a set for faster look-up
    selected_set = set(selected_classes)
    
    # Get the indices of the selected classes in y_train
    original_class_indices = np.argmax(y_train, axis=1)
    mask = np.isin(original_class_indices, selected_classes)
    
    # Filter the data and labels using the mask
    filtered_x = x_train[mask]
    filtered_y = y_train[mask]
    
    # Map the original class indices to new indices
    new_class_indices = np.vectorize(class_mapping.get)(original_class_indices[mask])
    
    # Create the new one-hot encoded labels
    new_one_hot_y = np.zeros((filtered_y.shape[0], len(selected_classes)))
    new_one_hot_y[np.arange(filtered_y.shape[0]), new_class_indices] = 1
    
    return filtered_x, new_one_hot_y

class WisdmDataset(Dataset):
     
    def __init__(self, data, transform=None, augument=None):
        xs, y = data
        self.x = []
        self.y = y 
        self.augument = augument
        self.transform = transform 
        if self.transform is not None:
            print("transforming array ....")
            for x in xs:
                tmp = self.transform(x)
                self.x.append(tmp)
            print(f' lenght of transformed array {len(self.x)}')
            self.x = np.array(self.x)

        else:
            self.x = xs

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        if self.augument:
            x = self.augument(x)
        else:
            x = torch.tensor(x, dtype=torch.float32)
        return x, y

    def __len__(self):
        return self.y.shape[0]
