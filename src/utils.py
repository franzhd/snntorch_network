import os
import sys
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
#import lava.lib.dl.slayer as slayer
import seaborn as sns 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import statistics
import csv
import itertools
import datetime as dt
from datetime import datetime
import json
import random as rn
import pandas as pd
import seaborn as sn
#from keras.utils import to_categorical

from sklearn.model_selection import train_test_split


class regularization_loss(object):
    def __init__(self, min_hz, max_hz ,time_window, pth = 0.99):
        """
        Initializes the regularization loss function.

        Args:
            min_hz (float): The minimum desired spike frequency in Hz.
            max_hz (float): The maximum desired spike frequency in Hz.
            time_window (float): The length of the time window in seconds.
            pth (float, optional): The percentile value used to calculate the threshold spike frequency. Defaults to 0.99.
        """
        self.min_hz = min_hz
        self.max_hz = max_hz
        self.pth = pth
        self.time_window = time_window        

    # def __call0__(self, spike_count_array: list[torch.float32]) -> torch.float32:
    #     """
    #     Calculates the regularization loss.

    #     Args:
    #         spike_count_array (list[torch.float32]): A list of spike count arrays.

    #     Returns:
    #         torch.float32: The regularization loss.
    #     """
    #     loss = 0
    #     """ [time, batch, channels]"""

    #     for i in range(len(spike_count_array)):

    #         layer_loss = 0

    #         for j in range(spike_count_array[i].shape[2]):
    #             #print('spike_count_array[i].shape',spike_count_array[i].shape)
    #             frequency_list = []

    #             # for z in range(spike_count_array[i].shape[1]):

    #             #     frequency_list.append(torch.sum(spike_count_array[i][:,z,j])/self.time_window)
                
    #             frequency_list = torch.sum(spike_count_array[i][:,:,j], dim=(0)) / self.time_window
    #             #print('frequency list shape', len(frequency_list))
    #             frequency_matrix = torch.sum(spike_count_array[i], dim=(0, 2)) / self.time_window
    #             frequency_list.sort()

    #             Rpth = frequency_list[int(self.pth*len(frequency_list))]

    #             layer_loss += (F.relu(Rpth - self.max_hz) + F.relu(self.min_hz - Rpth))**2 

    #         loss += layer_loss / spike_count_array[i].shape[1] 
        
    #     return loss
        
    def __call__(self, spike_count_array: list[torch.float32]) -> torch.float32:
        """
        Calculates the regularization loss.

        Args:
            spike_count_array (list[torch.float32]): A list of spike count arrays.

        Returns:
            torch.float32: The regularization loss.
        """
        loss = 0
        """ [time, batch, channels]"""

        for i in range(len(spike_count_array)):

            layer_loss = 0

            frequency_matrix = torch.sum(spike_count_array[i], dim=(0)) / self.time_window
            frequency_matrix = torch.sort(frequency_matrix, dim=-1).values
            Rpth = frequency_matrix[int(self.pth*frequency_matrix.shape[0])]
            
            for j in range(spike_count_array[i].shape[2]):
        
                layer_loss += (F.relu(Rpth[j] - self.max_hz) + F.relu(self.min_hz - Rpth[j]))**2 

            loss += layer_loss / spike_count_array[i].shape[1] 
            #print(f'net loss {loss}')
        return loss
    
def compute_output_labels(matrix):
    # Sum the elements along the last dimension
    #print(f'compute_output_labels matrix shape {matrix.shape}')
    summed_matrix = np.sum(matrix, axis=-1)
     
    # Divide the sum by the number of elements in the second dimension
    divided_matrix = summed_matrix / matrix.shape[1]

    # Find the maximum value along the second dimension and its index
    max_indices = np.argmax(divided_matrix, axis=1)

    return max_indices


def gen_confusion_matrix(predictions, labels, path):
    
    num_label = max(labels)
    #print(f'prediction shape {predictions.shape} and labels shape{len(labels)}')
    conf_matrix = confusion_matrix(predictions, labels)
    # Plot confusion matrix using Seaborn
    plt.figure(figsize=(5, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=range(num_label),
                yticklabels=range(num_label))

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    # Save the plot
    file_path = os.path.join(path,'confusion_matrix.png')
    plt.savefig(file_path, bbox_inches='tight')





def spike_plot(data,spike_data,  label, save=None):
    

    line_size = 0.5
    colors_spike = []
    color = []
    handles = []
    
    if len(spike_data.shape) == 3:
        for i in range(spike_data.shape[0]):
            r = np.random.random()
            b = np.random.random()
            g = np.random.random()
            color.append([r, g, b])
            for d in range(spike_data.shape[1]):
                colors_spike.append([r, g, b])
        
        plot_data =  spike_data.reshape((spike_data.shape[0]*spike_data.shape[1], spike_data.shape[2])) * np.arange(spike_data.shape[-1]) * 1.0/spike_data.shape[-1]

    if len(spike_data.shape) == 2:
        for i in range(spike_data.shape[0]):
            r = np.random.random()
            b = np.random.random()
            g = np.random.random()
            color.append([r, g, b])
            colors_spike.append([r, g, b])
            
        plot_data =  spike_data * np.arange(spike_data.shape[-1]) * 1.0/spike_data.shape[-1]
        print(f'plot data {plot_data.shape}')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(40, 30))
    fig.suptitle(f'Signlal label {label}')


    for i in range(data.shape[0]):
        ax1.plot(range(data.shape[-1]), data[i, :], color = color[i])
        handles.append(plt.Line2D([0], [0], color=color[i], lw=2, label=f'dimension {i}'))
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Dimension')
    ax1.set_title('Original signal dimensions')

    # print(f'colors spike {len(colors_spike)}')
    # print(f"plot data shape {plot_data.shape}")

    ax2.eventplot(plot_data, color=colors_spike, linelengths = line_size)     
    ax2.set_xlabel('Spike')
    ax2.set_ylabel('Channels')
    ax2.set_title('Spike Train encoding')
    plt.xlabel('Spike')
    plt.ylabel('Channels')

    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.95, 0.85))
    if save is None:
        plt.show()
    else:
        plt.savefig(save)

class SearchSpaceUpdater(object):
    def __init__(self, *initial_data, **kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])


def WisdmDf2Np(path,save_path, time_window=2, overlap =0, subset=0):
    phone = pd.read_pickle(path)

    ##### VARIABLES TO SET TIME AND ACTIVITIES SUBSET TO BE USED #####
    #time_window s of activity
    # 50% overlap has been used for the whole dataset (subset=0) to compare with baseline results of paper "A deep learning approach for human activities recognition from multimodal sensing devices"
    #overlap  overlapping fraction (FRACTION with respect to unity, NOT PERCENTAGE)
    #subset  use 0 to select all of the 18 activities
    ##################################################################

    # activities re-ordered according to non-hand, hand general and hand eating subsets
    act_map = {
        'A': 'walking',
        'B': 'jogging',
        'C': 'stairs',
        'D': 'sitting',
        'E': 'standing',
        'M': 'kicking',
        'P': 'dribbling',
        'O': 'catch',
        'F': 'typing',
        'Q': 'writing',
        'R': 'clapping',
        'G': 'teeth',
        'S': 'folding',
        'J': 'pasta',
        'H': 'soup',
        'L': 'sandwich',
        'I': 'chips',
        'K': 'drinking',
    }

    LABEL_SUBSETS = {
        1: [0, 1, 2, 3, 4, 5], # for smartphone and smartwatch
        2: [6, 7, 8, 9, 10, 11, 12], # for smartwatch only
        3: [13, 14, 15, 16, 17], # for smartwatch only
    }

    # LABEL_SUBSETS = {
    #     0: [1,6,7,8,13,14,17], # for smartphone and smartwatch
    #     1: [11,12,13,14,15,16,17], # for smartwatch only
    # }

    window_size = int(20*time_window) # 20 Hz sampling times the temporal length of the window
    stride = int(window_size*(1-overlap))

    frames = []
    for i in range(0, len(phone)-window_size, stride):
        window = phone.iloc[i:i+window_size]
        if window['activity'].nunique() == 1:
            frames.append(window)

    #activities = sorted(act_map.keys())
    activities = act_map.keys() 
    activity_encoding = {v: k for k, v in enumerate(activities)}

    X_list = []
    y_list = []

    #for each frame replace label with activity
    for frame in frames:
        X_list.append(frame[['watch_accel_x', 'watch_accel_y', 'watch_accel_z', 'watch_gyro_x', 'watch_gyro_y', 'watch_gyro_z']].values)
        y_list.append(activity_encoding[frame.iloc[0]['activity']])

    if subset!=0:
        idx = [ii for ii,lbl in enumerate(y_list) if lbl in LABEL_SUBSETS[subset]] # list(ACTIVITIES_LABEL.keys())[:6]
        X_sub = [X_list[ii] for ii in idx] #[jj for jj in X[:,] if jj in idx]
        y_sub = [y_list[jj] for jj in idx] #[kk for kk in y if kk in idx]

        X = np.array(X_sub)
        #y = np.array(to_categorical(y_sub))[:,min(LABEL_SUBSETS[subset]):]
        y = np.array(y_sub)[:,min(LABEL_SUBSETS[subset]):]

    else:
        X = np.array(X_list)
        #y = np.array(to_categorical(y_list))  
        y = np.array(y_list)
        print(f'y_list {y.shape}')
        y = one_hot_encode(y, np.max(y)+1)
        print(f'y_list {y.shape}')
    # 60/20/20 split
    X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=42)
    # if subset == 0:
    #     subset_name = '7BC'
    # elif subset == 1:
    #     subset_name = '7WC'

    savefile = save_path +'/data_watch'+'_subset_'+str(subset)+'_'+str(window_size)
    print (f'Saving data to {savefile}')
    np.savez(savefile, X_train, X_val, X_test, y_train, y_val, y_test)


def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]