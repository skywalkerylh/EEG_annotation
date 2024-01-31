
import os
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
plt.switch_backend('Qt5Agg')


def extractTimeValuefromtxt(txt):
    '''
    return
    values: int array
    time : string list
    '''
    data = txt.readlines()
    time= [line.split('\t')[0]for line in data]
    values= [line.split('\t')[1].split('\n')[0] for line in data]
    values= [int(line) for line in values]
    values= np.array(values).reshape(1,-1)
    return values, time

def valueConversion(data):
    data = (data - 2048) * 100 /2048
    return data

def readEEGtxt(file_path):
    txt= open(file_path, 'r')
    data, time= extractTimeValuefromtxt(txt)
    data= valueConversion(data)
    return data,time

#file_list= ['EEG1.txt', 'EEG2.txt']
folder_path= 'c:/Users/annie/Desktop/OHCA_data/P10'

file_path= os.path.join(folder_path, 'EEG1_vis_annot.txt')
data_ch1, time_ch1= readEEGtxt(file_path)
file_path= os.path.join(folder_path, 'EEG2_vis_annot.txt')
data_ch2, time_ch2= readEEGtxt(file_path)
assert data_ch1.shape== data_ch2.shape

# stack 2 ch eeg
data= np.concatenate((data_ch1, data_ch2),axis=0)

#create a mne raw object
info= mne.create_info(ch_names=['F3-P3','F4-P4'], sfreq=128, ch_types='misc', verbose=None)
raw_eeg= mne.io.RawArray(data, info, first_samp=0, copy='auto', verbose=None)
fig = raw_eeg.plot(start=0, duration=6,scalings= 1e3,block=True)
fig.fake_keypress("a")


