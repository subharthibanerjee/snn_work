# -*- coding: utf-8 -*-
"""
Spyder Editor


"""

import os, random
os.environ["KERAS_BACKEND"] = "theano"
os.environ["THEANO_FLAGS"] = ""
import numpy as np

import theano as th
import theano as T
from keras.utils import np_utils
import keras.models as models

from keras.layers.core import Reshape, Dense, Dropout, Activation, Flatten

from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import *
from keras.optimizers import adam
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import  random, sys, keras
import pandas as pd

'''
modulation_lbls = ['32PSK',
 '16APSK',
 '32QAM',
 'FM',
 'GMSK',
 '32APSK',
 'OQPSK',
 '8ASK',
 'BPSK',
 '8PSK',
 'AM-SSB-SC',
 '4ASK',
 '16PSK',
 '64APSK',
 '128QAM',
 '128APSK',
 'AM-DSB-SC',
 'AM-SSB-WC',
 '64QAM',
 'QPSK',
 '256QAM',
 'AM-DSB-WC',
 'OOK',
 '16QAM']
'''
modulation_lbls = ['OOK',
 '4ASK',
 '8ASK',
 'BPSK',
 'QPSK',
 '8PSK',
 '16PSK',
 '32PSK',
 '16APSK',
 '32APSK',
 '64APSK',
 '128APSK',
 '16QAM',
 '32QAM',
 '64QAM',
 '128QAM',
 '256QAM',
 'AM-SSB-WC',
 'AM-SSB-SC',
 'AM-DSB-WC',
 'AM-DSB-SC',
 'FM',
 'GMSK',
 'OQPSK']
sample_size = 2500
print("Modules loaded successfully..")

os.chdir("..")
cwd = os.getcwd()
filename = cwd+'\\2018.01\\' +'GOLD_XYZ_OSC.0001_1024.hdf5'
print("Opening file at",filename)
#f = h5py.File(filename, 'r')

iq_mod = []
# 2555904
with h5py.File(filename, 'r') as f:
    # List all groups
    print("Keys: %s" % f.keys())
    n_samples = f['X'].shape[0]
    samp = random.sample(range(1, n_samples), sample_size)
    
    samp.sort()
    samp = np.asarray(samp, dtype=np.int64)
    
    iq_data_obj = f['X']
    iq_data = iq_data_obj[samp, :, :]
    print("I/Q data...")
    mod_data_obj = f['Y']
    mod_data = mod_data_obj[samp, :]
    print("mod data..")
    snr_data_obj = f['Z']
    snr_data = snr_data_obj[samp, :]
    print("snr data..")

    # Get the data
r, c= np.where(mod_data==1)

# create modulation dataset

for j in range(0, sample_size):
    iq_mod.append(modulation_lbls[c[r[j]]])
    
for i in range (0, 2500):
    X = iq_data[i, :, 0]
    Y = iq_data[i, :, 1]
    title = "SNR = "+str(snr_data[i])+'dB'+' '+str(iq_mod[i])
    plt.title(title)
    plt.scatter(X, Y, color='red')
    
    plt.show()




print("Data loaded successfully from file ",filename)    

# try to plot modulation

