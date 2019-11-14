# -*- coding: utf-8 -*-
"""
Spyder Editor


"""

import os, random
os.environ["KERAS_BACKEND"] = "theano"
os.environ["THEANO_FLAGS"] = "device=cpu"
import numpy as np

import theano as th
import theano as T
from keras.utils import np_utils
import keras.models as models
import keras.backend as K

from keras.layers.core import Reshape, Dense, Dropout, Activation, Flatten

from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import *
from keras.optimizers import adam
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import  random, sys, keras
import pandas as pd

# keras hard setup

K.set_image_dim_ordering('th')
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
    samp = np.asarray(samp, dtype=np.int32)
    
    iq_data_obj = f['X']
    iq_data = iq_data_obj[samp, :, :]
    print("I/Q data...")
    mod_data_obj = f['Y']
    mod_data = mod_data_obj[samp, :]
    print("mod data..")
    snr_data_obj = f['Z']
    snr_data = snr_data_obj[samp, :]
    print("snr data..")
print("Data loaded successfully from file ",filename)    
    # Get the data
r, c= np.where(mod_data==1)

# create modulation dataset

for j in range(0, sample_size):
    iq_mod.append(modulation_lbls[c[r[j]]])
    '''
for i in range (0, 2500):
    X = iq_data[i, :, 0]
    Y = iq_data[i, :, 1]
    title = 'SNR='+str(snr_data[i][0])+'dB'+' mod='+str(iq_mod[i])
    plt.title(title)
    plt.scatter(X, Y, color='red')
    
    plt.show()
'''
# partition the data in train and test
    
n_train = round(sample_size * 0.5)
train_idx = np.random.choice(range(0, sample_size), size=n_train, replace=False)
test_idx = list(set(range(0,sample_size))-set(train_idx))
iq_data=iq_data.reshape(sample_size, 2, iq_data.shape[1])
X_train = iq_data[train_idx]
X_test = iq_data[test_idx]

# some hard coded trick to reshape array
in_shp = X_train.shape[1:]
in_shp = list(in_shp)

iq_mod = np.asarray(iq_mod, dtype='<S20')
iq_mod = iq_mod.reshape(sample_size,1)
Y_train = mod_data[train_idx]
Y_test = mod_data[test_idx]


## model details

dr = 0.5 # dropout rate (%)
model = models.Sequential()
model.add(Reshape([1]+in_shp, input_shape=in_shp))

model.add(ZeroPadding2D((0, 2)))
model.add(Conv2D(256, (1, 3), padding='valid', activation="relu", name="conv1",
                 kernel_initializer='glorot_uniform'))
model.add(Dropout(dr))
model.add(ZeroPadding2D((0, 2)))
model.add(Conv2D(80, (2, 3), padding="valid", activation="relu", name="conv2",
                 kernel_initializer='glorot_uniform'))
print("second conv2d")
model.add(Dropout(dr))
model.add(Flatten())
print('flattened')
model.add(Dense(1, activation='relu', kernel_initializer='he_normal', 
                name="dense1"))
print('dense done')
model.add(Dropout(dr))
model.add(Dense( len(modulation_lbls), kernel_initializer='he_normal', 
                name="dense2" ))
model.add(Activation('softmax'))
model.add(Reshape([len(modulation_lbls)]))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()

# try to plot modulation

# Set up some params 
nb_epoch = 2     # number of epochs to train on
batch_size = 128  # training batch size


#   - call the main training loop in keras for our network+dataset
filepath = 'convmodrecnets_SNN_1_0_1.wts.h5'
#history=model.fit(X_train, Y_train)

history = model.fit(X_train,
    Y_train,
    batch_size=batch_size,
    epochs=nb_epoch,
    verbose=2,
    callbacks = [keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', 
                                        verbose=2, save_best_only=True, mode='auto'),
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, 
                                      verbose=2, mode='auto')
    ])
keras.models.save_model(model, os.path.join(cwd, filepath))   
# we re-load the best weights once training is finished
model.load_weights(filepath)


# Show simple version of performance
score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
print(score)