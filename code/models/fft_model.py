from scipy.fftpack import fft
from sklearn.decomposition import PCA
from tqdm import tqdm
from models.base_model import ClassificationModel
import pickle
import numpy as np
import os
import time
import pandas as pd
import scipy.io as sio
import scipy.stats
import multiprocessing
import datetime as dt
from collections import defaultdict, Counter
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

def get_single_ecg_features(signal, num_bins=250):
    features = []
    for channel in signal.T:
        # Compute Fast Fourier Transform magnitude
        fft_vals = np.abs(fft(channel))
        # Keep positive frequencies (first half)
        fft_vals = fft_vals[:len(channel) // 2]
        # Resample to a fixed number of bins (num_bins) via interpolation
        x_new = np.linspace(0, 1, num_bins)
        x_old = np.linspace(0, 1, len(fft_vals))
        resampled = np.interp(x_new, x_old, fft_vals)
        features.append(resampled)
    return np.array(features).flatten()

def get_ecg_features(ecg_data, parallel=True):
    if parallel:
        pool = multiprocessing.Pool(18)
        results = pool.map(get_single_ecg_features, ecg_data)
        pool.close()
        pool.join()
        return np.array(results)
    else:
        list_features = []
        for signal in tqdm(ecg_data):
            features = get_single_ecg_features(signal)
            list_features.append(features)
        return np.array(list_features)

class FFTModel(ClassificationModel):
    def __init__(self, name, n_classes, freq, outputfolder, input_shape, classifier='NN'):
        self.name = name
        self.outputfolder = outputfolder
        self.n_classes = n_classes
        self.freq = freq
        self.classifier = classifier
        self.dropout = .25
        self.activation = 'relu'
        self.final_activation = 'sigmoid'
        self.n_dense_dim = 128
        self.epochs = 30

    def fit(self, X_train, y_train, X_val, y_val):
        XF_train = get_ecg_features(X_train)
        XF_val = get_ecg_features(X_val)
        
        if self.classifier == 'NN':
            from keras.layers import Dropout, Dense, Input
            from keras.models import Model
            from keras.callbacks import ModelCheckpoint

            # Standardize input data
            ss = StandardScaler()
            XFT_train = ss.fit_transform(XF_train)
            XFT_val = ss.transform(XF_val)
            pickle.dump(ss, open(self.outputfolder + 'ss.pkl', 'wb'))
            
            # Classification stage
            input_x = Input(shape=(XFT_train.shape[1],))
            x = Dense(self.n_dense_dim, activation=self.activation)(input_x)
            x = Dropout(self.dropout)(x)
            y = Dense(self.n_classes, activation=self.final_activation)(x)
            self.model = Model(input_x, y)
            
            self.model.compile(optimizer='adamax', loss='binary_crossentropy')
            
            # Monitor validation error
            mc_loss = ModelCheckpoint(self.outputfolder + 'best_loss_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
            self.model.fit(XFT_train, y_train, validation_data=(XFT_val, y_val), epochs=self.epochs, batch_size=128, callbacks=[mc_loss])
            self.model.save(self.outputfolder + 'last_model.h5')

    def predict(self, X):
        XF = get_ecg_features(X)
        if self.classifier == 'NN':
            from keras.models import load_model
            ss = pickle.load(open(self.outputfolder + 'ss.pkl', 'rb'))
            XFT = ss.transform(XF)
            model = load_model(self.outputfolder + 'best_loss_model.h5')
            return model.predict(XFT)
