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

def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1]/len(list_values) for elem in counter_values]
    entropy = scipy.stats.entropy(probabilities)
    return entropy

def calculate_statistics(list_values):
    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    n75 = np.nanpercentile(list_values, 75)
    n95 = np.nanpercentile(list_values, 95)
    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    std = np.nanstd(list_values)
    var = np.nanvar(list_values)
    rms = np.nanmean(np.sqrt(list_values**2))
    return [n5, n25, n75, n95, median, mean, std, var, rms]

def calculate_crossings(list_values):
    zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_zero_crossings, no_mean_crossings]

def get_features(list_values):
    entropy = calculate_entropy(list_values)
    crossings = calculate_crossings(list_values)
    statistics = calculate_statistics(list_values)
    return [entropy] + crossings + statistics

def get_single_ecg_features(signal):
    features = []
    for channel in signal.T:
        # Time-domain raw features (12 features)
        raw_feats = get_features(channel)
        
        # First derivative stats (9 features)
        diff1 = np.diff(channel)
        diff1_feats = calculate_statistics(diff1)
        
        # Second derivative stats (9 features)
        diff2 = np.diff(diff1)
        diff2_feats = calculate_statistics(diff2)
        
        channel_features = raw_feats + diff1_feats + diff2_feats
        features.append(channel_features)
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

class RawModel(ClassificationModel):
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
