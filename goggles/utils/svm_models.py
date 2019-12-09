from goggles import construct_image_affinity_matrices, GogglesDataset, infer_labels
from goggles.affinity_matrix_construction.construct import construct_audio_affinity_matrices
from goggles.affinity_matrix_construction.audio_AF.pretrained_models.vggish_wrapper import VGGish_wrapper
from goggles.affinity_matrix_construction.audio_AF.pretrained_models.soundnet_wrapper import Soundnet_wrapper
from goggles.utils.dataset import AudioDataset
import goggles.torch_vggish.audioset.vggish_input as vggish_input

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score
from sklearn import svm
from sklearn import preprocessing

import os
import torch
import shutil

import numpy as np
import pandas as pd
import pickle as pkl
import librosa as lb
import argparse as ap
import soundfile as sf
import multiprocessing as mp
import matplotlib.pyplot as plt

class VGGish_svm(object):
    """
    VGGish_svm is used to perform svm fit/predict on the
    output of the VGGish model's embedding. Feature scaling
    is performed before fitting and predicting on the input
    X feature array. A majority vote prediction on the individual
    1 second spectrogram predictions is used to produce the final
    set of predictions for the SVM classifier.
    """
    def __init__(self, wrapper_model):
        self.wrapper_model = wrapper_model
        self.svm_clf = svm.LinearSVC(max_iter=1000000, class_weight='balanced')
        self.scaler = preprocessing.StandardScaler()

    @classmethod
    def preprocess(cls, wav_file):
        return vggish_input.wavfile_to_examples(wav_file)

    def get_svm_data(self, wav_data):
        return self.wrapper_model.get_svm_data((wav_data,))

    def fit(self, X, y):
        self.svm_model = self.svm_clf.fit(self.scaler.fit_transform(X), y)

    def predict(self, X):
        return self.svm_model.predict(self.scaler.transform(X))

class Soundnet_svm(object):
    """
    Soundnet_svm is used to perform svm fit/predict on the
    output of the Soundnet model's Pool5 layer (layer 17 in torch model).
    Feature scaling is performed before fitting and predicting on the input
    X feature array. Waveforms are zero padded to a minimum of 4 seconds
    (to ensure equal length feature vectors) and limited to a maximum of
    20 seconds (Soundnet's maximum input size).
    """
    def __init__(self, wrapper_model):
        self.wrapper_model = wrapper_model
        self.svm_clf = svm.LinearSVC(max_iter=1000000, class_weight='balanced')
        self.scaler = preprocessing.StandardScaler()

    @classmethod
    def preprocess(cls, wav_file):
        wav_data, sr = lb.load(wav_file, mono=True)
        min_length = sr * 4
        max_length = sr * 20
        if wav_data.shape[0] < min_length:
            wav_data = np.concatenate((wav_data, np.zeros(min_length - wav_data.shape[0])))
        wav_data = wav_data[:max_length]
        wav_data = wav_data.reshape(1, -1, 1)
        wav_data = (((256 - -256)*((wav_data - wav_data.min()) / (wav_data.max() - wav_data.min()))) - 256)
        return wav_data, None

    def get_svm_data(self, wav_data, layer_idx=17):
        return self.wrapper_model.get_svm_data((wav_data,), layer_idx=layer_idx)

    def fit(self, X, y):
        self.svm_model = self.svm_clf.fit(self.scaler.fit_transform(X), y)

    def predict(self, X):
        return self.svm_model.predict(self.scaler.transform(X))
