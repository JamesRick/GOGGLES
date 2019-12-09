from goggles.affinity_matrix_construction.construct import construct_audio_affinity_matrices
from goggles.affinity_matrix_construction.audio_AF.pretrained_models.vggish_wrapper import VGGish_wrapper
from goggles.affinity_matrix_construction.audio_AF.pretrained_models.soundnet_wrapper import Soundnet_wrapper
from goggles.utils.dataset import AudioDataset
from skimage.feature import hog
from scipy import stats
from tqdm import trange

import os
import re
import sys
import torch
import shutil
import numpy as np
import pandas as pd
import pickle as pkl
import librosa as lb
import argparse as ap
import soundfile as sf
import matplotlib.pyplot as plt

def main(wav_file='../data/ESC-10/audio/1-100032-A-0.wav'):
    goggles_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wav_frames, wav_full = VGGish_wrapper.preprocess(wav_file)
    feature_data, hog_image = hog(wav_full.T, orientations=8, pixels_per_cell=(8, 8),
                        cells_per_block=(1, 1), visualize=True)
    import pdb; pdb.set_trace()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    ax1.axis('off')
    ax1.imshow(wav_full.T)
    ax1.set_title('Input Spectrogram')
    ax2.axis('off')
    ax2.imshow(hog_image, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()
    print("---End of visual---")

if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('--wav_file',
                        type=str,
                        default='../data/ESC-10/audio/1-100032-A-0.wav',
                        required=False)

    args = parser.parse_args()
    main(**args.__dict__)
