from goggles import construct_image_affinity_matrices, GogglesDataset, infer_labels
from goggles.affinity_matrix_construction.construct import construct_audio_affinity_matrices
from goggles.affinity_matrix_construction.audio_AF.pretrained_models.vggish_wrapper import VGGish_wrapper
from goggles.affinity_matrix_construction.audio_AF.pretrained_models.soundnet_wrapper import Soundnet_wrapper
from goggles.utils.dataset import AudioDataset

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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

def main():
    goggles_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(goggles_dir, 'data')
    urbansound_dir = os.path.join(data_dir, 'temp_urbansound8k/audio')
    for fold in os.listdir(urbansound_dir):
        fold_dir = os.path.join(urbansound_dir, fold)
        if not os.path.isdir(fold_dir):
            continue
        for wav_files in os.listdir(fold_dir):
            wav_file_path = os.path.join(fold_dir, wav_files)
            if '.wav' in wav_file_path and sf.info(wav_file_path).subtype != 'PCM_16':
                wav_data, sr = sf.read(wav_file_path)
                sf.write(wav_file_path, wav_data, sr, subtype='PCM_16')
                new_wav_data, new_sr = sf.read(wav_file_path, dtype='int16')
                print("Conversion Successful: ", sf.info(wav_file_path).subtype == 'PCM_16')
                if not sf.info(wav_file_path).subtype == 'PCM_16':
                    import pdb; pdb.set_trace()

    # import pdb; pdb.set_trace()
    # for datasets in os.listdir(data_dir):
    #     if datasets == 'UrbanSound8K' or datasets == 'temp_urbansound8k' or not os.path.isdir(datasets):
    #         continue
    #     dataset_path = os.path.join(data_dir, datasets)
    #     for audio in os.listdir(dataset_path):
    #         if audio != 'audio' and audio != 'data_rouen':
    #             continue
    #         audio_path = os.path.join(dataset_path, audio)
    #         for wav_files in os.listdir(audio_path):
    #             wav_file_path = os.path.join(audio_path, wav_files)
    #             if sf.info(wav_file_path).subtype != 'PCM_16':
    #                 print(datasets)
    #                 break

    import pdb; pdb.set_trace()
    print("---End of convert_to_pcm16---")

if __name__ == '__main__':
    parser = ap.ArgumentParser()
    # parser.add_argument('--layer_idx_list',
    #                     type=int,
    #                     nargs='+',
    #                     default=[3,7,17],
    #                     required=False)

    args = parser.parse_args()
    # main(**args.__dict__)
    main()
