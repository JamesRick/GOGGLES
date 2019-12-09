from goggles import construct_image_affinity_matrices, GogglesDataset, infer_labels
from goggles.affinity_matrix_construction.construct import construct_audio_affinity_matrices
from goggles.affinity_matrix_construction.audio_AF.pretrained_models.vggish_wrapper import VGGish_wrapper
from goggles.affinity_matrix_construction.audio_AF.pretrained_models.soundnet_wrapper import Soundnet_wrapper
from goggles.utils.dataset import AudioDataset

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import os
import torch
import shutil
import tqdm

import numpy as np
import pandas as pd
import pickle as pkl
import librosa as lb
import argparse as ap
import soundfile as sf
import multiprocessing as mp
import matplotlib.pyplot as plt

def plot_varying_dev_size(balanced_df, dataset_names, num_prototypes=5):
    '''
    Plot the balanced accuracy as the size of the development set increases.
    '''
    ymin, ymax=0.5, 1.0
    xmin, xmax=0, 22
    colors = ['red', 'blue', 'orange', 'green', 'magenta']
    plt.figure("Varying Size of Development Set")
    for k, (layer_idx_list, model_name) in enumerate(zip(['[3, 7, 17]', '[2, 5, 10, 15]'], ['SoundNet', 'VGGish'])):
        plt.subplot(211 + k)
        for i, dataset_name in enumerate(dataset_names):
            cur_df = balanced_df[(balanced_df['dataset_name'] == dataset_name) &\
                                      (balanced_df['model_name'] == model_name.lower()) &\
                                      (balanced_df['layer_idx_list'] == layer_idx_list) &\
                                      (balanced_df['num_prototypes'] == num_prototypes)].copy()
#
            x, y = cur_df['dev_set_size'].values, cur_df['balanced_accuracy'].values
            plt.plot(x, y, marker='o', c=colors[i], label=dataset_name); plt.ylim(ymin=ymin, ymax=ymax); plt.xlim(xmin=xmin, xmax=xmax)
            plt.title(model_name)
            plt.xlabel('Development Set Size')
            plt.ylabel('Balanced Accuracy')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.tight_layout()
    plt.show()

def plot_varying_afs(balanced_df, dataset_names, dev_set_size=5):
    '''
    Plot the balanced accuracy as the number of affinity functions increases.
    '''
    ymin, ymax=0.5, 1.0
    xmin, xmax=0, 70
    colors = ['red', 'blue', 'orange', 'green', 'magenta']
    plt.figure("Varying Number of Affinity Function")
    for k, (layer_idx_list, model_name) in enumerate(zip(['[3, 7, 17]', '[2, 5, 10, 15]'], ['SoundNet', 'VGGish'])):
        plt.subplot(211 + k)
        for i, dataset_name in enumerate(dataset_names):
            cur_df = balanced_df[(balanced_df['dataset_name'] == dataset_name) &\
                                      (balanced_df['model_name'] == model_name.lower()) &\
                                      (balanced_df['layer_idx_list'] == layer_idx_list) &\
                                      (balanced_df['dev_set_size'] == dev_set_size)].copy()
#
            x, y = cur_df['num_afs'].values, cur_df['balanced_accuracy'].values
            plt.plot(x, y, marker='o', c=colors[i], label=dataset_name); plt.ylim(ymin=ymin, ymax=ymax); plt.xlim(xmin=xmin, xmax=xmax)
            plt.title(model_name)
            plt.xlabel('Number of Affinity Functions')
            plt.ylabel('Balanced Accuracy')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            plt.tight_layout()
    plt.show()

def parse_directories(output_dir):
    '''
    Parses the GOGGLES/goggles/output directory to create a dataframe from the
    pickle files. Writes the dataframe to GOGGLES/goggles/results/full_results.csv
    '''
    goggles_dir = os.path.dirname(output_dir)
    output_dict = {
                   'accuracy': [],
                   'balanced_accuracy': [],
                   'kmeans_accuracy': [],
                   'precision': [],
                   'recall': [],
                   'f1': [],
                   'dataset_name': [],
                   'model_name': [],
                   'layer_idx_list': [],
                   'num_prototypes': [],
                   'num_afs': [],
                   'dev_set_size': [],
                   'version': [],
                   'seed': [],
                   'classes': []
                  }
    for dataset_dir in tqdm.tqdm(os.listdir(output_dir), position=2, leave=True):
        dataset_path = os.path.join(output_dir, dataset_dir)
        for model_dir in tqdm.tqdm(os.listdir(dataset_path), position=1, leave=False):
            model_path = os.path.join(dataset_path, model_dir)
            for version_dir in os.listdir(model_path):
                version_path = os.path.join(model_path, version_dir)
                for dk_dir in tqdm.tqdm(os.listdir(version_path), position=0, leave=False):
                    dk_path = os.path.join(version_path, dk_dir)
                    for seed_dir in os.listdir(dk_path):
                        seed_path = os.path.join(dk_path, seed_dir)
                        output_pkl_file = os.path.join(seed_path, 'output_dict.pkl')
                        with open(output_pkl_file, 'rb') as inp_fle:
                            pkl_dict = pkl.load(inp_fle)

                        if pkl_dict['version'] == 'v7' and \
                            ((pkl_dict['model_name'] == 'vggish' and pkl_dict['layer_idx_list'] == [2, 5, 10, 15]) or \
                            (pkl_dict['model_name'] == 'soundnet' and pkl_dict['layer_idx_list'] == [3, 7, 17]) or \
                            (pkl_dict['model_name'] == 'vggish_svm' and pkl_dict['layer_idx_list'] == [21]) or \
                            (pkl_dict['model_name'] == 'soundnet_svm' and pkl_dict['layer_idx_list'] == [17])):
                            for key in pkl_dict.keys():
                                if key == 'layer_idx_list' or key == 'classes':
                                    output_dict[key].append(str(pkl_dict[key]))
                                else:
                                    output_dict[key].append(pkl_dict[key])
    df = pd.DataFrame(output_dict)
    results_dir = os.path.join(goggles_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    df.to_csv(os.path.join(results_dir, 'full_results.csv'), sep=',')
    return df

def main(results_csv=None):
    goggles_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(goggles_dir, 'output')
    if results_csv is None:
        df = parse_directories(output_dir)
    else:
        df = pd.read_csv(results_csv, sep=',')
    df_group = df.groupby(["dataset_name", "classes", "model_name", "layer_idx_list", "num_prototypes", "num_afs", "dev_set_size", "version"])
    df_filtered = df_group.filter(lambda x: len(x) == 5)
    df_filtered = df_filtered[(df_filtered['layer_idx_list'] == '[3, 7, 17]') | (df_filtered['layer_idx_list'] == '[2, 5, 10, 15]') | (df_filtered['model_name'] == 'soundnet_svm') | (df_filtered['model_name'] == 'vggish_svm')]
    df_group = df_filtered.groupby(["dataset_name", "classes", "model_name", "layer_idx_list", "num_prototypes", "num_afs", "dev_set_size", "version"])
    intermediate_balanced_df = pd.DataFrame({'balanced_accuracy' : df_group['balanced_accuracy'].mean()}).reset_index()
    intermediate_group = intermediate_balanced_df.groupby(["dataset_name", "model_name", "layer_idx_list", "num_prototypes", "num_afs", "dev_set_size", "version"])
    balanced_df = pd.DataFrame({'balanced_accuracy' : intermediate_group['balanced_accuracy'].mean()}).reset_index()
    dataset_names = ['UrbanSound8K', 'TUT-UrbanAcousticScenes', 'ESC-10', 'ESC-50', 'LITIS']
    print('VGGish: ', balanced_df[(balanced_df['dev_set_size'] == 5) & (balanced_df['num_prototypes'] == 5) & (balanced_df['model_name'] == 'vggish')])
    print('VGGish_svm: ', balanced_df[(balanced_df['dev_set_size'] == 5) & (balanced_df['num_prototypes'] == 5) & (balanced_df['model_name'] == 'vggish_svm')])
    print('SoundNet: ', balanced_df[(balanced_df['dev_set_size'] == 5) & (balanced_df['num_prototypes'] == 5) & (balanced_df['model_name'] == 'soundnet')])
    print('SoundNet_svm: ', balanced_df[(balanced_df['dev_set_size'] == 5) & (balanced_df['num_prototypes'] == 5) & (balanced_df['model_name'] == 'soundnet_svm')])
    plot_varying_afs(balanced_df, dataset_names, dev_set_size=5)
    plot_varying_dev_size(balanced_df, dataset_names, num_prototypes=5)
    print("---End of postprocess---")

if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('--results_csv',
                        type=str,
                        default=None,
                        required=False)
    args = parser.parse_args()
    main(**args.__dict__)
