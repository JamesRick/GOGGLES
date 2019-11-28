from goggles import construct_image_affinity_matrices, GogglesDataset, infer_labels
from goggles.affinity_matrix_construction.construct import construct_audio_affinity_matrices
from goggles.affinity_matrix_construction.audio_AF.pretrained_models.vggish_wrapper import VGGish_wrapper
from goggles.affinity_matrix_construction.audio_AF.pretrained_models.soundnet_wrapper import Soundnet_wrapper
from goggles.utils.dataset import AudioDataset

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import os
import shutil

import numpy as np
import pandas as pd
import pickle as pkl
import librosa as lb
import argparse as ap
import soundfile as sf
import multiprocessing as mp
import matplotlib.pyplot as plt

def main(layer_idx_list=[3,7,17],
         num_prototypes=10,
         dev_set_size=5,
         model_name='vggish',
         cache=False,
         dataset_name='ESC-10',
         version='ESC-10',
         random_targets=True):

    goggles_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("Model: " + model_name)
    if model_name == 'vggish':
        model = VGGish_wrapper()
        if max(layer_idx_list) >= len(model._features):
            print("Invalid layer_idx_list: " + str(layer_idx_list))
            layer_idx_list = [2, 5, 10, 15]
            print("Defaulting to layer_idx_list = " + str(layer_idx_list))
    elif model_name == 'soundnet':
        model = Soundnet_wrapper()
        if max(layer_idx_list) >= len(model._features):
            print("Invalid layer_idx_list: " + str(layer_idx_list))
            layer_idx_list = [3, 7, 17]
            print("Defaulting to layer_idx_list = " + str(layer_idx_list))
    else:
        raise Exception("Model " + model_name + " not found.\n\
                        Currently implemented models are:\n\
                        1.\tvggish\n\
                        2.\tsoundnet\n")

    if dataset_name == 'ESC-10':
        dataset_csv = os.path.join(goggles_dir, 'data', dataset_name, 'meta/esc10.csv')
        dataset_audio = os.path.join(goggles_dir, 'data', dataset_name, 'audio')
    else:
        raise Exception("Dataset " + dataset_name + " not found.\n\
                        Currently implemented datasets are:\n\
                        1.\tESC-10\n")
                        #2.\t\n")

    # num_cpus = int(os.cpu_count())
    np.random.seed(151)
    df = pd.read_csv('../data/ESC-10/meta/esc10.csv', sep=',').sort_values(by=['filename'])
    df = df[df['esc10']]
    df = df[['filename', 'fold', 'target', 'category']]
    if random_targets:
        targets = np.random.choice(np.unique(df['target'].values), size=2, replace=False)
    df = df[(df['target'] == targets[0]) | (df['target'] == targets[1])]
    df = df.reset_index(drop=True)
    classes = np.unique(df['category'].values)
    accuracies = []
    cf_matrices = []
    print("Running with classes: %s, %s" % (classes[0], classes[1]))
    for i in np.unique(df['fold']):
        cur_df = df[df['fold'] == i].reset_index(drop=True)
        dataset = AudioDataset.load_all_data("../data/ESC-10/audio", model.preprocess, meta_df=cur_df)
        afs = construct_audio_affinity_matrices(dataset, layer_idx_list, model,
                                                    num_prototypes=num_prototypes,
                                                    cache=cache,
                                                    version=version)
        print("")
        print("Complete")
        map_dict = dict(zip(np.unique(cur_df['target']).tolist(), np.arange(np.unique(cur_df['target']).size).tolist()))

        dev_set_indices = []
        dev_set_labels = []
        for unique_label in np.unique(cur_df['target']):
            cur_target = np.random.choice(cur_df[cur_df['target'] == unique_label].index.values, size=5, replace=False)
            print("For Class: ", map_dict[unique_label], "Using Indices: ", cur_target)
            # cur_target = df[df['target'] == unique_label].index.values[:dev_set_size]
            dev_set_indices.extend(cur_target.tolist())
            dev_set_labels.extend(cur_df.iloc[cur_target]['target'].values.tolist())

        dev_set_labels = [map_dict[k] for k in dev_set_labels]
        y_true = [map_dict[k] for k in cur_df['target'].values.tolist()]
        prob = infer_labels(afs, dev_set_indices, dev_set_labels)
        pred_labels = np.argmax(prob, axis=1).astype(int)
        accuracy = accuracy_score(y_true, pred_labels)
        cf_matrix = confusion_matrix(y_true, pred_labels)
        accuracies.append(accuracy)
        cf_matrices.append(cf_matrix)
        print("Fold " + str(i) + " Accuracy: " + str(accuracy))
        print("Confusion Matrix")
        print(cf_matrix)
    cf_matrices = np.array(cf_matrices)
    accuracies = np.array(accuracies)
    row_sums = cf_matrices.sum(axis=2).reshape(-1,2,1)
    cf_matrices_norm = cf_matrices / row_sums
    print("Average Accuracy: " + str(accuracies.mean()))

    print("Average Confusion Matrix")
    print(cf_matrices.mean(axis=0))

    print("Average Confusion Matrix Norm")
    print(cf_matrices_norm.mean(axis=0))

    output_dict = {
                   'accuracy': accuracy.mean(),
                   'layer_idx_list': layer_idx_list,
                   'num_prototypes': num_prototypes,
                   'dev_set_size': dev_set_size,
                   'version': version
                  }

    output_dir = os.path.join(goggles_dir, 'output')
    run_output_dir = os.path.join(output_dir, str(dataset_name), str(model_name), str(version), str(num_prototypes) + '_' + str(dev_set_size))
    os.makedirs(run_output_dir, exist_ok=True)
    with open(os.path.join(run_output_dir, 'output_dict.pkl')) as out_fle:
        pkl.dump(output_dict, out_fle)


    print("---End of test_vggish---")

if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('--layer_idx_list',
                        type=int,
                        nargs='+',
                        default=[3,7,17],
                        required=False)
    parser.add_argument('--num_prototypes',
                        type=int,
                        default=10,
                        required=False)
    parser.add_argument('--dev_set_size',
                        type=int,
                        default=5,
                        required=False)
    parser.add_argument('--model_name',
                        type=str,
                        default='vggish',
                        required=False)
    parser.add_argument('--cache',
                        type=bool,
                        default=False,
                        required=False)
    parser.add_argument('--dataset_name',
                        type=str,
                        default='ESC-10',
                        required=False)
    parser.add_argument('--version',
                        type=str,
                        default='v0',
                        required=False)
    parser.add_argument('--random_targets',
                        type=bool,
                        default=True,
                        required=False)

    args = parser.parse_args()
    main(**args.__dict__)
