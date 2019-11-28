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
import librosa as lb
import argparse as ap
import soundfile as sf
import multiprocessing as mp
import matplotlib.pyplot as plt

def main(layer_idx_list=[2,5,10,15],
         num_prototypes=10,
         dev_set_size=5,
         cache=False,
         version='ESC-10',
         random_targets=True):

    # num_cpus = int(os.cpu_count())
    np.random.seed(151)
    model = VGGish_wrapper()
    df = pd.read_csv('../data/UrbanSound8K/metadata/UrbanSound8K.csv', sep=',').sort_values(by=['slice_file_name'])
    #df = df[df['esc10']]
    #df  = df.head(150)
    df = df[['slice_file_name', 'fold', 'classID', 'class']]
    df.columns = ['filename', 'fold', 'target', 'category']
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
        dataset = AudioDataset.load_all_data("../data/UrbanSound8K/audio/fold"+str(i), model.preprocess, meta_df=cur_df)
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
    row_sums = cf_matrices.sum(axis=2).reshape(-1,2,1)
    cf_matrices_norm = cf_matrices / row_sums
    print("Average Accuracy: " + str(np.array(accuracies).mean()))

    print("Average Confusion Matrix")
    print(cf_matrices.mean(axis=0))

    print("Average Confusion Matrix Norm")
    print(cf_matrices_norm.mean(axis=0))

    print("---End of test_vggish---")

if __name__ == '__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('--layer_idx_list',
                        type=int,
                        nargs='+',
                        default=[2,5,10,15],
                        required=False)
    parser.add_argument('--num_prototypes',
                        type=int,
                        default=10,
                        required=False)
    parser.add_argument('--dev_set_size',
                        type=int,
                        default=5,
                        required=False)
    parser.add_argument('--cache',
                        type=bool,
                        default=False,
                        required=False)
    parser.add_argument('--version',
                        type=str,
                        default='v0',
                        required=False)
    args = parser.parse_args()
    main(**args.__dict__)
