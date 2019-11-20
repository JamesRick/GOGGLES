from goggles import construct_image_affinity_matrices, GogglesDataset,infer_labels
from goggles.utils.dataset import RawAudioDataset
from goggles.affinity_matrix_construction.construct import construct_soundnet_affinity_matrices
import numpy as np
import pandas as pd
import argparse as ap
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import torchaudio.transforms as audio_transforms
import soundfile as sf
import librosa as lb
import shutil
import os

def main(layer_idx_list=[3,7,17], num_prototypes=10, dev_set_size=5, cache=True, version='ESC-10'):

    num_cpus = int(os.cpu_count())
    df = pd.read_csv('../data/ESC-10/meta/esc10.csv', sep=',').sort_values(by=['filename'])
    df = df[df['esc10']]
    df = df[['filename', 'target', 'category']]
    # df = df[(df['target'] == 10) | (df['target'] == 11)]
    df = df.reset_index(drop=True)

    dataset = RawAudioDataset.load_all_data("../data/ESC-10/audio", meta_df=df)

    import pdb; pdb.set_trace()
    # afs = construct_soundnet_affinity_matrices(dataset, layer_idx_list,
    #                                         num_prototypes=num_prototypes,
    #                                         cache=cache,
    #                                         version=version)
    print("Complete")
    map_dict = dict(zip(np.unique(df['target']).tolist(), np.arange(np.unique(df['target']).size).tolist()))

    dev_set_indices = []
    dev_set_labels = []
    for unique_label in np.unique(df['target']):
        cur_target = df[df['target'] == unique_label].index.values[:dev_set_size]
        dev_set_indices.extend(cur_target.tolist())
        dev_set_labels.extend(df.iloc[cur_target]['target'].values.tolist())

    dev_set_labels = [map_dict[k] for k in dev_set_labels]
    y_true = [map_dict[k] for k in df['target'].values.tolist()]
    prob = infer_labels(afs, dev_set_indices, dev_set_labels)
    pred_labels = np.argmax(prob, axis=1).astype(int)
    print("accuracy", accuracy_score(y_true, pred_labels))
    print(confusion_matrix(y_true, pred_labels))
    print("---End of test_soundnet---")

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