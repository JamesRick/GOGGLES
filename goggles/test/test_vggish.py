from goggles import construct_image_affinity_matrices, GogglesDataset,infer_labels
from goggles.utils.dataset import AudioDataset
from goggles.affinity_matrix_construction.construct import construct_audio_affinity_matrices
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

import torchaudio.transforms as audio_transforms
import soundfile as sf
import shutil
import os

def to_binary(x, class_zero=40):
    if x == class_zero:
        return 0
    else:
        return 1

if __name__ == '__main__':

    import pdb; pdb.set_trace()
    version='v1'
    dataset = AudioDataset.load_all_data("../data/ESC-10/audio_40_41")
    afs = construct_audio_affinity_matrices(dataset, cache=True, version=version)
    # dataset = GogglesDataset.load_all_data("../data/ESC-50-master/audio")
    print("Complete")
    # afs = construct_image_affinity_matrices(dataset, cache=False)
    df = pd.read_csv('../data/ESC-10/meta/esc10.csv', sep=',').sort_values(by=['filename'])
    df = df[df['esc10']][['filename', 'target', 'category']]
    df = df[(df['target'] == 40) | (df['target'] == 41)]
    df = df.reset_index(drop=True)
    # dev_set_indices = df.iloc[df[(df['target'] == 41)].index.values[:5]].index.values.tolist() + \
    #                   df.iloc[df[(df['target'] == 40)].index.values[:5]].index.values.tolist()

    df_target_40 = df[(df['target'] == 40)].index.values[:20]
    df_target_41 = df[(df['target'] == 41)].index.values[:20]

    dev_indices_40, dev_labels_40 = df.iloc[df_target_40].index.values, df.iloc[df_target_40]['target'].values
    dev_indices_41, dev_labels_41 = df.iloc[df_target_41].index.values, df.iloc[df_target_41]['target'].values

    dev_set_indices = np.concatenate((dev_indices_40, dev_indices_41)).tolist()
    dev_set_labels = np.concatenate((dev_labels_40, dev_labels_41)).tolist()

    dev_set_labels = list(map(to_binary, dev_set_labels))
    # dev_set_indices_40 = df.iloc[df[(df['target'] == 40)].index.values[:5]]
    # dev_set_labels_40 = df.iloc[df[(df['target'] == 41)].index.values[:5]]['target'].values.tolist()

    # dev_set_indices, dev_set_labels = df.index.values.tolist(), df['target'].values.tolist()
    
    import pdb; pdb.set_trace()
    # for filename in df['filename'].values.tolist():
    #     shutil.copyfile(os.path.join('../data/ESC-10/audio', filename),
    #         os.path.join('../data/ESC-10/audio_40_41', filename))

    # dev_set_indices, dev_set_labels = [0,1,2,90,91,92],[0,0,0,1,1,1]
    # y_true = pd.read_csv("../data/cub_dataset/labels.csv")
    # y_true = y_true['y'].values
    y_true = list(map(to_binary, df['target'].values.tolist()))
    # y_true[:int(y_true.shape[0]/2)] = 0
    prob = infer_labels(afs, dev_set_indices, dev_set_labels)
    pred_labels = np.argmax(prob,axis=1).astype(int)

    import pdb; pdb.set_trace()

    print("accuracy", accuracy_score(y_true,pred_labels))