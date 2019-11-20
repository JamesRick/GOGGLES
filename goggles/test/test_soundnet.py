from goggles import construct_image_affinity_matrices, GogglesDataset,infer_labels
from goggles.utils.dataset import RawAudioDataset
from goggles.affinity_matrix_construction.construct import construct_soundnet_affinity_matrices
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import torchaudio.transforms as audio_transforms
import soundfile as sf
import shutil
import os

def to_relative_class(x, class_zero=37):
    return x - class_zero

def to_binary(x, class_zero=40):
    if x == class_zero:
        return 0
    else:
        return 1

if __name__ == '__main__':
    # df = pd.read_csv('../data/ESC-50-master/meta/esc50.csv', sep=',').sort_values(by=['filename'])
    # 37_40
    num_cpus = int(os.cpu_count())
    dev_set_size = 5
    num_prototypes = 5
    layer_idx_list = [3,7,17]
    # layer_idx_list = [3, 7, 8, 11, 14, 17, 19, 21]
    cache = True
    # version='ESC-10_0_21_v1_trimmed'
    # version='ESC-10_40_41_v1_trimmed'
    # version='ESC-10_v1_trimmed'
    # version='ESC-10_v1_10_11'
    # version='ESC-10_layer.3.7.17_class.10.11'
    version='ESC-10_v5'
    # version='ESC-10_test'
    df = pd.read_csv('../data/ESC-10/meta/esc10.csv', sep=',').sort_values(by=['filename'])
    df = df[df['esc10']]
    df = df[['filename', 'target', 'category']]
    # df = df[(df['target'] == 10) | (df['target'] == 11)]
    # df = df[(df['target'] == 0) | (df['target'] == 21)]
    # df = df[(df['target'] == 40) | (df['target'] == 41)]
    df = df.reset_index(drop=True)

    dataset = RawAudioDataset.load_all_data("../data/ESC-10/audio", meta_df=df)

    print('Dataset Size: ', len(dataset))
    # dataset = AudioDataset.load_all_data("../data/ESC-10/audio_40_41")

    afs = construct_soundnet_affinity_matrices(dataset, layer_idx_list,
                                            num_prototypes=num_prototypes,
                                            cache=cache,
                                            version=version)
    print("Complete")
    map_dict = dict(zip(np.unique(df['target']).tolist(), np.arange(np.unique(df['target']).size).tolist()))

    dev_set_indices = []
    dev_set_labels = []
    for unique_label in np.unique(df['target']):
        cur_target = df[df['target'] == unique_label].index.values[:dev_set_size]
        dev_set_indices.extend(cur_target.tolist())
        dev_set_labels.extend(df.iloc[cur_target]['target'].values.tolist())

    dev_set_labels = [map_dict[k] for k in dev_set_labels]
    
    # for filename in df['filename'].values.tolist():
    #     shutil.copyfile(os.path.join('../data/ESC-10/audio', filename),
    #         os.path.join('../data/ESC-10/audio_40_41', filename))

    y_true = [map_dict[k] for k in df['target'].values.tolist()]
    prob = infer_labels(afs, dev_set_indices, dev_set_labels)
    pred_labels = np.argmax(prob, axis=1).astype(int)

    print("accuracy", accuracy_score(y_true, pred_labels))
    print(confusion_matrix(y_true, pred_labels))

    import pdb; pdb.set_trace()

    print("---End of test_soundnet---")