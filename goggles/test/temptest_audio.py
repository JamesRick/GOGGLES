from goggles import construct_image_affinity_matrices, GogglesDataset, infer_labels
from goggles.affinity_matrix_construction.construct import construct_audio_affinity_matrices
from goggles.affinity_matrix_construction.audio_AF.pretrained_models.vggish_wrapper import VGGish_wrapper
from goggles.affinity_matrix_construction.audio_AF.pretrained_models.soundnet_wrapper import Soundnet_wrapper
from goggles.utils.dataset import AudioDataset

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import os
import sys
import time
import tqdm
import torch
import shutil
import datetime

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
         model_name='soundnet',
         cache=False,
         dataset_name='ESC-10',
         version='v0',
         seed=151,
         random_targets=True):

    np.random.seed(seed)
    cur_time = datetime.datetime.now()
    print("\n\nStart Time:", cur_time)
    print("\n")
    sys.stdout.flush()

    start_time = time.time()
    goggles_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("Model: " + model_name)
    print("Using cuda: " + str(torch.cuda.is_available()))
    sys.stdout.flush()

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
        sys.stdout.flush()
        raise Exception("Model " + model_name + " not found.\n\
                        Currently implemented models are:\n\
                        1.\tvggish\n\
                        2.\tsoundnet\n")
    sys.stdout.flush()

    if dataset_name == 'ESC-10':
        dataset_csv = os.path.join(goggles_dir, 'data', dataset_name, 'meta/esc10.csv')
        dataset_audio = os.path.join(goggles_dir, 'data', dataset_name, 'audio')
        df = pd.read_csv(dataset_csv, sep=',')
        df = df[df['esc10']]
        df = df[['filename', 'fold', 'target', 'category']]
        df = df.sort_values(by=['filename'])
    elif dataset_name == 'ESC-50':
        dataset_csv = os.path.join(goggles_dir, 'data', dataset_name, 'meta/esc50.csv')
        dataset_audio = os.path.join(goggles_dir, 'data', dataset_name, 'audio')
        df = pd.read_csv(dataset_csv, sep=',')
        df = df[['filename', 'fold', 'target', 'category']]
        df = df.sort_values(by=['filename'])
    elif dataset_name == 'UrbanSound8K':
        dataset_csv = os.path.join(goggles_dir, 'data', dataset_name, 'metadata/UrbanSound8K.csv')
        dataset_audio = os.path.join(goggles_dir, 'data', dataset_name, 'audio')
        df = pd.read_csv(dataset_csv, sep=',')
        # import pdb; pdb.set_trace()
        df = df[['slice_file_name', 'fold', 'classID', 'class']]
        df.columns = ['filename', 'fold', 'target', 'category']
        df = df.sort_values(by=['filename'])
        # df[df['filename'] == '162540-1-0-0.wav']
        # waveform = dataset_audio + '/fold1/162540-1-0-0.wav'
        # wav_data = VGGish_wrapper.preprocess(waveform)
        # import pdb; pdb.set_trace()

        # df = pd.read_csv('../data/UrbanSound8K/meta/UrbanSound8K.csv', sep=',')
    elif dataset_name == 'Litis_Rounen':
        pass
    elif dataset_name == 'Vox':
        pass
    else:
        sys.stdout.flush()
        raise Exception("Dataset " + dataset_name + " not found.\n\
                        Currently implemented datasets are:\n\
                        1.\tESC-10\n\
                        2.\tESC-50\n\
                        3.\tUrbanSound8K\n")
                        #2.\t\n")
    sys.stdout.flush()

    # num_cpus = int(os.cpu_count())
    if random_targets:
        targets = np.random.choice(np.unique(df['target'].values), size=2, replace=False)
    df = df[(df['target'] == targets[0]) | (df['target'] == targets[1])]
    df = df.reset_index(drop=True)
    classes = np.unique(df['category'].values)
    accuracies = []
    cf_matrices = []
    print("Running with classes: %s, %s" % (classes[0], classes[1]))
    sys.stdout.flush()

    for i in np.unique(df['fold']):
        cur_df = df[df['fold'] == i].reset_index(drop=True)

        if dataset_name == 'ESC-10' or dataset_name == 'ESC-50':
            dataset = AudioDataset.load_all_data(dataset_audio, model.preprocess, name=dataset_name, meta_df=cur_df)
        elif dataset_name == 'UrbanSound8K':
            dataset = AudioDataset.load_all_data(os.path.join(dataset_audio, 'fold' + str(i)), model.preprocess, name=dataset_name, meta_df=cur_df)

        # afs = construct_audio_affinity_matrices(dataset, layer_idx_list, model,
        #                                             num_prototypes=num_prototypes,
        #                                             cache=cache,
        #                                             version=version,
        #                                             seed=str(seed),
        #                                             fold=str(i))
        print("")
        print("Complete")
        sys.stdout.flush()

        X = []
        y = []
        for j in tqdm.tqdm(range(len(dataset))):
            wav_data, sr = lb.load(dataset_audio + '/fold' + str(i) + '/' + dataset.audio_filename_list[j], mono=True)
            min_length = sr * 4
            max_length = sr * 20
            if wav_data.shape[0] < min_length:
                wav_data = np.concatenate((wav_data, np.zeros(min_length - wav_data.shape[0])))
            wav_data = wav_data[:max_length]
            wav_data = wav_data.reshape(1, -1, 1)
            # X.append(model.get_svm_data(dataset[i], layer_idx=17))
            X.append(model.get_svm_data((wav_data,), layer_idx=17))
            y.append(cur_df['target'].loc[j])
            # if X[j].shape[1] != 2816:
            #     print(j, X[j].shape[1])
            #     print(wav_data.shape)
            #     import pdb; pdb.set_trace()
        X = np.array(X).squeeze()
        y = np.array(y)

        # Setting up the development set for cluster to class assignment
        map_dict = dict(zip(np.unique(cur_df['target']).tolist(), np.arange(np.unique(cur_df['target']).size).tolist()))
        dev_set_indices = []
        dev_set_labels = []
        for unique_label in np.unique(cur_df['target']):
            cur_target = np.random.choice(cur_df[cur_df['target'] == unique_label].index.values, size=dev_set_size, replace=False)
            print("For Class: ", map_dict[unique_label], "Using Indices: ", cur_target)
            sys.stdout.flush()
            # cur_target = df[df['target'] == unique_label].index.values[:dev_set_size]
            dev_set_indices.extend(cur_target.tolist())
            dev_set_labels.extend(cur_df.iloc[cur_target]['target'].values.tolist())
        dev_set_labels = [map_dict[k] for k in dev_set_labels]
        y_true = [map_dict[k] for k in cur_df['target'].values.tolist()]

        # import pdb; pdb.set_trace()
        y = np.array([map_dict[k] for k in y])
        X_train, y_train = X[dev_set_indices, :], y[dev_set_indices]
        svm_clf = svm.SVC(kernel='linear')
        svm_model = svm_clf.fit(X_train, y_train)
        y_pred = svm_model.predict(X)

        # import pdb; pdb.set_trace()

        accuracy = accuracy_score(y_true, y_pred)
        cf_matrix = confusion_matrix(y_true, y_pred)
        accuracies.append(accuracy)
        cf_matrices.append(cf_matrix)
        print("Fold " + str(i) + " Accuracy: " + str(accuracy))
        print("Confusion Matrix")
        print(cf_matrix)
        sys.stdout.flush()

        # import pdb; pdb.set_trace()

        # prob = infer_labels(afs, dev_set_indices, dev_set_labels)
        # pred_labels = np.argmax(prob, axis=1).astype(int)
        # accuracy = accuracy_score(y_true, pred_labels)
        # cf_matrix = confusion_matrix(y_true, pred_labels)
        # accuracies.append(accuracy)
        # cf_matrices.append(cf_matrix)
        # print("Fold " + str(i) + " Accuracy: " + str(accuracy))
        # print("Confusion Matrix")
        # print(cf_matrix)
        # sys.stdout.flush()

    cf_matrices = np.array(cf_matrices)
    row_sums = cf_matrices.sum(axis=2).reshape(-1,2,1)
    cf_matrices_norm = cf_matrices / row_sums
    print("Average Accuracy: " + str(np.array(accuracies).mean()))

    print("Average Confusion Matrix")
    print(cf_matrices.mean(axis=0))

    print("Average Confusion Matrix Norm")
    print(cf_matrices_norm.mean(axis=0))

    time_diff = time.time() - start_time
    print("\n\nEnd Time:", cur_time + datetime.timedelta(seconds=int(time_diff)))
    print("\n\nRun Time:", datetime.timedelta(seconds=int(time_diff)))
    sys.stdout.flush()

    # output_dict = {
    #                'accuracy': accuracy.mean(),
    #                'dataset_name': dataset_name,
    #                'model_name': model_name,
    #                'layer_idx_list': layer_idx_list,
    #                'num_prototypes': num_prototypes,
    #                'dev_set_size': dev_set_size,
    #                'version': version,
    #                'seed': seed
    #               }
    #
    # output_dir = os.path.join(goggles_dir, 'output')
    # run_output_dir = os.path.join(output_dir, str(dataset_name), str(model_name), str(version), 'd' + str(dev_set_size) + '_k' + str(num_prototypes), 'seed_' + str(seed))
    # os.makedirs(run_output_dir, exist_ok=True)
    # with open(os.path.join(run_output_dir, 'output_dict.pkl'), 'wb') as out_fle:
    #     pkl.dump(output_dict, out_fle)

    print("---End of test_audio---")

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
                        default='soundnet',
                        required=False)
    parser.add_argument('--cache',
                        type=bool,
                        default=False,
                        required=False)
    parser.add_argument('--dataset_name',
                        type=str,
                        default='ESC-10',
                        required=False)
    parser.add_argument('--seed',
                        type=int,
                        default=151,
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
