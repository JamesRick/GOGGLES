from goggles import construct_image_affinity_matrices, GogglesDataset, infer_labels
from goggles.affinity_matrix_construction.construct import construct_audio_affinity_matrices
from goggles.affinity_matrix_construction.audio_AF.pretrained_models.vggish_wrapper import VGGish_wrapper
from goggles.affinity_matrix_construction.audio_AF.pretrained_models.soundnet_wrapper import Soundnet_wrapper
from goggles.utils.svm_models import VGGish_svm
from goggles.utils.svm_models import Soundnet_svm
from goggles.utils.dataset import AudioDataset

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn import preprocessing
from sklearn import decomposition
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
import multiprocessing as mp
import matplotlib.pyplot as plt

def load_df(goggles_dir, dataset_name):
    if dataset_name == 'ESC-10':
        dataset_csv = os.path.join(goggles_dir, 'data', dataset_name, 'meta/esc10.csv')
        dataset_audio = os.path.join(goggles_dir, 'data', dataset_name, 'audio')
        df = pd.read_csv(dataset_csv, sep=',')
        df = df[df['esc10']]
        df = df[['filename', 'fold', 'target', 'category']]
        df = df.sort_values(by=['category'])
    elif dataset_name == 'ESC-50':
        dataset_csv = os.path.join(goggles_dir, 'data', dataset_name, 'meta/esc50.csv')
        dataset_audio = os.path.join(goggles_dir, 'data', dataset_name, 'audio')
        df = pd.read_csv(dataset_csv, sep=',')
        df = df[['filename', 'fold', 'target', 'category']]
        df = df.sort_values(by=['category'])
    elif dataset_name == 'UrbanSound8K':
        dataset_csv = os.path.join(goggles_dir, 'data', dataset_name, 'metadata/UrbanSound8K.csv')
        dataset_audio = os.path.join(goggles_dir, 'data', dataset_name, 'audio')
        df = pd.read_csv(dataset_csv, sep=',')
        df = df[['slice_file_name', 'fold', 'classID', 'class']]
        df.columns = ['filename', 'fold', 'target', 'category']
        df = df.sort_values(by=['category'])
    elif dataset_name == 'LITIS':
        dataset_csv = os.path.join(goggles_dir, 'data', dataset_name, 'relation_wav_examples.txt')
        dataset_audio = os.path.join(goggles_dir, 'data', dataset_name, 'data_rouen')
        df = pd.read_csv(dataset_csv, sep=' ', header=None)
        df.columns=['filename','fileindex']
        df['category'] = df['filename'].str.extract('^([^\d]*).*')
        df['target'] = (df.groupby(['category']).cumcount()==0).astype(int)
        df['target'] = df['target'].cumsum()
        df['fold'] = 1
        df = df[['filename', 'fold', 'target', 'category']]
        df['filename'] = df['filename'].str.replace('\t', '')
        df = df.sort_values(by=['category'])
    elif dataset_name == 'TUT-UrbanAcousticScenes':
        dataset_csv = os.path.join(goggles_dir, 'data', dataset_name, 'meta/TUT-UrbanAcousticScenes.csv')
        dataset_audio = os.path.join(goggles_dir, 'data', dataset_name, 'audio')
        df = pd.read_csv(dataset_csv, sep='\t')
        df['filename'] = df.loc[:, 'filename'].apply(lambda x: re.sub('audio/', '', x))
        target_dict = dict(zip(np.unique(df['scene_label']), np.arange(np.unique(df['scene_label']).shape[0])))
        df['fold'] = 1
        df['target'] = df['scene_label'].map(target_dict)
        df = df[['filename', 'fold', 'target', 'scene_label']]
        df.columns = ['filename', 'fold', 'target', 'category']
        df = df.sort_values(by=['category'])
    else:
        raise Exception("Dataset " + dataset_name + " not found.\n\
                        Currently implemented datasets are:\n\
                        1.\tESC-10\n\
                        2.\tESC-50\n\
                        3.\tUrbanSound8K\n\
                        4.\tLITIS\n\
                        5.\tTUT-UrbanAcousticScenes\n")
    return dataset_audio, df

def main(layer_idx_list=[3,7,17],
         num_prototypes=10,
         dev_set_size=5,
         model_name='soundnet',
         cache=False,
         dataset_name='ESC-10',
         version='v0',
         seed=151,
         random_targets=True,
         classes=None):
    np.random.seed(seed)
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
    elif model_name == 'vggish_svm':
        model = VGGish_svm(VGGish_wrapper())
    elif model_name == 'soundnet_svm':
        model = Soundnet_svm(Soundnet_wrapper())
    else:
        raise Exception("Model " + model_name + " not found.\n\
                        Currently implemented models are:\n\
                        1.\tvggish\n\
                        2.\tsoundnet\n\
                        3.\tvggish_svm\n\
                        4.\tsoundnet_svm\n")
    #
    dataset_audio, df = load_df(goggles_dir, dataset_name)

    if random_targets and classes is None:
        targets = np.random.choice(np.unique(df['target'].values), size=2, replace=False)
        target_mask = np.zeros(df['target'].shape).astype(bool)
        for target in targets:
            target_mask = (df['target'] == target) | target_mask
        df = df[target_mask].sort_values(by=['category']).reset_index(drop=True)
    else:
        class_mask = np.zeros(df['target'].shape).astype(bool)
        for class_name in classes:
            class_mask = (df['category'] == class_name) | class_mask
        df = df[class_mask].sort_values(by=['category']).reset_index(drop=True)

    # Reduce the number of instances since there are only 5 days left for running.
    sizes = df.groupby(['category']).size()
    total = sizes.sum()
    max_instance_size = 200
    if total > max_instance_size:
        reduce_mask = np.zeros(df.shape[0])
        for idx, item in sizes.iteritems():
            print(idx, item, int((item / total) * max_instance_size))
            reduce_mask[np.random.choice(df[df['category'] == idx].index.values, size=int((item / total) * max_instance_size), replace=False)] = 1
        df = df[reduce_mask.astype(bool)].sort_values(by=['category']).reset_index(drop=True)

    classes = np.unique(df['category'].values)
    classes_name = re.sub(", ", "_", re.sub(r"\[|\]", "", str(classes)))
    classes_name = re.sub(" ", "-", re.sub("'", "", classes_name))
    balanced_accuracies = []
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    cf_matrices = []
    print("Running with classes: %s" % (str(classes))); sys.stdout.flush()

    df = df.sort_values(by=['category']).reset_index(drop=True)
    map_dict = dict(zip(np.unique(df['target']).tolist(), np.arange(np.unique(df['target']).size).tolist()))
    dev_set_indices = []
    dev_set_labels = []
    for unique_label in np.unique(df['target']):
        cur_target = np.random.choice(df[df['target'] == unique_label].index.values, size=dev_set_size, replace=False)
        print("For Class: ", map_dict[unique_label], "Using Indices: ", cur_target); sys.stdout.flush()
        dev_set_indices.extend(cur_target.tolist())
        dev_set_labels.extend(df.iloc[cur_target]['target'].values.tolist())
    dev_set_labels = [map_dict[k] for k in dev_set_labels]
    y_true = np.array([map_dict[k] for k in df['target'].values.tolist()])

    if dataset_name == 'ESC-10' or dataset_name == 'ESC-50' or dataset_name == 'LITIS' or dataset_name == 'TUT-UrbanAcousticScenes':
        dataset = AudioDataset.load_all_data(dataset_audio, model.preprocess, name=dataset_name, meta_df=df)
    elif dataset_name == 'UrbanSound8K':
        dataset = AudioDataset.load_all_data(dataset_audio, model.preprocess, name=dataset_name, meta_df=df)
#
    # plt.imshow(afs[1], origin='lower'); plt.show()
    i = 1
    afs=[]
    if model_name == 'soundnet' or model_name == 'vggish':
        afs = construct_audio_affinity_matrices(dataset, layer_idx_list, model,
                                                    num_prototypes=num_prototypes,
                                                    cache=cache,
                                                    version=version,
                                                    seed=str(seed),
                                                    classes=classes_name)
#
        # I was using these lines to plot different heatmaps of the affinity functions.
        # import pdb; pdb.set_trace()
        # plt.imshow(afs[0], origin='lower'); plt.show()
        prob = infer_labels(afs, dev_set_indices, dev_set_labels)
        kmeans_pred_labels = KMeans(n_clusters=len(classes)).fit_predict(np.hstack(afs))
        pred_labels = np.argmax(prob, axis=1).astype(int)
    elif model_name == 'soundnet_svm':
        X = []
        y = []
        for j in trange(len(dataset)):
            wav_data, _ = dataset[j]
            X.append(model.get_svm_data(wav_data, layer_idx=layer_idx_list[0]))
            y.append(df['target'].loc[j])
        X = np.array(X).squeeze()
        y = np.array([map_dict[k] for k in y])
        X_train, y_train = X[dev_set_indices, :], y[dev_set_indices]
        model.fit(X_train, y_train)
        pred_labels = model.predict(X)
    elif model_name == 'vggish_svm':
        X = []
        y = []
        for j in trange(len(dataset)):
            wav_frames, wav_full = dataset[j]
            X_example = []
            y_example = []
            example = []
            for wav_data in wav_frames:
                if wav_data.min() != wav_data.max():
                    example.append(model.get_svm_data(wav_data))
            X.append(np.array(example))
            y.append(df['target'].loc[j])
        X = np.array(X).squeeze()
        y = np.array([map_dict[k] for k in y])
        selected_dev_X, selected_dev_y = X[dev_set_indices], y[dev_set_indices]
        selected_train_X = []
        selected_train_y = []
        for x_instance, y_label in zip(selected_dev_X, selected_dev_y):
            for x_features in x_instance:
                selected_train_X.append(x_features.flatten())
                selected_train_y.append(y_label)
        X_train, y_train = np.array(selected_train_X), np.array(selected_train_y)
        model.fit(X_train, y_train)
        full_X = []
        full_y = []
        idx = 0
        idx_split = 0
        vote_indices = {}
        for x_instance, y_label in zip(X, y):
            vote_list = []
            for x_features in x_instance:
                full_X.append(x_features.flatten())
                full_y.append(y_label)
                vote_list.append(idx_split)
                idx_split += 1
            vote_indices[idx] = np.array(vote_list)
            idx += 1
        temp_y_pred = model.predict(full_X)
        pred_labels = []
        for key in vote_indices:
            prediction = stats.mode(temp_y_pred[vote_indices[key]]).mode[0]
            pred_labels.append(prediction)
        pred_labels = np.array(pred_labels)

    mask = np.ones(len(pred_labels))
    mask[np.array(dev_set_indices)] = 0
    pred_labels = pred_labels[mask.astype(bool)]
    y_true = y_true[mask.astype(bool)]

    print("")
    print("Complete")
    sys.stdout.flush()
    balanced_accuracy = balanced_accuracy_score(y_true, pred_labels)
    accuracy = accuracy_score(y_true, pred_labels)
    if model_name == 'soundnet' or model_name == 'vggish':
        kmeans_pred_labels = kmeans_pred_labels[mask.astype(bool)]
        kmeans_accuracy = max(balanced_accuracy_score(y_true, kmeans_pred_labels), 1 - balanced_accuracy_score(y_true, kmeans_pred_labels))
    else:
        kmeans_accuracy = max(balanced_accuracy_score(y_true, pred_labels), 1 - balanced_accuracy_score(y_true, pred_labels))
    print("Kmeans Accuracy: " + str(kmeans_accuracy))
    precision = precision_score(y_true, pred_labels, average='weighted')
    recall = recall_score(y_true, pred_labels, average='weighted')
    f1 = f1_score(y_true, pred_labels, average='weighted')
    cf_matrix = confusion_matrix(y_true, pred_labels)

    print("Fold " + str(i) + " Accuracy: " + str(accuracy))
    print("Fold " + str(i) + " Balanced Accuracy: " + str(balanced_accuracy))
    print("Confusion Matrix")
    print(cf_matrix)
    sys.stdout.flush()

    output_dict = {
                   'accuracy': accuracy,
                   'balanced_accuracy': balanced_accuracy,
                   'precision': precision,
                   'recall': recall,
                   'f1': f1,
                   'kmeans_accuracy': kmeans_accuracy,
                   'dataset_name': dataset_name,
                   'model_name': model_name,
                   'layer_idx_list': layer_idx_list,
                   'num_prototypes': num_prototypes,
                   'num_afs': len(afs),
                   'dev_set_size': dev_set_size,
                   'version': version,
                   'seed': seed,
                   'classes': classes
                  }

    layer_idx_name = re.sub(", ", "_", re.sub(r"\[|\]", "", str(layer_idx_list)))
    output_dir = os.path.join(goggles_dir, 'output')
    run_output_dir = os.path.join(output_dir, str(dataset_name),
                        str(model_name) + '_' + layer_idx_name + '_' + classes_name,
                        str(version), 'd' + str(dev_set_size) + '_k' + str(num_prototypes),
                        'seed_' + str(seed))
    os.makedirs(run_output_dir, exist_ok=True)
    with open(os.path.join(run_output_dir, 'output_dict.pkl'), 'wb') as out_fle:
        pkl.dump(output_dict, out_fle)

    print("---End of run_audio---")
    sys.stdout.flush()


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
    parser.add_argument('--classes',
                        type=str,
                        nargs='+',
                        default=None,
                        required=False)

    args = parser.parse_args()
    main(**args.__dict__)
