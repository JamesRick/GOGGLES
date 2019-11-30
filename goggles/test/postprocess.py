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

# def main(layer_idx_list=[3,7,17],
#          num_prototypes=10,
#          dev_set_size=5,
#          model_name='soundnet',
#          cache=False,
#          dataset_name='ESC-10',
#          version='v0',
#          seed=151,
#          random_targets=True):

def main():
    goggles_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(goggles_dir, 'output')

    output_dict = {
                   'accuracy': [],
                   'balanced_accuracy': [],
                   'precision': [],
                   'recall': [],
                   'f1': [],
                   'dataset_name': [],
                   'model_name': [],
                   'layer_idx_list': [],
                   'num_prototypes': [],
                   'dev_set_size': [],
                   'version': [],
                   'seed': []
                  }

    for dataset_dir in os.listdir(output_dir):
        dataset_path = os.path.join(output_dir, dataset_dir)
        for model_dir in os.listdir(dataset_path):
            model_path = os.path.join(dataset_path, model_dir)
            for version_dir in os.listdir(model_path):
                version_path = os.path.join(model_path, version_dir)
                for dk_dir in os.listdir(version_path):
                    dk_path = os.path.join(version_path, dk_dir)
                    for seed_dir in os.listdir(dk_path):
                        seed_path = os.path.join(dk_path, seed_dir)
                        output_pkl_file = os.path.join(seed_path, 'output_dict.pkl')
                        with open(output_pkl_file, 'rb') as inp_fle:
                            pkl_dict = pkl.load(inp_fle)

                        if pkl_dict['dev_set_size'] == 5:
                            for key in pkl_dict.keys():
                                output_dict[key].append(pkl_dict[key])

    import pdb; pdb.set_trace()
    df = pd.DataFrame(output_dict)
    results_dir = os.path.join(goggles_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    df.to_csv(os.path.join(results_dir, 'full_results.csv'), sep=',')

    # new_df = pd.DataFrame(columns=["accuracy", "dataset_name", "model_name", "layer_idx_list",  "num_prototypes",  "dev_set_size"])
    df_group = df.groupby(["dataset_name", "model_name", "num_prototypes", "dev_set_size"])
    accuracy_df = pd.DataFrame({'accuracy' : df_group['accuracy'].mean()}).reset_index()
    balanced_df = pd.DataFrame({'balanced_accuracy' : df_group['balanced_accuracy'].mean()}).reset_index()
    precision_df = pd.DataFrame({'precision' : df_group['precision'].mean()}).reset_index()
    recall_df = pd.DataFrame({'recall' : df_group['recall'].mean()}).reset_index()
    f1_df = pd.DataFrame({'f1' : df_group['f1'].mean()}).reset_index()

    urban_soundnet_5a = accuracy_df[(accuracy_df['dataset_name'] == 'UrbanSound8K') & (accuracy_df['model_name'] == 'soundnet') & (accuracy_df['dev_set_size'] == 5)].copy()
    urban_vggish_5a = accuracy_df[(accuracy_df['dataset_name'] == 'UrbanSound8K') & (accuracy_df['model_name'] == 'vggish') & (accuracy_df['dev_set_size'] == 5)].copy()

    urban_soundnet_5b = balanced_df[(balanced_df['dataset_name'] == 'UrbanSound8K') & (balanced_df['model_name'] == 'soundnet') & (balanced_df['dev_set_size'] == 5)].copy()
    urban_vggish_5b = balanced_df[(balanced_df['dataset_name'] == 'UrbanSound8K') & (balanced_df['model_name'] == 'vggish') & (balanced_df['dev_set_size'] == 5)].copy()

    urban_soundnet_5p = precision_df[(precision_df['dataset_name'] == 'UrbanSound8K') & (precision_df['model_name'] == 'soundnet') & (precision_df['dev_set_size'] == 5)].copy()
    urban_vggish_5p = precision_df[(precision_df['dataset_name'] == 'UrbanSound8K') & (precision_df['model_name'] == 'vggish') & (precision_df['dev_set_size'] == 5)].copy()

    urban_soundnet_5r = recall_df[(recall_df['dataset_name'] == 'UrbanSound8K') & (recall_df['model_name'] == 'soundnet') & (recall_df['dev_set_size'] == 5)].copy()
    urban_vggish_5r = recall_df[(recall_df['dataset_name'] == 'UrbanSound8K') & (recall_df['model_name'] == 'vggish') & (recall_df['dev_set_size'] == 5)].copy()

    urban_soundnet_5f = f1_df[(f1_df['dataset_name'] == 'UrbanSound8K') & (f1_df['model_name'] == 'soundnet') & (f1_df['dev_set_size'] == 5)].copy()
    urban_vggish_5f = f1_df[(f1_df['dataset_name'] == 'UrbanSound8K') & (f1_df['model_name'] == 'vggish') & (f1_df['dev_set_size'] == 5)].copy()

    import pdb; pdb.set_trace()
    urban_soundnet_5a['num_afs'] = (urban_soundnet_5a.loc[:, 'num_prototypes'] * 3)
    urban_vggish_5a['num_afs'] = urban_vggish_5a.loc[:, 'num_prototypes'] * 4

    urban_soundnet_5b['num_afs'] = urban_soundnet_5b.loc[:, 'num_prototypes'] * 3
    urban_vggish_5b['num_afs'] = urban_vggish_5b.loc[:, 'num_prototypes'] * 4

    urban_soundnet_5p['num_afs'] = urban_soundnet_5p.loc[:, 'num_prototypes'] * 3
    urban_vggish_5p['num_afs'] = urban_vggish_5p.loc[:, 'num_prototypes'] * 4

    urban_soundnet_5r['num_afs'] = urban_soundnet_5r.loc[:, 'num_prototypes'] * 3
    urban_vggish_5r['num_afs'] = urban_vggish_5r.loc[:, 'num_prototypes'] * 4

    urban_soundnet_5f['num_afs'] = urban_soundnet_5f.loc[:, 'num_prototypes'] * 3
    urban_vggish_5f['num_afs'] = urban_vggish_5f.loc[:, 'num_prototypes'] * 4

    fx, fy = urban_vggish_5f['num_afs'].values, urban_vggish_5f['f1'].values
    bx, by = urban_vggish_5b['num_afs'].values, urban_vggish_5b['balanced_accuracy'].values
    plt.figure()
    plt.scatter(fx, fy, marker='x', c='b'); plt.plot(fx, fy); plt.ylim(ymin=0.5, ymax=1.0)
    plt.scatter(bx, by, marker='x', c='r'); plt.plot(bx, by); plt.ylim(ymin=0.5, ymax=1.0)
    plt.show()

    import pdb; pdb.set_trace()
    # plt.plot(balanced_df[(balanced_df['dataset_name'] == 'UrbanSound8K') & (balanced_df['model_name'] == 'vggish') & (balanced_df['dev_set_size'] == 5)]['balanced_accuracy'].values); plt.ylim(ymin=0.0, ymax=1.0); plt.show()
    # plt.plot(balanced_df[(balanced_df['dataset_name'] == 'UrbanSound8K') & (balanced_df['model_name'] == 'vggish') & (balanced_df['num_prototypes'] == 5)]['balanced_accuracy'].values); plt.ylim(ymin=0.0, ymax=1.0); plt.show()
    # plt.plot(us_df['num_prototypes'].values.flatten(), us_df['balanced_accuracy'].values.flatten())
    # plt.savefig('pic.png')
    # plt.show()
    # for dev_i in [5]:
    #     for num_proto_i in np.arange(1, 11):
    #         for model_i in ['soundnet', 'vggish']:
    #             for data_i in ['ESC-10', 'UrbanSound8K']:
    #                 cur_df = df[(df['dataset_name'] == data_i) & (df['model_name'] == model_i) & (df['num_prototypes'] == model_i) & (df['dev_set_size'] == num_proto_i)]

    import pdb; pdb.set_trace()
    print("---End of postprocess---")

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
    #
    # print("---End of test_audio---")

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
