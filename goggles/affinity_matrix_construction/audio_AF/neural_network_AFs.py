import os
import sys
import time
import torch
import torch.nn.functional as F
from tqdm import trange
import numpy as np
from goggles.utils.constants import *
from .pretrained_models.vggish_wrapper import VGGish_wrapper
from .pretrained_models.soundnet_wrapper import Soundnet_wrapper
from multiprocessing import Pool

import torch.nn as nn

_make_cuda = lambda x: x.cuda() if torch.cuda.is_available() else x

class Context:
    def __init__(self, model, dataset, layer_idx):
        self.model = model
        self.dataset = dataset

        self._layer_idx = layer_idx
        self._model_out_dict = dict()

    def get_model_output(self, audio_idx):
        """
        VGGish Steps:
        1. Retrieves the preprocessed spectrogram frames from the AudioDatset object
        2. For every spectrogram frame retrieved:
                Retrieve the output from layer _layer_idx of the model.
                Add that output to a frame dictionary, with the key being its frame_idx.
        3. Add the frame dictionary to the _model_out_dict with audio_idx as the key.
        4. Return the framed spectrogram's output of the ith layer of the model for this audio_idx.

        SoundNet Steps:
        1. Retrieves the audio waveform
        2. Add the audio waveform to the frame dictionary
        3. return the audio waveform output of the ith layer of the model for this audio_idx.
        """
        if audio_idx not in self._model_out_dict:
            x_frames, x_full = self.dataset[audio_idx]
            frame_dict = dict()
            for frame_idx in range(x_frames.shape[0]):
                if x_frames[frame_idx].min() != x_frames[frame_idx].max():
                    x = x_frames[frame_idx]
                    x = torch.from_numpy(x).unsqueeze(dim=0)
                    x = x.view((1,) + x.size()).type('torch.FloatTensor')
                    x = _make_cuda(torch.autograd.Variable(x, requires_grad=False))
                    frame_dict[frame_idx] = \
                        self.model.forward(x, layer_idx=self._layer_idx).squeeze(dim=0)
            self._model_out_dict[audio_idx] = frame_dict
        z = self._model_out_dict[audio_idx]
        return z

def _get_patches(z, patch_idxs, normalize=False):
    """
    z: CxHxW
    patch_idxs: K
    """
    c = z.size(0)
    patches = z.view(c, -1).t()[patch_idxs]
    if normalize:
        patches = F.normalize(patches, dim=1)
    return patches

def _get_most_activated_channels(z, num_channels=5):
    """
    z: CxHxW
    """
    per_channel_max_activations, _ = z.max(1)[0].max(1)
    most_activated_channels = \
        torch.topk(per_channel_max_activations, num_channels)[1]
    return most_activated_channels

def _get_most_activated_patch_idxs_from_channels(z, channel_idxs):
    """
    z: CxHxW
    channel_idxs: K
    """
    k = channel_idxs.shape[0]
    most_activated_patch_idxs = z[channel_idxs].view(k, -1).max(1)[1]
    # pid_list = torch.unique(most_activated_patch_idxs, sorted=False).cpu().numpy().tolist()[::-1]
    pid_list = list(most_activated_patch_idxs.cpu().numpy())
    d_list = [(k - i - 1, p) for i, p in enumerate(reversed(pid_list))]
    d_list = list(sorted(d_list))
    rank,uid = list(zip(*d_list))
    return _make_cuda(torch.LongTensor(uid)), \
        _make_cuda(torch.LongTensor(rank))


def _get_score_matrix_for_audio(audio_idx, num_max_proposals, context):
    score_matrix = list()
    column_ids = list()
    z_frames = context.get_model_output(audio_idx)
    z = torch.cat(tuple([z_frame for z_frame in z_frames.values()]), 1)

    # Number of patches is the H x W of spectrogram.
    num_patches = z.size(1) * z.size(2)
    ch = _get_most_activated_channels(z, num_channels=num_max_proposals)

    # In the paper this portion is extracting the channel vectors using
    # (HxW) index of the maximum value of the the most activated channels.
    # Duplicate channel vectors should be dropped.
    #
    # In this code, the vectors are indexed using a 1-D index computed from
    # flattening the channel matrix to a one dimensional vector
    pids, ranks = _get_most_activated_patch_idxs_from_channels(z, ch)
    proto_patches = _get_patches(z, pids, normalize=True)
    for patch_idx, rank in zip(pids.cpu().numpy(), ranks.cpu().numpy()):
        column_ids.append([audio_idx, patch_idx, rank])
    for audio_idx_ in trange(len(context.dataset), position=0, leave=False):
        z_frames_ = context.get_model_output(audio_idx_)
        z_ = torch.cat(tuple([z_frame_ for z_frame_ in z_frames_.values()]), 1)
        num_patches = z_.size(1) * z_.size(2)
        spectrogram_patches = _get_patches(z_, range(num_patches), normalize=True)
        # max_pool = nn.MaxPool1d(5, stride=5)
        # spectrogram_patches = max_pool(spectrogram_patches.unsqueeze(dim=1)).squeeze()
        # compressed_proto = max_pool(proto_patches.unsqueeze(dim=1)).squeeze()
        scores = torch.matmul(spectrogram_patches, proto_patches.t()).max(0)[0]
        # scores = torch.matmul(spectrogram_patches, compressed_proto.t()).max(0)[0]
        scores = scores.cpu().numpy()
        score_matrix.append(scores)
    return np.array(score_matrix), column_ids


def audio_nn_AFs(dataset, layer_idx, model, num_max_proposals=10, cache=False, version='v0', seed='151', classes=None):
    """
    Computes the affinity scores between every instance in the dataset using the
    layer_idx as the layer to gather the top-k prototypes from.
    This method is called for every max pooling layer of the VGGish/Soundnet model.

    Arguments:
    dataset   -- The AudioDataset object containing the instances for labeling.
    layer_idx -- The layer index of the model to gather top-k prototypes from.
    num_max_proposals -- The number of prototypes to gather from the model, i.e. k.

    Keyword Arguments:
    cache   -- Declares saving and loading from cache file. (default False)
    version -- Version tag for the cache. Used in cache filename. (default 'v0')
    """

    affinity_matrix_list = [[] for _ in range(num_max_proposals)]
    out_filename = '.'.join([
    dataset.name,
    str(version),
    str(seed),
    str(classes),
    model.name + f'_wrapper_layer{layer_idx:02d}',
    f'k{num_max_proposals:02d}',
    'scores.npz'])
    out_dirpath = os.path.join(SCRATCH_DIR, 'scores')
    os.makedirs(out_dirpath, exist_ok=True)
    out_filepath = os.path.join(out_dirpath, out_filename)
    # This section tries to load data from the cache file if it exists.
    if cache:
        try:
            # Loads the cache
            print("Attempting to load cache: " + str(out_filepath))
            affinity_matrix_arr = np.load(out_filepath)['scores']

            # Added this statement here in case I didn't intend to run
            # caching while trying various combinations of specific classes
            # for a dataset. This will just end the run early.
            # (possibly throwing an error at some point)
            dataset_shape = num_max_proposals * len(dataset) * len(dataset)
            cache_shape = affinity_matrix_arr.shape[0] * affinity_matrix_arr.shape[1] * affinity_matrix_arr.shape[2]
            if dataset_shape != cache_shape:
                print(str(dataset_shape) + '!=' + str(cache_shape))
                return
            for i in range(num_max_proposals):
                affinity_matrix_list[i] = (np.squeeze(affinity_matrix_arr[i,:,:]))
            return affinity_matrix_list
        except Exception as e:
            print("Load Cache Failed")
            print("\n" + str(e) + "\n")

    # Loads the VGGish_wrapper model
    model = _make_cuda(model)
    print("\nLoaded model: ", model.name, "\n")
    context = Context(model=model, dataset=dataset, layer_idx=layer_idx)

    # Main Loop:
    #    Loop through every instance in the dataset.
    #    for each instance, compute a score matrix
    #    which is a matrix of affinity scores between
    #    the current example and every other example in
    #    the dataset. Affinity scores are computed for
    #    every affinity function so the matrix will be
    #    N X S, where N is the number of examples and
    #    S is the number of affinity functions.
    for audio_idx in trange(len(context.dataset), position=1):
        scores, cols = _get_score_matrix_for_audio(
            audio_idx, num_max_proposals, context)
        sys.stdout.flush()
        for i in range(min(num_max_proposals, scores.shape[1])):
            affinity_matrix_list[i].append(scores[:,i])
        #all_column_ids += cols
    for i in range(num_max_proposals):
        affinity_matrix_list[i] = np.array(affinity_matrix_list[i]).T
    # Saves to cache file if enabled
    if cache:
        print('saving output to %s' % out_filepath)
        np.savez(
           out_filepath, version=2,
           scores=np.array(affinity_matrix_list),
           num_max_proposals=num_max_proposals)
    return affinity_matrix_list
