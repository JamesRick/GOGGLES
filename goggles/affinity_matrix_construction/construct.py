import numpy as np
from goggles.affinity_matrix_construction.image_AF.neural_network_AFs import nn_AFs
from goggles.affinity_matrix_construction.audio_AF.neural_network_AFs import audio_nn_AFs


def construct_image_affinity_matrices(dataset,cache=True):
    """
    :param GogglesDataset instance
    :return: a list of affinity matrices
    """
    matrix_list = []
    for layer_idx in [4,9,16,23,30]:#all max pooling layers
        matrix_list.extend(nn_AFs(dataset,layer_idx,10,cache))
    return matrix_list

def construct_audio_affinity_matrices(dataset,
                                      layer_idx_list,
                                      model,
                                      num_prototypes=10,
                                      cache=True,
                                      version='v0',
                                      seed='151',
                                      classes=None):
    """
    :param dataset: AudioDataset instance
    :param layer_idx_list: list of layer indexes to gather AFs from
    :param model: The model instance used to gather AFs
    :param num_prototypes: The maximum number of prototypes per layer
    :param cache: Boolean to determine cache writes/reads
    :param version: Version tag for the cache filename
    :param seed: Seed string for the cache filename
    :param classes: classes string for the cache filename,
                    None if random targets was used
    :return: a list of affinity matrices
    """
    matrix_list = []
    for layer_idx in layer_idx_list:
        matrix_list.extend(audio_nn_AFs(dataset, layer_idx, model, num_prototypes, cache, version=version, seed=seed, classes=classes))
    return matrix_list
