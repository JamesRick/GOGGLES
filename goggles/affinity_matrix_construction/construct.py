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

def construct_audio_affinity_matrices(dataset, cache=True, version='v0'):
    """
    :param GogglesDataset instance
    :return: a list of affinity matrices
    """
    matrix_list = []
    for layer_idx in [2,5,10,15]:#all max pooling layers
        matrix_list.extend(audio_nn_AFs(dataset, layer_idx, 10, cache, version=version))
    return matrix_list
