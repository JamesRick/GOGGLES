from collections import OrderedDict

import numpy as np
import librosa as lb

import torch
import torch.nn as nn
from torchvision import models

import goggles.torch_soundnet.soundnet as soundnet

class Soundnet_wrapper(nn.Module):
    def __init__(self, freeze=True):
        super(Soundnet_wrapper, self).__init__()
        self.name = 'SoundNet'
        self._is_cuda = False

        # self.input_size = 224
        self.input_frame_size = 20 * 22050
        self.input_freq_size = 1

        base_model = soundnet.get_model(pretrained=True)

        features = list(base_model.features)
        self._features = nn.ModuleList(features).eval()

        embedding = [base_model.conv8_objs, base_model.conv8_scns]
        self._embedding = nn.Sequential(*embedding).eval()

        self._is_frozen = freeze
        self.freeze(freeze)

        self._config = None
        self._parse_config()

    def _make_cuda(self, x):
        return x.cuda() if self._is_cuda else x

    def _parse_config(self):
        self.zero_grad()

        x = self._make_cuda(torch.autograd.Variable(
            torch.rand(1, 1, self.input_frame_size, self.input_freq_size),
            requires_grad=False))

        self._config = OrderedDict()
        for i, layer in enumerate(self._features):
            x = layer(x)
            self._config[i] = (
                layer.__class__.__name__,
                tuple(x.size()[-3:]),)

        self.zero_grad()

    def cuda(self, device_id=None):
        self._is_cuda = True
        return super(Soundnet_wrapper, self).cuda(device_id)

    def freeze(self, freeze=True):
        requires_grad = not freeze
        for parameter in self.parameters():
            parameter.requires_grad = requires_grad

        self._is_frozen = freeze

    def forward(self, x, layer_idx=None):
        if layer_idx is None:
            layer_idx = len(self._config) - 1

        for i, model in enumerate(self._features):
            x = model(x)
            if i == layer_idx:
                return x

    def embed(self, x, layer='logits'):
        assert layer in ['pre_fc', 'logits']
        x = self.forward(x)
        x = x.view(x.size(0), -1)
        if layer == 'logits':
            x = self._embedding(x)
        return x

    def get_svm_data(self, x, layer_idx=None):
        assert layer_idx is not None
        x = x[0]
        x = torch.from_numpy(x)
        x = x.view((1,) + x.size()).type('torch.FloatTensor')
        x = torch.autograd.Variable(x, requires_grad=False)
        x = self.forward(x, layer_idx=layer_idx)
        x = x.view(x.size(0), -1)
        return x.numpy()

    def get_layer_type(self, layer_idx):
        return self._config[layer_idx][0]

    def get_layer_output_dim(self, layer_idx):
        return self._config[layer_idx][1]

    def get_receptive_field(self, patch, layer_idx=None):
        is_originally_frozen = self._is_frozen
        self.zero_grad()
        self.freeze(False)
        batch_shape = (1, 1, self.input_frame_size, self.input_freq_size)

        x = self._make_cuda(torch.autograd.Variable(
            torch.rand(*batch_shape), requires_grad=True))
        z = self.forward(x, layer_idx=layer_idx)
        z_patch = patch.forward(z)

        torch.sum(z_patch).backward()

        rf = x.grad.data.cpu().numpy()
        rf = rf[0, 0]
        rf = list(zip(*np.where(np.abs(rf) > 1e-6)))

        (i_nw, j_nw), (i_se, j_se) = rf[0], rf[-1]

        rf_w, rf_h = (j_se - j_nw + 1,
                      i_se - i_nw + 1)

        self.zero_grad()
        self.freeze(is_originally_frozen)

        return (i_nw, j_nw), (rf_w, rf_h)

    @classmethod
    def preprocess(cls, wav_file):
        wav_data, sr = lb.load(wav_file, mono=True)
        min_length = sr * 1
        max_length = sr * 20
        if wav_data.shape[0] < min_length:
            wav_data = np.concatenate((wav_data, np.zeros(min_length - wav_data.shape[0])))
        wav_data = wav_data[:max_length]
        wav_data = wav_data.reshape(1, -1, 1)
        wav_data = (((256 - -256)*((wav_data - wav_data.min()) / (wav_data.max() - wav_data.min()))) - 256)
        return wav_data, None


if __name__ == '__main__':
    net = Soundnet_wrapper()
    print(net._config)
