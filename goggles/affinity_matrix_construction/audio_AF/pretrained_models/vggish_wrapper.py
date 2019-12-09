from collections import OrderedDict

import numpy as np
import librosa as lb

import torch
import torch.nn as nn
from torchvision import models

import goggles.torch_vggish.vggish as vggish
import goggles.torch_vggish.audioset.vggish_input as vggish_input

class VGGish_wrapper(nn.Module):
    def __init__(self, freeze=True):
        super(VGGish_wrapper, self).__init__()
        self.name = 'VGGish'
        self._is_cuda = False

        # self.input_size = 224
        self.input_frame_size = 96 # Spectrogram Temporal Frames
        self.input_freq_size = 64  # Spectrogram Frequency Bands

        # base_model = models.vgg16(pretrained=True)
        base_model = vggish.get_model(pretrained=True)

        features = list(base_model.features)
        self._features = nn.ModuleList(features).eval()

        classifier = list(base_model.fc)
        embedding = classifier
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
            torch.rand(1, 1, self.input_freq_size, self.input_frame_size),
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
        return super(VGGish_wrapper, self).cuda(device_id)

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

    def get_svm_data(self, x):
        x = x[0]
        x = torch.from_numpy(x).unsqueeze(dim=0)
        x = x.view((1,) + x.size()).type('torch.FloatTensor')
        x = torch.autograd.Variable(x, requires_grad=False)
        x = self.forward(x)
        x = x.view(x.size(0), -1)
        x = self._embedding(x)
        return x.numpy()

    def get_layer_type(self, layer_idx):
        return self._config[layer_idx][0]

    def get_layer_output_dim(self, layer_idx):
        return self._config[layer_idx][1]

    def get_receptive_field(self, patch, layer_idx=None):
        is_originally_frozen = self._is_frozen
        self.zero_grad()
        self.freeze(False)
        batch_shape = (1, 1, self.input_freq_size, self.input_frame_size)

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
        return vggish_input.wavfile_to_examples(wav_file)


if __name__ == '__main__':
    net = VGGish_wrapper()
    print(net._config)
