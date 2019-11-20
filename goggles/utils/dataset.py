import os
import numpy as np
import pandas as pd
import librosa as lb

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import goggles.torch_vggish.audioset.vggish_input as vggish_input


class GogglesDataset(Dataset):
    def __init__(self,path,transform):
        valid_images = [".jpg", ".gif", ".png"]
        self._data_path = path
        self.images_filename_list = []
        for f in os.listdir(path):
            ext = os.path.splitext(f)[1]
            if ext.lower() not in valid_images:
                continue
            self.images_filename_list.append(f)
        self.images_filename_list = list(sorted(self.images_filename_list))
        if transform is not None:
            self._transform = transform
        else:
            self._transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        """
        read a image only when it is used
        :param idx: integer
        :return:
        """
        filename = self.images_filename_list[idx]
        try:
            image_file = os.path.join(self._data_path, filename)
            image = Image.open(image_file).convert('RGB')
            image = self._transform(image)
        except:
            image = None
        return image

    def __len__(self):
        return len(self.images_filename_list)

    @classmethod
    def load_all_data(cls, root_dir, input_image_size=224):
        try:
            transform_resize = transforms.Resize(
                (input_image_size, input_image_size))
        except AttributeError:
            transform_resize = transforms.Scale(
                (input_image_size, input_image_size))

        transform_to_tensor = transforms.ToTensor()
        transform_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225])
        transformation = transforms.Compose([
            transform_resize, transform_to_tensor, transform_normalize])
        dataset = cls(
            root_dir,
            transform=transformation)
        return dataset


class AudioDataset(Dataset):
    def __init__(self, path, transform, meta_df=None):
        valid_audio = ['.wav']
        self._data_path = path
        self.audio_filename_list = []
        for f in os.listdir(path):
            ext = os.path.splitext(f)[1]
            if ext.lower() not in valid_audio:
                continue
            if meta_df is not None and os.path.basename(f) not in meta_df['filename'].values:
                continue
            self.audio_filename_list.append(f)
        self.audio_filename_list = list(sorted(self.audio_filename_list))
        # if transform is not None:
        #     self._transform = transform
        # else:
        #     self._transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        """
        read a image only when it is used
        :param idx: integer
        :return:
        """
        filename = self.audio_filename_list[idx]
        try:
            wav_file = os.path.join(self._data_path, filename)
            wav_examples = vggish_input.wavfile_to_examples(wav_file, trim_audio=True)
            wav_spectrogram = wav_examples
        except:
            wav_spectrogram = None
        return wav_spectrogram

    def __len__(self):
        return len(self.audio_filename_list)

    @classmethod
    def load_all_data(cls, root_dir, meta_df=None, input_height=96, input_width=64):
        
        # transform_resize = transforms.Resize((input_height, input_width))

        transform_to_tensor = transforms.ToTensor()
        # transform_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                            std=[0.229, 0.224, 0.225])
        # transformation = transforms.Compose([
        #     transform_resize, transform_to_tensor])
        transformation = transforms.Compose([transform_to_tensor])
        dataset = cls(
            root_dir,
            transform=transformation,
            meta_df=meta_df)
        return dataset

class RawAudioDataset(Dataset):
    '''
    Not Implemented yet.

    TODO: Implement this dataset for loading data in soundnet's preprocessed format.

    Note: We could probably just combine this with the above class AudioDataset,
          keeping them separate for now to avoid conflicts while developing.
    '''

    def __init__(self, path, transform, meta_df=None):
        valid_audio = ['.wav']
        self._data_path = path
        self.audio_filename_list = []
        for f in os.listdir(path):
            ext = os.path.splitext(f)[1]
            if ext.lower() not in valid_audio:
                continue
            if meta_df is not None and os.path.basename(f) not in meta_df['filename'].values:
                continue
            self.audio_filename_list.append(f)
        self.audio_filename_list = list(sorted(self.audio_filename_list))
        # if transform is not None:
        #     self._transform = transform
        # else:
        #     self._transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        """
        read a image only when it is used
        :param idx: integer
        :return:
        """
        filename = self.audio_filename_list[idx]
        try:
            length = 22050 * 20
            wav_file = os.path.join(self._data_path, filename)
            wav_data, sr = lb.load(wav_file)
            # import pdb; pdb.set_trace()
            wav_data = np.concatenate((wav_data, np.zeros(length - wav_data.shape[0])))
            # wav_data = np.tile(wav_data, int(length / wav_data.shape[0] + 1))
            wav_data = wav_data[:length]
            wav_data = wav_data.reshape(1, -1, 1)
            # wav_examples = vggish_input.wavfile_to_examples(wav_file)
            # wav_spectrogram = wav_examples
        except:
            wav_data = None
        return wav_data, None

    def __len__(self):
        return len(self.audio_filename_list)

    @classmethod
    def load_all_data(cls, root_dir, meta_df=None, input_height=96, input_width=64):
        
        # transform_resize = transforms.Resize((input_height, input_width))

        transform_to_tensor = transforms.ToTensor()
        # transform_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                            std=[0.229, 0.224, 0.225])
        # transformation = transforms.Compose([
        #     transform_resize, transform_to_tensor])
        transformation = transforms.Compose([transform_to_tensor])
        dataset = cls(
            root_dir,
            transform=transformation,
            meta_df=meta_df)
        return dataset

# class AudioDatasetParallel(Dataset):
#     def __init__(self, path, transform, meta_df=None):
#         valid_audio = ['.wav']
#         self._data_path = path
#         self.audio_filename_list = []
#         for f in os.listdir(path):
#             ext = os.path.splitext(f)[1]
#             if ext.lower() not in valid_audio:
#                 continue
#             if meta_df is not None and os.path.basename(f) not in meta_df['filename'].values:
#                 continue
#             self.audio_filename_list.append(f)
#         self.audio_filename_list = list(sorted(self.audio_filename_list))
#         # if transform is not None:
#         #     self._transform = transform
#         # else:
#         #     self._transform = transforms.Compose([transforms.ToTensor()])

#     def __getitem__(self, idx):
#         """
#         read a image only when it is used
#         :param idx: integer
#         :return:
#         """
#         filename = self.audio_filename_list[idx]
#         try:
#             wav_file = os.path.join(self._data_path, filename)
#             wav_examples = vggish_input.wavfile_to_examples(wav_file)
#             wav_spectrogram = wav_examples
#         except:
#             wav_spectrogram = None
#         return wav_spectrogram

#     def __len__(self):
#         return len(self.audio_filename_list)

#     @classmethod
#     def load_all_data(cls, root_dir, meta_df=None, input_height=96, input_width=64):
        
#         # transform_resize = transforms.Resize((input_height, input_width))

#         transform_to_tensor = transforms.ToTensor()
#         # transform_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#         #                                            std=[0.229, 0.224, 0.225])
#         # transformation = transforms.Compose([
#         #     transform_resize, transform_to_tensor])
#         transformation = transforms.Compose([transform_to_tensor])
#         dataset = cls(
#             root_dir,
#             transform=transformation,
#             meta_df=meta_df)
#         return dataset