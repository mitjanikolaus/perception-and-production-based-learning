import os
import pickle
import random

import nltk
import pandas as pd
import h5py as h5py
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms


from preprocess import (
    MEAN_ABSTRACT_SCENES,
    STD_ABSTRACT_SCENES,
    encode_caption,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CaptionDataset(Dataset):
    """
    PyTorch Dataset that provides batches of images of a given split
    """

    CAPTIONS_PER_IMAGE = 6

    def __init__(
        self,
        data_folder,
        features_filename,
        captions_filename,
        vocab,
        dataset_size=1.0,
        features_scale_factor=1 / 255.0,
    ):
        """
        :param data_folder: folder where data files are stored
        :param features_filename: Filename of the image features file
        :param normalize: PyTorch normalization transformation
        :param dataset_size: Fraction of dataset to use
        :param features_scale_factor: Additional scale factor, applied before normalization
        """
        self.images = h5py.File(os.path.join(data_folder, features_filename), "r")

        self.features_scale_factor = features_scale_factor

        # Load captions
        with open(os.path.join(data_folder, captions_filename), "rb") as file:
            self.captions = pickle.load(file)

        # Set pytorch transformation pipeline
        self.normalize = transforms.Normalize(
            mean=MEAN_ABSTRACT_SCENES, std=STD_ABSTRACT_SCENES
        )

        self.image_ids = [int(i) for i in list(self.images.keys())]

        if dataset_size < 1:
            self.image_ids = random.sample(self.image_ids, round(len(self.image_ids) * dataset_size))

        self.vocab = vocab

    def get_image_features(self, id, channels_first=True, normalize=True):
        image_data = self.images[str(id)][()]

        image = torch.FloatTensor(image_data)

        if channels_first:
            image = image.permute(2, 0, 1)

        if normalize:
            image = self.normalize(image)

        # scale the features with given factor
        image = image * self.features_scale_factor

        return image

    def __getitem__(self, i):
        image_id = self.image_ids[i // self.CAPTIONS_PER_IMAGE]
        caption_id = i % self.CAPTIONS_PER_IMAGE

        image = self.get_image_features(image_id)

        caption = self.captions[image_id][caption_id]

        caption = torch.LongTensor(caption)

        return image, caption, image_id

    def __len__(self):
        return len(self.image_ids) * self.CAPTIONS_PER_IMAGE

    def pad_collate(batch):
        images = torch.stack([s[0] for s in batch])
        captions = [s[1] for s in batch]
        image_ids = torch.tensor([s[2] for s in batch])

        sequence_lengths = torch.tensor([len(c) for c in captions])
        padded_captions = pad_sequence(captions, batch_first=True)

        return (
            images.to(device),
            padded_captions.to(device),
            sequence_lengths.to(device),
            image_ids,
        )


class CaptionRLDataset(Dataset):
    """
    PyTorch Dataset that provides batches of images along with all gold captions of a given split
    """

    CAPTIONS_PER_IMAGE = 6

    def __init__(
        self,
        data_folder,
        features_filename,
        captions_filename,
        vocab,
        dataset_size=1.0,
        features_scale_factor=1 / 255.0,
    ):
        """
        :param data_folder: folder where data files are stored
        :param features_filename: Filename of the image features file
        :param normalize: PyTorch normalization transformation
        :param features_scale_factor: Additional scale factor, applied before normalization
        """
        self.images = h5py.File(os.path.join(data_folder, features_filename), "r")

        self.features_scale_factor = features_scale_factor

        # Load captions
        with open(os.path.join(data_folder, captions_filename), "rb") as file:
            self.captions = pickle.load(file)

        # Set pytorch transformation pipeline
        self.normalize = transforms.Normalize(
            mean=MEAN_ABSTRACT_SCENES, std=STD_ABSTRACT_SCENES
        )

        self.image_ids = [int(i) for i in list(self.images.keys())]

        if dataset_size < 1:
            self.image_ids = random.sample(self.image_ids, round(len(self.image_ids) * dataset_size))

        self.vocab = vocab

    def get_image_features(self, id, channels_first=True, normalize=True):
        image_data = self.images[str(id)][()]

        image = torch.FloatTensor(image_data)

        if channels_first:
            image = image.permute(2, 0, 1)

        if normalize:
            image = self.normalize(image)

        # scale the features with given factor
        image = image * self.features_scale_factor

        return image

    def __getitem__(self, i):
        image_id = self.image_ids[i]

        image = self.get_image_features(image_id)

        captions = self.captions[image_id]

        captions = [torch.tensor(caption, device=device, dtype=torch.long) for caption in captions]

        return image, captions, image_id

    def __len__(self):
        return len(self.image_ids)

    def pad_collate(batch):
        images = torch.stack([s[0] for s in batch])
        captions = [s[1] for s in batch]
        image_ids = torch.tensor([s[2] for s in batch], device=device)

        # flatten captions in order to pad
        flattened_captions = []
        for captions_image in captions:
            flattened_captions.extend(captions_image)
        sequence_lengths = torch.tensor([len(c) for c in flattened_captions], device=device)
        padded_captions = pad_sequence(flattened_captions, batch_first=True)

        # separate back into captions per image
        padded_captions = padded_captions.reshape(images.shape[0], CaptionRLDataset.CAPTIONS_PER_IMAGE, -1)
        sequence_lengths = sequence_lengths.reshape(images.shape[0], -1)

        return (
            images.to(device),
            padded_captions.to(device),
            sequence_lengths.to(device),
            image_ids,
        )


class SemanticsEvalDataset(Dataset):
    """
    PyTorch Dataset that provides sets of target and distractor images for syntax learning evaluation
    """

    def __init__(
        self,
        data_folder,
        features_filename,
        captions_filename,
        eval_csv,
        vocab,
        features_scale_factor=1 / 255.0,
    ):
        """
        :param data_folder: folder where data files are stored
        :param features_filename: Filename of the image features file
        :param data_indices: dataset split, indices of images that should be included
        :param features_scale_factor: Additional scale factor, applied before normalization
        """
        self.images = h5py.File(os.path.join(data_folder, features_filename), "r")

        self.features_scale_factor = features_scale_factor

        # Set pytorch transformation pipeline
        self.normalize = transforms.Normalize(
            mean=MEAN_ABSTRACT_SCENES, std=STD_ABSTRACT_SCENES
        )

        self.vocab = vocab

        self.data = pd.read_csv(eval_csv)

    def get_image_features(self, id, channels_first=True, normalize=True):
        image_data = self.images[str(id)][()]

        # show_image(image_data)

        image = torch.tensor(image_data, device=device, dtype=torch.float)

        if channels_first:
            image = image.permute(2, 0, 1)

        if normalize:
            image = self.normalize(image)

        # scale the features with given factor (convert values from [0, 256] to [0, 1]
        image = image * self.features_scale_factor

        return image

    def __getitem__(self, i):
        img_id, target_sentence, distractor_sentence = self.data.iloc[i]
        img = self.get_image_features(img_id)

        target_sentence = nltk.word_tokenize(target_sentence)
        target_sentence = encode_caption(target_sentence, self.vocab)
        target_sentence = torch.tensor(target_sentence, device=device)

        distractor_sentence = nltk.word_tokenize(distractor_sentence)
        distractor_sentence = encode_caption(distractor_sentence, self.vocab)
        distractor_sentence = torch.tensor(distractor_sentence, device=device)

        return img, target_sentence, distractor_sentence

    def __len__(self):
        length = len(self.data)

        return length


class VisualRefGameDataset(Dataset):
    """
    PyTorch Dataset that provides sets of target and distractor images and captions
    """

    def __init__(
        self,
        data_folder,
        features_filename,
        captions_filename,
        batch_size,
        features_scale_factor=1 / 255.0,
        max_samples=None,
    ):
        """
        :param data_folder: folder where data files are stored
        :param features_filename: Filename of the image features file
        :param data_indices: dataset split, indices of images that should be included
        :param features_scale_factor: Additional scale factor, applied before normalization
        """
        self.images = h5py.File(os.path.join(data_folder, features_filename), "r")

        self.features_scale_factor = features_scale_factor

        # Set pytorch transformation pipeline
        self.normalize = transforms.Normalize(
            mean=MEAN_ABSTRACT_SCENES, std=STD_ABSTRACT_SCENES
        )

        # Load captions
        with open(os.path.join(data_folder, captions_filename), "rb") as file:
            self.captions = pickle.load(file)

        all_image_ids = list(self.images.keys())
        self.sample_image_ids = []
        for i in all_image_ids:
            for j in all_image_ids:
                self.sample_image_ids.append((i, j))

        if max_samples:
            self.sample_image_ids = random.sample(self.sample_image_ids, max_samples)

        self.batch_size = batch_size

    def get_image_features(self, id, channels_first=True, normalize=True):
        image_data = self.images[str(id)][()]

        image = torch.FloatTensor(image_data)

        if channels_first:
            image = image.permute(2, 0, 1)

        if normalize:
            image = self.normalize(image)

        # scale the features with given factor (convert values from [0, 256] to [0, 1]
        image = image * self.features_scale_factor

        return image

    def __getitem__(self, i):
        target_image_id, distractor_image_id = self.sample_image_ids[i]

        target_image = self.get_image_features(target_image_id)
        distractor_image = self.get_image_features(distractor_image_id)

        # The receiver gets target and distractor in random order
        target_position = np.random.choice(2)
        if target_position == 0:
            images = target_image, distractor_image
        else:
            images = distractor_image, target_image
        target_label = target_position

        target_image_id = int(target_image_id)
        distractor_image_id = int(distractor_image_id)

        caption_id = random.choice(range(6))
        caption = self.captions[target_image_id][caption_id]

        caption = torch.tensor(caption, device=device)

        return images, target_label, target_image_id, distractor_image_id, caption

    def __len__(self):
        length = len(self.sample_image_ids)

        # discard last incomplete batch
        return length - (length % self.batch_size)


def pad_collate_visual_ref(batch):
    images = torch.stack(
        (torch.stack([s[0][0] for s in batch]), torch.stack([s[0][1] for s in batch]))
    )
    target_labels = torch.tensor([s[1] for s in batch], device=device)
    target_image_ids = torch.tensor([s[2] for s in batch], device=device)
    distractor_image_ids = torch.tensor([s[3] for s in batch], device=device)
    captions = [s[4] for s in batch]

    sequence_lengths = torch.tensor([len(c) for c in captions], device=device)
    padded_captions = pad_sequence(captions, batch_first=True)

    sender_inputs = (
        images,
        target_labels,
        target_image_ids,
        distractor_image_ids,
        padded_captions,
        sequence_lengths,
    )
    receiver_inputs = images

    return sender_inputs, target_labels, receiver_inputs
