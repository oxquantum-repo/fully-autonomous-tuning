import logging
import numpy as np

import torchvision

import torch
from torch import nn, optim
import torch.nn.functional as F


class HighResDDClassifier:
    def __init__(self, path_to_nn):
        self.device = "cpu"
        net = torchvision.models.resnet18(pretrained=False)
        net.conv1 = nn.Conv2d(
            1,
            net.conv1.out_channels,
            net.conv1.kernel_size,
            net.conv1.stride,
            net.conv1.padding,
            bias=net.conv1.bias,
        )

        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, 2)
        net = net.to(self.device)
        net.eval()
        net.load_state_dict(
            torch.load(path_to_nn, map_location=torch.device(self.device))
        )#"cpu"))
        self.net = net

    def predict(self, sample):
        """Feed trough classifier.

        We expect input to be of shape (48, 48).
        """

        sample = self.normalise_arrays(sample[np.newaxis, :, :])
        sample = sample[np.newaxis, :, :]

        outputs = self.net(torch.FloatTensor(sample).to(self.device))
        predicted = torch.max(outputs.data, 1).indices.detach().cpu()

        return predicted

    def normalise_arrays(self, data):

        # Check data is a numpy array
        assert type(data) == np.ndarray
        # Check data is an array of arrays
        assert data.ndim == 3
        logging.debug(f"Data shape: {data.shape}")

        # Flatten each array within data
        all_arrays_flat = data.reshape([data.shape[0], -1])
        # Get the mins and maxs and calculate range
        maximums = np.atleast_2d(np.amax(all_arrays_flat, axis=1)).T
        minimums = np.atleast_2d(np.amin(all_arrays_flat, axis=1)).T
        data_range = maximums - minimums

        # Normalise
        scaled = (all_arrays_flat - minimums) / data_range

        # Reshape to original data shape
        normalised_data = scaled.reshape(data.shape)
        return normalised_data
