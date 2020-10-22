# We use the torchmeta package to generate synthetic regression tasks
# https://github.com/tristandeleu/pytorch-meta/tree/master/torchmeta

import torch
from torchmeta.toy import Sinusoid
from torchmeta.transforms import ClassSplitter
from torchmeta.utils.data import BatchMetaDataLoader


class ToTensor1D(object):
    """Convert a `numpy.ndarray` to tensor. Unlike `ToTensor` from torchvision,
    this converts numpy arrays regardless of the number of dimensions.

    Converts automatically the array to `float32`.
    """

    def __call__(self, array):
        return torch.from_numpy(array.astype("float32"))

    def __repr__(self):
        return self.__class__.__name__ + "()"


def get_sine_loader(batch_size, num_steps, shots=10, test_shots=15):
    dataset_transform = ClassSplitter(
        shuffle=True, num_train_per_class=shots, num_test_per_class=test_shots
    )
    transform = ToTensor1D()
    dataset = Sinusoid(
        shots + test_shots,
        num_tasks=batch_size * num_steps,
        transform=transform,
        target_transform=transform,
        dataset_transform=dataset_transform,
    )
    loader = BatchMetaDataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
    )
    return loader
