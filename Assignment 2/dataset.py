import torch
import h5py


def read_h5(path):
    data = h5py.File(path, 'r')
    image = data['image'][:]
    label = data['label'][:]
    return image, label


class LAHeart(torch.utils.data.Dataset):
    def __init__(self, split='train', transform=None):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError
