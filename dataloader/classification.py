import site
import cv2, os
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
try:
    from vujade import vujade_str as str_
    from vujade import vujade_csv as csv_
    from vujade.vujade_transformation import image_restoration as trans_
    from vujade.vujade_debug import printf
except Exception as e:
    site.addsitedir(sitedir=os.getcwd())
    from vujade import vujade_str as str_
    from vujade import vujade_csv as csv_
    from vujade.vujade_transformation import image_restoration as trans_
    from vujade.vujade_debug import printf


# DataLoader
class ClassificationDataLoader(object):
    def __init__(self, spath_csv, spath_dataset, is_train, size=(32, 32), padding=4, prob_flip=0.5,
                 mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010),
                 shuffle=True, batch_size=1, n_workers=1, pin_memory=True):
        super(ClassificationDataLoader, self).__init__()
        self.spath_csv = spath_csv
        self.spath_dataset = spath_dataset
        self.is_train = is_train
        self.size = size if len(size) == 2 else [size[-2], size[-1]]
        self.padding = padding
        self.prob_flip = prob_flip
        self.mean = mean
        self.std = std
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.pin_memory = pin_memory

        self.dataset = ClassificationDataset(
            _spath_csv=self.spath_csv,
            _spath_dataset=self.spath_dataset,
            _is_train=self.is_train,
            _size=self.size,
            _padding=self.padding,
            _prob_flip=self.prob_flip,
            _mean=self.mean,
            _std=self.std
        )

    @property
    def loader(self):
        return DataLoader(
            dataset=self.dataset,
            num_workers=self.n_workers,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            pin_memory=self.pin_memory
        )


# Dataset
class ClassificationDataset(Dataset):
    def __init__(self, _spath_csv, _spath_dataset, _is_train, _size=(32, 32), _padding=4, _prob_flip=0.5,
                 _mean=(0.4914, 0.4822, 0.4465), _std=(0.2023, 0.1994, 0.2010)):
        super(ClassificationDataset, self).__init__()
        self.spath_csv = _spath_csv
        self.spath_dataset = _spath_dataset
        self.is_train = _is_train
        self.size = _size if len(_size) == 2 else [_size[-2], _size[-1]]
        self.padding = _padding
        self.prob_flip = _prob_flip
        self.mean = _mean
        self.std = _std

        self.dataset = csv_.CSV(_spath_filename=self.spath_csv).read()
        self.filenames = self.dataset['filenames']
        self.labels = self.dataset['labels']

        if self.is_train:
            self.trans = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.RandomCrop(size=self.size, padding=self.padding),
                 transforms.RandomHorizontalFlip(p=self.prob_flip),
                 transforms.Normalize(mean=self.mean, std=self.std)]
            )
        else:
            self.trans = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize(mean=self.mean, std=self.std)]
            )

    def __getitem__(self, idx):
        name_img, label = self.filenames[idx], self.labels[idx]
        path_img = os.path.join(self.spath_dataset, name_img)

        img = cv2.imread(filename=path_img).astype(np.float32) / 255.0
        img = self.trans(img)

        return img, label

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    # Training set
    dataset_train = ClassificationDataset(
        _spath_csv=os.path.join('/DATA/Dataset/CIFAR_10/CIFAR_10_train.csv'),
        _spath_dataset=os.path.join('/DATA/Dataset/CIFAR_10/CIFAR_10_train'),
        _is_train=True
    )
    data_train = dataset_train.__getitem__(0)
    print(type(data_train[0]), data_train[0].shape, data_train[0].dtype, data_train[0].min(), data_train[0].max())
    print(type(data_train[1]), data_train[1].shape, data_train[1].dtype)
    # print(data_train[0], data_train[1])

    # Testing set
    dataset_test = ClassificationDataset(
        _spath_csv=os.path.join('/DATA/Dataset/CIFAR_10/CIFAR_10_test.csv'),
        _spath_dataset=os.path.join('/DATA/Dataset/CIFAR_10/CIFAR_10_test'),
        _is_train=False
    )
    data_test = dataset_test.__getitem__(0)
    print(type(data_test[0]), data_test[0].shape, data_test[0].dtype, data_test[0].min(), data_test[0].max())
    print(type(data_test[1]), data_test[1].shape, data_test[1].dtype)
    # print(data_test[0], data_test[1])
