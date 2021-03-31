import argparse
import os
import pickle
from tqdm import tqdm
import numpy as np
from vujade import vujade_utils as utils_
from vujade import vujade_imgcv as imgcv_
from vujade import vujade_csv as csv_


def get_args():
    parser = argparse.ArgumentParser(description='Dataset for cifar')
    parser.add_argument('--name_dataset', type=str, default='CIFAR_10', help='Dataset: CIFAR_10')
    parser.add_argument('--path_dataset', type=str, required=True, help='Path for the dataset')
    args = parser.parse_args()

    return args


def get_dataset_cifar(_list_name_batch_phase: list, _path_base: str, _phase: str, _name_dataset: str):
    path_phase = os.path.join(_path_base, '{}_{}'.format(_name_dataset, _phase))
    path_csv = os.path.join(_path_base, '{}_{}.csv'.format(_name_dataset, _phase))
    utils_.makedirs(_path=path_phase, _exist_ok=True)
    csv_cifar = csv_.vujade_csv(_path_filename=path_csv, _header=['filenames', 'labels'])

    for idx, _name_batch in enumerate(_list_name_batch_phase):
        print('Progress [{}]/[{}]: {}'.format(idx, len(_list_name_batch_phase) -1 , _name_batch))
        path_batch = os.path.join(_path_base, 'temp', _name_batch)

        dict_data = unpickle(_file=path_batch)

        batch_label = dict_data[b'batch_label']
        labels = dict_data[b'labels']
        data = dict_data[b'data']
        filenames = dict_data[b'filenames']

        for idy in tqdm(range(data.shape[0])):
            img_vec = data[idy, :]
            img = cifar_vec2img(_img_vec=img_vec)
            img_label = labels[idy]
            img_name = (filenames[idy]).decode('utf-8')
            path_img = os.path.join(path_phase, img_name)

            imgcv_.imwrite(_filename=path_img, _ndarr=img)
            csv_data = np.array([img_name, img_label]).reshape(1, -1)
            csv_cifar.write(_ndarr=csv_data)


def cifar_vec2img(_img_vec: np.ndarray, _img_shape: tuple = (32, 32)) -> np.ndarray:
    img_shape= _img_shape
    img_height = img_shape[0]
    img_width = img_shape[1]
    img_len = img_height * img_width
    img_channel = 3

    img_r = _img_vec[0:img_len].reshape(img_shape)
    img_g = _img_vec[img_len:2*img_len].reshape(img_shape)
    img_b = _img_vec[2*img_len:3*img_len].reshape(img_shape)

    img = np.zeros(shape=(img_height, img_width, img_channel), dtype=np.uint8)
    img[:, :, 0] = img_b
    img[:, :, 1] = img_g
    img[:, :, 2] = img_r

    return img


def unpickle(_file: str) -> dict:
    with open(_file, 'rb') as f:
        dict_data = pickle.load(f, encoding='bytes')

    return dict_data


if __name__ == '__main__':
    args = get_args()

    name_dataset = args.name_dataset
    path_base = args.path_dataset

    if name_dataset == 'CIFAR_10':
        list_name_batch_train = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        list_name_batch_test = ['test_batch']
    else:
        raise NotImplementedError


    get_dataset_cifar(_list_name_batch_phase=list_name_batch_train, _path_base=path_base, _phase='train', _name_dataset=name_dataset)
    get_dataset_cifar(_list_name_batch_phase=list_name_batch_test, _path_base=path_base, _phase='test', _name_dataset=name_dataset)




