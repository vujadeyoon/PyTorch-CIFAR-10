# PyTorch Image Classification: CIFAR-10


## Table of contents
1.  [Notice](#notice)
2.  [Summarized development environment](#env)
3.  [How to set up the project](#setup)
4.  [How to get the CIFAR-10 dataset](#dataset)
5.  [How to train the CNN model for the CIFAR-10 dataset](#train)
6.  [How to test the CNN model for the CIFAR-10 dataset](#test)
7.  [Result](#result)
8.  [Todo](#todo)
9.  [License](#license)
10. [Acknowledgement](#ack)
11. [Reference](#ref)


## 1. Notice <a name="notice"></a>
- PyTorch based image classification for CIFAR-10.
- The CNN model is the ResNet-18.
- This repository is inspired by PyTorch Template Project [1] and Train CIFAR10 with PyTorch [2].
- However, the repository is detached from the PyTorch Template Project in order to concentrate on researching and
  developing the advanced features rapidly without concerning backward compatibility.
  In particular, you can deal with your own dataset easily using the dataloader/ unlike the PyTorch Template Project.
- I recommend that you should ignore the commented instructions with an octothorpe, #.
- Modified date: Mar. 31, 2021.


## 2. Summarized development environment <a name="env"></a>
- Operating System (OS): Ubuntu MATE 18.04.3 LTS (Bionic)
- Graphics Processing Unit (GPU): NVIDIA TITAN Xp, 1ea
- GPU driver: Nvidia-450.102.04
- CUDA toolkit: CUDA 10.2
- cuDNN: cuDNN v7.6.5
- Python3: Python 3.7.9
- PyTorch: 1.8.1
- Torchvision: 0.8.1
- opencv-python==4.5.1.48
- <a href="https://github.com/vujadeyoon/vujade"> vujade</a>==0.3.1 <em># The pip3 is not yet supported.</em>


## 3. How to set up the project <a name="setup"></a>
- You can run the below bash command to install the required python3 packages.
  ```bash
  pip3 install -r ./requirements.txt
  ```
- The repository includes and utilizes the vujade [3] that is a collection of useful
  classes and functions based on the Python3 for deep learning research and development.
- Some of the features in the vujade should be compiled by the Cython for each development envrionment (e.g. Python version).
- The vujade in this repository already has pre-compiled .so files by the Cython for Python 3.7. Nevertheless,
  if it is not compatible with your development environment, please run a bash script, <em>bash_setup.sh</em> as follows:
  ```bash
  bash ./bash/bash_setup.sh
  ```


## 4. How to get the CIFAR-10 dataset <a name="dataset"></a>
- You can run the below bash script, <em>bash_data.sh</em> to download the CIFAR-10 dataset form the official site to the path, <em>/DATA/Dataset/CIFAR_10</em>.
  Please note that the bash script also makes both csv files to be used in the training and testing phases in the path.
  ```bash
  bash ./bash/bash_data.sh
  ```
- The torchvision provides the API to download and utilize the CIFAR-10 dataset such as <em>torchvision.datasets.CIFAR10</em>.
  However, I recommend you download the CIFAR-10 dataset form the official site directly using the given bash script
  because you need to become familiar with the dataloder/ in order to deal with your own dataset.


## 5. How to train the CNN model for the CIFAR-10 dataset <a name="train"></a>
- You can train the CNN model by running the bash script <em>bash_train.sh</em> as follows:
  ```bash
  bash ./bash/bash_train.sh
  ```


## 6. How to test the CNN model for the CIFAR-10 dataset <a name="test"></a>
- You can test the CNN model by running the bash script <em>bash_test.sh</em> as follows:
  ```bash
  bash ./bash/bash_test.sh ./saved/model/CIFAR_10/best/config.json ./saved/model/CIFAR_10/best/ckpt-best.pth
  ```


## 7. Result <a name="result"></a>
|CNN model|Cross entropy loss|Top-1 accuracy|Top-3 accuracy|
|:-------:|:----:|:-----:|:-----:|
|ResNet-18|0.1703|95.35 %|99.42 %|


## 8. Todo <a name="todo"></a>
- [ ] Support the <a href="https://github.com/wandb/client"> wandb</a>
- [ ] Support Learning rate finder
- [ ] Improve logger
- [ ] TensorRT


## 9. License <a name="license"></a>
- I respect and follow the license of the used libraries including python3 packages and the dataset.
- They are licensed by their own licenses.
- Please note that the only providen vujadeyoon's own codes, wrapper-codes comply with the MIT license.


## 10. Acknowledgement <a name="ack"></a>
- This project is inspired by both repositories: i) PyTorch Template Project [1] and ii) Train CIFAR10 with PyTorch [2].


## 11. Reference <a name="ref"></a>
1. <a href="https://github.com/victoresque/pytorch-template"> PyTorch Template Project</a>
2. <a href="https://github.com/kuangliu/pytorch-cifar"> Train CIFAR10 with PyTorch</a>
3. <a href="https://github.com/vujadeyoon/vujade"> vujade</a>
