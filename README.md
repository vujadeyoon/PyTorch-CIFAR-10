# PyTorch Image Classification: CIFAR-10


## Table of contents
1. [Notice](#notice)
2. [Summarized development environment](#env)
3. [How to set up the project](#setup)
4. [How to get the CIFAR-10 dataset](#dataset)
5. [How to analyze the CNN model in development](#anlyze)
6. [How to train the CNN model for the CIFAR-10 dataset](#train)
7. [How to test the CNN model for the CIFAR-10 dataset](#test)
8. [How to export the trained CNN model](#export)
9. [How to inference the trained CNN model](#inference)
10. [Result](#result)
11. [Useful commands](#useful_commands)
12. [Todo](#todo)
13. [License](#license)
14. [Acknowledgement](#ack)
15. [Reference](#ref)


## 1. Notice <a name="notice"></a>
- PyTorch based image classification for CIFAR-10.
- The CNN model is the ResNet-18.
- This repository is inspired by PyTorch Template Project [1] and Train CIFAR10 with PyTorch [2].
- However, the repository is detached from the PyTorch Template Project in order to concentrate on researching and
  developing the advanced features rapidly without concerning backward compatibility.
  In particular, you can deal with your own dataset easily using the dataloader/ unlike the PyTorch Template Project.
- I recommend that you should ignore the commented instructions with an octothorpe, #.


## 2. Summarized development environment <a name="env"></a>
- Operating System (OS): Ubuntu MATE 18.04.3 LTS (Bionic)
- Graphics Processing Unit (GPU): NVIDIA TITAN Xp, 1ea
- GPU driver: Nvidia-525.125.06
- CUDA toolkit: CUDA 11.7
- cuDNN: cuDNN v8.8.0
- Python3: Python 3.8.10
- PyTorch: 1.13.0
- Torchvision: 0.14.0
- opencv-python==4.7.0.72
- <a href="https://github.com/vujadeyoon/vujade"> vujade</a>==0.6.8.3 <em># The pip3 is not yet supported.</em>


## 3. How to set up the project <a name="setup"></a>
- You can run the below bash command to install the required python3 packages.
  ```bash
  $ pip3 install -r ./requirements.txt
  ```
- The repository includes and utilizes the vujade [3] that is a collection of useful
  classes and functions based on the Python3 for deep learning research and development.
- Some of the features in the vujade should be compiled by the Cython for each development envrionment (e.g. Python version).
- The vujade in this repository already has pre-compiled .SO files by the Cython. Nevertheless,
  if it is not compatible with your development environment, please refer to the [vujade-python](https://github.com/vujadeyoon/vujade-python).


## 4. How to get the CIFAR-10 dataset <a name="dataset"></a>
- You can run the below bash script, <em>bash_data.sh</em> to download the CIFAR-10 dataset form the official site to the path, <em>/DATA/Dataset/CIFAR_10</em>.
  Please note that the bash script also makes both csv files to be used in the training and testing phases in the path.
  ```bash
  $ bash ./script/cifar_10/bash_data.sh
  ```
- The torchvision provides the API to download and utilize the CIFAR-10 dataset such as <em>torchvision.datasets.CIFAR10</em>.
  However, I recommend you download the CIFAR-10 dataset form the official site directly using the given bash script
  because you need to become familiar with the dataloder/ in order to deal with your own dataset.


## 5. How to analyze the CNN model in development <a name="anlyze"></a>
### 1. How to analyze parameters, macs and flops for the CNN model
```bash
$ python3 main_analysis.py --mode dev
```
### 2. How to analyze parameters, macs and flops with time complexity for the CNN model
```bash
$ python3 main_analysis.py --mode analysis
```
### 3. How to analyze parameters, macs and flops for the CNN model 
```bash
$ python3 main_analysis.py --mode summary
```


## 6. How to train the CNN model for the CIFAR-10 dataset <a name="train"></a>
### 1. How to use TensorBoard
- TensorBoard - offline
  ```bash
  $ tensorboard --logdir ./saved/
  ```
- TensorBoard - online
  ```bash
  $ tensorboard dev upload --logdir ./saved/ --name <NAME> --description <DESCRIPTION>
  $ tensorboard dev list
  $ tensorboard dev delete --experiment_id <EXPERIMENT_ID>
  ```
### 2. How to train
```bash
$ bash ./script/cifar_10/bash_train.sh 0,1 exist_ok
```
### 3. How to activate and deactivate to prevent the folder (e.g. ./saved/) from being deleted
```bash
$ sudo chattr +a ./saved/ # Activate
$ sudo chattr -a ./saved/ # Deactivate
```


## 7. How to test the CNN model for the CIFAR-10 dataset <a name="test"></a>
```bash
$ bash ./script/cifar_10/bash_test.sh 0,1 ./saved/model/CIFAR_10/exist_ok/config.yaml ./saved/model/CIFAR_10/exist_ok/ckpt-best.pth
```


## 8. How to export the trained CNN model <a name="export"></a>
- Please note that the torch.jit.script has not been supported with torch.nn.DataParallel yet.
  ```bash
  $ bash ./script/cifar_10/bash_export.sh ./saved/model/CIFAR_10/exist_ok/config.yaml ./saved/model/CIFAR_10/exist_ok/ckpt-best.pth
  ```


## 9. How to inference the trained CNN model <a name="inference"></a>
```bash
$ bash ./script/cifar_10/bash_inference.sh 0,1 true ./saved/model/CIFAR_10/exist_ok/model_scripted.pth ./asset/image/abandoned_ship_s_000213.png "[0.4914, 0.4822, 0.4465]" "[0.2023, 0.1994, 0.2010]"
```
```bash
[main_inference.py:37] tensor_output:  <class 'torch.Tensor'> cuda:0 torch.float32 torch.Size([1, 10]) tensor([[-12.4480,  17.1307,   6.6553,  29.2508, -14.8528,  -7.1591,   1.7728, 1.8443, -19.9303,  -2.2677]], device='cuda:0')
[main_inference.py:38] result:  <class 'int'> 3
```


## 10. Result <a name="result"></a>
|CNN model|Cross entropy loss|Top-1 accuracy|Top-3 accuracy|
|:-------:|:----:|:-----:|:-----:|
|ResNet-18|0.1796|95.26 %|99.33 %|


## 11. Useful commands <a name="useful_commands"></a>
### 1. How to remove experiment
```bash
$ bash ./script/bash_remove_experiment.sh CIFAR_10 exist_ok
```

### 2. How to clean Slack channel
```bash
$ bash ./bash/bash_clean_slack.sh research ${SLACK_TOKEN_USR} ${SLACK_TOKEN_BOT}
```


## 12. Todo <a name="todo"></a>
- [ ] Support the <a href="https://github.com/wandb/client"> wandb</a>
- [ ] Support learning rate finder
- [ ] Improve logger
- [ ] Support TensorRT
- [ ] 41dc06f (Dec. 15, 2020)
- [X] Check SEED
- [X] Automatic Mixed Precision (AMP)
- [X] OmegaConf


## 13. License <a name="license"></a>
- I respect and follow the license of the used libraries including python3 packages and the dataset.
- They are licensed by their own licenses.
- Please note that the only providen vujadeyoon's own codes, wrapper-codes comply with the MIT license.


## 14. Acknowledgement <a name="ack"></a>
- This project is inspired by both repositories: i) PyTorch Template Project [1] and ii) Train CIFAR10 with PyTorch [2].


## 15. Reference <a name="ref"></a>
1. [PyTorch Template Project](https://github.com/victoresque/pytorch-template)
2. [Train CIFAR10 with PyTorch](https://github.com/kuangliu/pytorch-cifar)
3. [vujade-python](https://github.com/vujadeyoon/vujade-python)
