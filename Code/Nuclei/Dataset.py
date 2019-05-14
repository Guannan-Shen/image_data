##### Preprocess 30 images and mat annotation files #############
## test pytorch and environment
#
# check CUDA version
# nvcc --version
# from pytorch website, install pytorch for python 3.6, CUDA 9.0 and linux
# https://pytorch.org/
# conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
# check anaconda environments
# conda info --envs
# conda env list
# Setting an existing project interpreter in pycharm to python 3.6 anaconda3

# import torch.nn as nn
# import torch

# from sklearn.model_selection import train_test_split
from scipy import io

######## import dataset #######
## random split to train validation and test
## then
import tensorflow as tf
print(os.getcwd())