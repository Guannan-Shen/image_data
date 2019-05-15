# conda install -c anaconda tensorflow-gpu
import tensorflow as tf            # with anaconda
import pandas as pd
import numpy as np

import keras
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.optimizers import Adam      # Adam or Adaptive Moment Optimization algorithms

import cv2                         # To read and manipulate images
import os                          # For filepath, directory handling
import sys                         # System-specific parameters and functions
import tqdm                        # Use smart progress meter
import matplotlib.pyplot as plt    # Python 2D plotting library
import matplotlib.cm as cm         # Color map
import datetime                    # To measure running time
from PIL import Image
import scipy.io as sio                 # load .mat file


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
tf.test.gpu_device_name()


plt.interactive(True)


CW_DIR = os.getcwd()
dir_ = '/home/guanshim/Documents/image_data/'
dir_res = dir_ + 'Reports/Nuclei/'
TRAIN_DIR = dir_ + 'DataProcessed/Nuclei/' + 'train/'
TEST_DIR = os.path.join(os.path.dirname(dir_), 'DataProcessed/Nuclei/', 'test/')

IMG_WIDTH = 500    # Default image width
IMG_HEIGHT = 500   # Default image height
IMG_CHANNELS = 3      # Default number of channels
target_size = (IMG_WIDTH, IMG_HEIGHT)

# for InceptionV3
input_width = 299
input_height = 299

## import dataset
## image
def read_image(filepath, color_mode=cv2.IMREAD_COLOR, target_size=None):
    """Read an image from a file and resize it."""
    img = cv2.imread(filepath, color_mode)
    if target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return img

def read_mat(filepath):
    mat_tmp= sio.loadmat(filepath)
    loc = mat_tmp['Positions'].T
    label = mat_tmp['Labels'].T
    mat = np.hstack((loc, label))
    ## sort by the columns of the image actually
    return mat[mat[:, 0].argsort()]           # sort by y (the column in image)

def mat_to_maskraw(mat, target_size):
    """Binary Mask and single pixel nuclei"""
    mask = np.zeros(target_size)
    for x, y, label in mat.astype(int):
        ## filp the y and x
        if x == target_size[0] & y == target_size[1]:
            mask[(y - 1), (x - 1)] = 255
        elif x == target_size[0]:
            mask[y, (x-1)] = 255
        elif y == target_size[1]:
            mask[(y-1), x] = 255
        else:
            mask[y, x] = 255
    mask1 = mask.astype(np.uint8)
    return mask1

img = read_image(TRAIN_DIR + 'image/img009.png')
img.shape

mat1 = read_mat(TRAIN_DIR + 'mat/img009_annotations.mat')

def mat_to_patch(mat, img, patch_size, target_size, dir, imgname):
    """
    Same size of maskraw and img get 27*27 patches
    Save to /patches/ .png

    """
    IMG_WIDTH = target_size[0]
    IMG_HEIGHT = target_size[1]
    for y, x, label in np.floor(mat).astype(int):
        # set boundaries
        size = int((patch_size[0] - 1) / 2)
        xmin = max(0, x - size);
        xmax = min(IMG_WIDTH - 1, x + size + 1)
        ymin = max(0, y - size);
        ymax = min(IMG_HEIGHT - 1, y + size + 1)
        # take patches
        img_pat_tmp = img[ymin:ymax, xmin:xmax, :]
        # resize
        img_pat = cv2.resize(img_pat_tmp, patch_size, interpolation=cv2.INTER_AREA)
        # save
        im = Image.fromarray(img_pat)
        im.save(dir + 'patches/' + imgname + '_' + str(x) + '_' + str(y) + '_' + str(label) + '_.png')

# data for ImageDataGenerators
# input_width 299
# input_height 299
def mat_to_patch_idg(mat, img, patch_size, target_size, dir, imgname):
    """
    Same size of maskraw and img get 27*27 patches
    Save to /patches/ .png

    """
    IMG_WIDTH = target_size[0]
    IMG_HEIGHT = target_size[1]
    for y, x, label in np.floor(mat).astype(int):
        # set boundaries
        size = int((patch_size[0] - 1) / 2)
        xmin = max(0, x - size);
        xmax = min(IMG_WIDTH - 1, x + size + 1)
        ymin = max(0, y - size);
        ymax = min(IMG_HEIGHT - 1, y + size + 1)
        # take patches
        img_pat_tmp = img[ymin:ymax, xmin:xmax, :]
        # resize
        img_pat = cv2.resize(img_pat_tmp, (input_width, input_height), interpolation=cv2.INTER_AREA)
        # save
        im = Image.fromarray(img_pat)
        if int(label) == 1:
            im.save(dir + 'cnn_class_data/' + str(label) + '/' + str(x) + '_' + str(y) + '_' + imgname + '.png')
        elif int(label) == 2:
            im.save(dir + 'cnn_class_data/' + str(label) + '/' + str(x) + '_' + str(y) + '_' + imgname + '.png')
        elif int(label) == 3:
            im.save(dir + 'cnn_class_data/' + str(label) + '/' + str(x) + '_' + str(y) + '_' + imgname + '.png')
        elif int(label) == 4:
            im.save(dir + 'cnn_class_data/' + str(label) + '/' + str(x) + '_' + str(y) + '_' + imgname + '.png')


def make_patches(patch_size, image_size = (IMG_HEIGHT, IMG_WIDTH), ):
    directory = TRAIN_DIR + 'image/'
    mat_dir = TRAIN_DIR + 'mat/'
    for i, filename in enumerate(next(os.walk(directory))[2]):
        img_path = os.path.join(directory, filename)
        img = read_image(img_path, target_size=image_size)
        # get corresponding mat
        mat_names = next(os.walk(mat_dir))[2]
        mat_n = mat_dir + mat_names[i]
        mat = read_mat(mat_n)
        mat_to_patch(mat, img, patch_size, target_size, TRAIN_DIR, filename)

    directory = TEST_DIR + 'image/'
    mat_dir = TEST_DIR + 'mat/'
    for i, filename in enumerate(next(os.walk(directory))[2]):
        img_path = os.path.join(directory, filename)
        img = read_image(img_path, target_size=image_size)
        # get corresponding mat
        mat_names = next(os.walk(mat_dir))[2]
        mat_n = mat_dir + mat_names[i]
        mat = read_mat(mat_n)
        mat_to_patch(mat, img, patch_size, target_size, TEST_DIR, filename)


def make_patches_idg(patch_size, image_size = (IMG_HEIGHT, IMG_WIDTH), ):
    directory = TRAIN_DIR + 'image/'
    mat_dir = TRAIN_DIR + 'mat/'
    for i, filename in enumerate(next(os.walk(directory))[2]):
        img_path = os.path.join(directory, filename)
        img = read_image(img_path, target_size=image_size)
        # get corresponding mat
        mat_names = next(os.walk(mat_dir))[2]
        mat_n = mat_dir + mat_names[i]
        mat = read_mat(mat_n)
        #
        filename = filename.replace('.png', '')
        mat_to_patch_idg(mat, img, patch_size, target_size, TRAIN_DIR, filename)

    directory = TEST_DIR + 'image/'
    mat_dir = TEST_DIR + 'mat/'
    for i, filename in enumerate(next(os.walk(directory))[2]):
        img_path = os.path.join(directory, filename)
        img = read_image(img_path, target_size=image_size)
        # get corresponding mat
        mat_names = next(os.walk(mat_dir))[2]
        mat_n = mat_dir + mat_names[i]
        mat = read_mat(mat_n)
        filename = filename.replace('.png', '')
        mat_to_patch_idg(mat, img, patch_size, target_size, TEST_DIR, filename)

make_patches_idg((27, 27))



base_model = InceptionV3(weights='imagenet', include_top=False)

CLASSES = 4

x = base_model.output
x = GlobalAveragePooling2D(name='avg_pool')(x)
x = Dropout(0.5)(x)
predictions = Dense(CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# check layers
for i, layer in enumerate(model.layers):
  print(i, layer.name)

# we need to freeze all our base_model layers and train the last ones.
# Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.

# freeze all our base_model layers and train the last ones
if False:
    for layer in base_model.layers:
        layer.trainable = False

else:  # or if we want to set the first 20 layers of the network to be non-trainable
    for layer in model.layers[:20]:
        layer.trainable=False
    for layer in model.layers[20:]:
        layer.trainable=True

model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

dir_train = TRAIN_DIR + 'cnn_class_data/'

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest') #included in our dependencies

EPOCHS = 10
# Found 9294 images belonging to 4 classes.
BATCH_SIZE = 32
# TOTAL / batch size
9294/32
1794/32
STEPS_PER_EPOCH = 291
VALIDATION_STEPS = 57
# input_width 299
train_generator = train_datagen.flow_from_directory(
    dir_train,
    target_size=(input_width, input_width),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

dir_train = TEST_DIR + 'cnn_class_data/'
# validation data
validation_generator = train_datagen.flow_from_directory(
    dir_train,
    target_size=(input_width, input_width),
    batch_size=BATCH_SIZE,
    class_mode='categorical')
# Found 1794 images belonging to 4 classes.

MODEL_FILE = os.path.join(os.path.dirname(dir_), 'DataProcessed/Nuclei/cnn_model/') + '5_15_inceptionv3.model'

# Transfer learning
history = model.fit_generator(
    train_generator,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=validation_generator,
    validation_steps=VALIDATION_STEPS)

model.save(MODEL_FILE)


def predict(model, img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds[0]

filename = TEST_DIR + 'cnn_class_data/1/2_92_img090.png'
img = image.load_img(filename, target_size=(input_width, input_width))
preds = predict(load_model(MODEL_FILE), img)
preds
