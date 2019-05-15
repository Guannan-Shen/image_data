# import modules
# virtual environment
# conda create -n tensorflow_gpuenv tensorflow-gpu
# conda activate tensorflow_gpuenv

# conda install -c anaconda tensorflow-gpu
import tensorflow as tf            # with anaconda
tf.Session(config=tf.ConfigProto(log_device_placement=True))

# nvcc --version
# nvidia-smi
import pandas as pd
import numpy as np
import sklearn.model_selection     # For using KFold
import keras.preprocessing.image   # For using image generation
import datetime                    # To measure running time
import skimage.transform           # For resizing images
import skimage.morphology          # For using image labeling
import cv2                         # To read and manipulate images
import os                          # For filepath, directory handling
import sys                         # System-specific parameters and functions
import tqdm                        # Use smart progress meter
import seaborn as sns              # For pairplots
import matplotlib.pyplot as plt    # Python 2D plotting library
from PIL import Image
import matplotlib.cm as cm         # Color map
import scipy as sc                 # load .mat file
import statistics as stats
plt.interactive(True)
CW_DIR = os.getcwd()
dir = '/home/guanshim/Documents/image_data/'
dir_res = dir + 'Reports/Nuclei/'
# global constants and parameter
# Global constants.

IMG_CHANNELS = 3      # Default number of channels

TRAIN_DIR = dir + 'DataProcessed/Nuclei/' + 'train/'
TEST_DIR = os.path.join(os.path.dirname(dir), 'DataProcessed/Nuclei/', 'test/')

# Global variables.
min_object_size = 1       # Minimal nucleous size in pixels
x_train = []
y_train = []
x_test = []
y_test_pred_proba = {}
y_test_pred = {}

# Display working/train/test directories.
print('CW_DIR = {}'.format(CW_DIR))
print('TRAIN_DIR = {}'.format(TRAIN_DIR))
print('TEST_DIR = {}'.format(TEST_DIR))

## import dataset
## image
def read_image(filepath, color_mode=cv2.IMREAD_COLOR, target_size=None):
    """Read an image from a file and resize it."""
    img = cv2.imread(filepath, color_mode)
    if target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return img

def read_mask(filepath, target_size=None):
    """Read and resize masks contained in a given directory."""
    mask = cv2.imread(filepath, 0)
    if target_size:
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_AREA)
    return mask


def read_mat(filepath):
    mat_tmp= sc.io.loadmat(filepath)
    loc = mat_tmp['Positions'].T
    label = mat_tmp['Labels'].T
    mat = np.hstack((loc, label))
    ## sort by the columns of the image actually
    return mat[mat[:, 0].argsort()]

# flatten and unflatten image
# img.reshape(-1,3).shape

target_size = (500,500)

mat1 = read_mat(TRAIN_DIR + 'mat/img009_annotations.mat')
mat1.shape

def mat_to_maskraw(mat, target_size):
    """Binary Mask"""
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
mask1 = mat_to_maskraw(mat1, (500,500))

plt.imshow(mask1, cmap='gray')
plt.show
plt.savefig(dir_res + 'binary_mask_raw.png')

def mat_to_mask2(mat, target_size):
    """Binary Mask"""
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
    kernel = np.ones((3, 3), np.uint8)
    # Dilation increases object boundary to background.
    # from center to nuclei region
    mask2 = cv2.dilate(mask1, kernel, iterations=4)
    # from region to the center
    # mask3 = cv2.erode(mask2, kernel, iterations=4)
    return mask2.astype(np.uint8)

mask2 = mat_to_mask2(mat1, (500,500))
mask2.shape

plt.imshow(mask2, cmap='gray')
plt.show
plt.savefig(dir_res + 'binary_mask.png')

def mat_to_mask4(mat, target_size):
    """Classes Mask"""
    mask = np.zeros(target_size)
    for x, y, label in mat.astype(int):
        ## filp the y and x
        if x == target_size[0] & y == target_size[1]:
            mask[(y - 1), (x - 1)] = label
        elif x == target_size[0]:
            mask[y, (x - 1)] = label
        elif y == target_size[1]:
            mask[(y - 1), x] = label
        else:
            mask[y, x] = label
    mask1 = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    # Dilation increases object boundary to background.
    # from center to nuclei region
    mask2 = cv2.dilate(mask1, kernel, iterations=4)
    # from region to the center
    # mask3 = cv2.erode(mask2, kernel, iterations=4)
    return mask2.astype(np.uint8)

mask4 = mat_to_mask4(mat1, (500,500))
plt.imshow(mask4, cmap='gray')
plt.show
plt.savefig(dir_res + 'img009classes_mask.png')



img = read_image(TRAIN_DIR + 'image/img009.png')
plt.imshow(img)
plt.show
plt.savefig(dir_res + 'img009.png')

# 1, 2, 3, and 4 represent
# epithelial, fibroblast, inflammatory, and other nuclei, respectively.

def label_image(mat, img, target_size):
    """Label Original Image"""
    # Red 	(255,0,0) 1
    # lime  (0,255,0) 2
    # blue  (0,0,255) 3
    # yellow (255,255,0) 4
    mask = np.zeros(target_size)
    kernel = np.ones((3, 3), np.uint8)
    for x, y, label in mat.astype(int):
        ## filp the y and x
        mask[y, x] = label
    mask1 = mask.astype(np.uint8)
    # Dilation increases object boundary to background.
    # from center to nuclei region
    mask2 = cv2.dilate(mask1, kernel, iterations=3)
    # label img
    img[mask2 == 1] = (255, 0, 0)
    img[mask2 == 2] = (0, 255, 0)
    img[mask2 == 3] = (0, 0, 255)
    img[mask2 == 4] = (255, 255, 0)
    return img

img2 = label_image(mat1, img, (500,500))
plt.imshow(img2)
plt.show
plt.savefig(dir_res + 'img009_withlabel.png')


def imgpat_nuclei():
    """ Get 27*27 image patches based on the location of nuclei"""


## image data property
directory = TRAIN_DIR + 'image/'
img_shape = []
for i,filename in enumerate(next(os.walk(directory))[2]):
        img_path = os.path.join(directory, filename)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_shape.append(img.shape)
# all images are 500 by 500

## .mat data property
directory = TRAIN_DIR + 'mat/'
num_nuclei = []
for i,filename in enumerate(next(os.walk(directory))[2]):
        mat_path = os.path.join(directory, filename)
        mat = sc.io.loadmat(mat_path)
        label = mat['Labels']
        num_nuclei.append(len(label[0]))

fig, ax = plt.subplots(tight_layout=True)
ax.hist(num_nuclei)
ax.annotate('Mean = %.1f\ns.d. = %.4lf\nProportion = %.4lf'%(stats.mean(num_nuclei),
                                        stats.stdev(num_nuclei),
                                        stats.mean([x/250000 for x in num_nuclei] )),
            xy=(600, 6))

fig.savefig(dir_res + 'hist_num_nuclei.png')
print('Unbalanced problem, 1:670')

## mat property
# mat = sc.io.loadmat(TRAIN_DIR + 'mat/img009_annotations.mat')    # a dict object, contains 3 object
# label = mat['Labels']                      # axis 0 and size 1 ndarray, labels type 1,2,3,4
# pos = mat['Positions']                     # axis 0 and size 2, assuming pos[0] as rows and pos[1] as cols

y_test_pred_proba = {}
y_test_pred = {}
# float to integer
IMG_WIDTH = 500      # Default image width
IMG_HEIGHT = 500      # Default image height
# load raw data #
def load_raw_data(image_size=(IMG_HEIGHT, IMG_WIDTH)):
    x_train, y_train, x_test, y_test = [], [], [], []
    # TRAIN_DIR TEST_DIR
    # train
    directory = TRAIN_DIR + 'image/'
    for i, filename in enumerate(next(os.walk(directory))[2]):
        img_path = os.path.join(directory, filename)
        img = read_image(img_path, target_size=image_size)
        x_train.append(img)

    directory = TRAIN_DIR + 'mat/'
    for i, filename in enumerate(next(os.walk(directory))[2]):
        mat_path = os.path.join(directory, filename)
        mat = read_mat(mat_path)
        # all the input images are within 500, 500
        mask2 = mat_to_mask2(mat, (500, 500))
        filename = filename.replace('_annotations.mat', 'mask')
        im = Image.fromarray(mask2)
        im.save(TRAIN_DIR + 'mask2/' + filename + '.png')

    directory = TRAIN_DIR + 'mask2/'
    for i, filename in enumerate(next(os.walk(directory))[2]):
        mask_path = os.path.join(directory, filename)
        mask = read_mask(mask_path, target_size=image_size)
        y_train.append(mask)
    # test
    directory = TEST_DIR + 'image/'
    for i, filename in enumerate(next(os.walk(directory))[2]):
        img_path = os.path.join(directory, filename)
        img = read_image(img_path, target_size=image_size)
        x_test.append(img)

    directory = TEST_DIR + 'mat/'
    for i, filename in enumerate(next(os.walk(directory))[2]):
        mat_path = os.path.join(directory, filename)
        mat = read_mat(mat_path)
        mask2 = mat_to_mask2(mat, (500, 500))
        filename = filename.replace('_annotations.mat', 'mask')
        im = Image.fromarray(mask2)
        im.save(TEST_DIR + 'mask2/' + filename + '.png')

    directory = TEST_DIR + 'mask2/'
    for i, filename in enumerate(next(os.walk(directory))[2]):
        mask_path = os.path.join(directory, filename)
        mask = read_mask(mask_path, target_size=image_size)
        y_test.append(mask)

    x_train = np.array(x_train)
    y_train = np.expand_dims(np.array(y_train), axis=4)
    x_test = np.array(x_test)
    y_test = np.expand_dims(np.array(y_test), axis=4)

    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_raw_data()
x_train.shape
y_train.shape
x_test.shape
y_test.shape


# data properties
def read_train_data_properties(img_dir, mask_dir):
    """Read basic properties of training images and masks"""
    tmp = []
    for i, filename in enumerate(next(os.walk(img_dir))[2]):
        img_path = os.path.join(img_dir, filename)
        num_masks =  1   #  len(next(os.walk(mask_dir))[2])
        img_name = filename
        img_name_id = os.path.splitext(img_name)[0]
        img_shape = read_image(img_path).shape
        mask_names = next(os.walk(mask_dir))[2]
        mask_n = mask_dir + mask_names[i]
        tmp.append(['{}'.format(img_name_id), img_shape[0], img_shape[1],
                    img_shape[0]/img_shape[1], img_shape[2], num_masks,
                    img_path, mask_n])

    train_df = pd.DataFrame(tmp, columns = ['img_id', 'img_height', 'img_width',
                                            'img_ratio', 'num_channels',
                                            'num_masks', 'image_path', 'mask_dir'])
    return train_df
train_df = read_train_data_properties(TRAIN_DIR + 'image/', TRAIN_DIR + 'mask2/')
test_df = read_train_data_properties(TEST_DIR + 'image/', TEST_DIR + 'mask2/')
print('train_df:')
print(train_df.describe())
print('')
print('test_df:')
print(test_df.describe())

# Study the pixel intensity. On average the red, green and blue channels have similar
# intensities for all images. It should be noted that the background can be dark
# (black) as  as well as light (white).
def img_intensity_pairplot(x):
    """Plot intensity distributions of color channels."""
    df = pd.DataFrame()
    df['Gray'] = np.mean(x[:,:,:,:], axis=(1,2,3))
    if x.shape[3]==3:
        df['Red'] = np.mean(x[:,:,:,0], axis=(1,2))
        df['Blue'] = np.mean(x[:,:,:,1], axis=(1,2))
        df['Green'] = np.mean(x[:,:,:,2], axis=(1,2))
    return df

color_df = img_intensity_pairplot(np.concatenate([x_train, x_test]))
color_df['images'] = ['train']*len(x_train) + ['test']*len(x_test)
sns.pairplot(color_df, hue = 'images')
plt.savefig(dir_res + 'Train_test_raw_intensity.png')


# Collection of methods for basic data manipulation like normalizing, inverting,
# color transformation and generating new images/masks

def normalize_imgs(data):
    """Normalize images."""
    return normalize(data, type_=1)


def normalize_masks(data):
    """Normalize masks."""
    return normalize(data, type_=1)


def normalize(data, type_=1):
    """Normalize data."""
    if type_ == 0:
        # Convert pixel values from [0:255] to [0:1] by global factor
        data = data.astype(np.float32) / data.max()
    if type_ == 1:
        # Convert pixel values from [0:255] to [0:1] by local factor
        div = data.max(axis=tuple(np.arange(1, len(data.shape))), keepdims=True)
        div[div < 0.01 * data.mean()] = 1.  # protect against too small pixel intensities
        data = data.astype(np.float32) / div
    if type_ == 2:
        # Standardisation of each image
        data = data.astype(np.float32) / data.max()
        mean = data.mean(axis=tuple(np.arange(1, len(data.shape))), keepdims=True)
        std = data.std(axis=tuple(np.arange(1, len(data.shape))), keepdims=True)
        data = (data - mean) / std

    return data


def trsf_proba_to_binary(y_data):
    """Transform propabilities into binary values 0 or 1."""
    return np.greater(y_data, .5).astype(np.uint8)


def invert_imgs(imgs, cutoff=.5):
    '''Invert image if mean value is greater than cutoff.'''
    imgs = np.array(list(map(lambda x: 1. - x if np.mean(x) > cutoff else x, imgs)))
    return normalize_imgs(imgs)


def imgs_to_grayscale(imgs):
    '''Transform RGB images into grayscale spectrum.'''
    if imgs.shape[3] == 3:
        imgs = normalize_imgs(np.expand_dims(np.mean(imgs, axis=3), axis=3))
    return imgs


def generate_images(imgs, seed=None):
    """Generate new images."""
    # Transformations.
    image_generator = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=90., width_shift_range=0.02, height_shift_range=0.02,
        zoom_range=0.10, horizontal_flip=True, vertical_flip=True)

    # Generate new set of images
    imgs = image_generator.flow(imgs, np.zeros(len(imgs)), batch_size=len(imgs),
                                shuffle=False, seed=seed).next()
    return imgs[0]


def generate_images_and_masks(imgs, masks):
    """Generate new images and masks."""
    seed = np.random.randint(10000)
    imgs = generate_images(imgs, seed=seed)
    masks = trsf_proba_to_binary(generate_images(masks, seed=seed))
    return imgs, masks


def preprocess_raw_data(x_train, y_train, x_test, grayscale=False, invert=False):
    """Preprocessing of images and masks."""
    # Normalize images and masks
    x_train = normalize_imgs(x_train)
    y_train = trsf_proba_to_binary(normalize_masks(y_train))
    x_test = normalize_imgs(x_test)
    print('Images normalized.')

    if grayscale:
        # Remove color and transform images into grayscale spectrum.
        x_train = imgs_to_grayscale(x_train)
        x_test = imgs_to_grayscale(x_test)
        print('Images transformed into grayscale spectrum.')

    if invert:
        # Invert images, such that each image has a dark background.
        x_train = invert_imgs(x_train)
        x_test = invert_imgs(x_test)
        print('Images inverted to remove light backgrounds.')

    return x_train, y_train, x_test

x_train, y_train, x_test = preprocess_raw_data(x_train, y_train, x_test, invert=True)

def get_nuclei_sizes():
    nuclei_sizes = []
    mask_idx = []
    for i in range(len(y_train)):
        mask = y_train[i].reshape(y_train.shape[1], y_train.shape[2])
        lab_mask = skimage.morphology.label(mask > .5)
        (mask_labels, mask_sizes) = np.unique(lab_mask, return_counts=True)
        nuclei_sizes.extend(mask_sizes[1:])
        mask_idx.extend([i]*len(mask_sizes[1:]))
    return mask_idx, nuclei_sizes

mask_idx, nuclei_sizes = get_nuclei_sizes()
nuclei_sizes_df = pd.DataFrame()
nuclei_sizes_df['mask_index'] = mask_idx
nuclei_sizes_df['nucleous_size'] = nuclei_sizes

print(nuclei_sizes_df.describe())
nuclei_sizes_df.sort_values(by='nucleous_size', ascending=True).head(10)

# after preprocessing
color_df = img_intensity_pairplot(np.concatenate([x_train, x_test]))
color_df['images'] = ['train']*len(x_train) + ['test']*len(x_test)
sns.pairplot(color_df, hue = 'images')
plt.savefig(dir_res + 'train_test_processed.png')

def imshow_args(x):
    """Matplotlib imshow arguments for plotting."""
    if len(x.shape)==2: return x, cm.gray
    if x.shape[2]==1: return x[:,:,0], cm.gray
    return x, None

# Check the image transformation procedure (resizing, normalizing, inverting)
# by looking at a sample.
def img_comparison_plot(n):
    """Plot the original and transformed images/masks."""
    fig, axs = plt.subplots(1,4,figsize=(20,20))
    axs[0].imshow(read_image(train_df['image_path'].loc[n]))
    axs[0].set_title('{}.) original image'.format(n))
    img, img_type = imshow_args(x_train[n])
    axs[1].imshow(img, img_type)
    axs[1].set_title('{}.) transformed image'.format(n))
    axs[2].imshow(read_mask(train_df['mask_dir'].loc[n]), cm.gray)
    axs[2].set_title('{}.) original mask'.format(n))
    axs[3].imshow(y_train[n,:,:,0], cm.gray)
    axs[3].set_title('{}.) transformed mask'.format(n));

n = 15 # np.random.randint(0, len(x_train))
img_comparison_plot(n)


# Generate new images/masks via transformations applied on the original
# images/maks. Data augmentations can be used for regularization.
def plot_generated_image_mask(n):
    fig, axs = plt.subplots(1,4,figsize=(20,20))
    img_new, mask_new = generate_images_and_masks(x_train[n:n+1], y_train[n:n+1])
    img, img_type = imshow_args(x_train[n])
    axs[0].imshow(img, img_type)
    axs[0].set_title('{}. original image'.format(n))
    img, img_type = imshow_args(img_new[0])
    axs[1].imshow(img, img_type)
    axs[1].set_title('{}. generated image'.format(n))
    axs[2].imshow(y_train[n,:,:,0], cmap='gray')
    axs[2].set_title('{}. original mask'.format(n))
    axs[3].imshow(mask_new[0,:,:,0], cmap='gray')
    axs[3].set_title('{}. generated mask'.format(n));

np.random.seed(7)
n = np.random.randint(len(x_train))
n = 15
plot_generated_image_mask(n)
plt.savefig(dir_res + 'data_augmentation.png')

# score metric #
def get_labeled_mask(mask, cutoff=.5):
    """Object segmentation by labeling the mask."""
    mask = mask.reshape(mask.shape[0], mask.shape[1])
    lab_mask = skimage.morphology.label(mask > cutoff)

    # Keep only objects that are large enough.
    (mask_labels, mask_sizes) = np.unique(lab_mask, return_counts=True)
    if (mask_sizes < min_object_size).any():
        mask_labels = mask_labels[mask_sizes < min_object_size]
        for n in mask_labels:
            lab_mask[lab_mask == n] = 0
        lab_mask = skimage.morphology.label(lab_mask > cutoff)

    return lab_mask


def get_iou(y_true_labeled, y_pred_labeled):
    """Compute non-zero intersections over unions."""
    # Array of different objects and occupied area.
    (true_labels, true_areas) = np.unique(y_true_labeled, return_counts=True)
    (pred_labels, pred_areas) = np.unique(y_pred_labeled, return_counts=True)

    # Number of different labels.
    n_true_labels = len(true_labels)
    n_pred_labels = len(pred_labels)

    # Each mask has at least one identified object.
    if (n_true_labels > 1) and (n_pred_labels > 1):

        # Compute all intersections between the objects.
        all_intersections = np.zeros((n_true_labels, n_pred_labels))
        for i in range(y_true_labeled.shape[0]):
            for j in range(y_true_labeled.shape[1]):
                m = y_true_labeled[i, j]
                n = y_pred_labeled[i, j]
                all_intersections[m, n] += 1

                # Assign predicted to true background.
        assigned = [[0, 0]]
        tmp = all_intersections.copy()
        tmp[0, :] = -1
        tmp[:, 0] = -1

        # Assign predicted to true objects if they have any overlap.
        for i in range(1, np.min([n_true_labels, n_pred_labels])):
            mn = list(np.unravel_index(np.argmax(tmp), (n_true_labels, n_pred_labels)))
            if all_intersections[mn[0], mn[1]] > 0:
                assigned.append(mn)
            tmp[mn[0], :] = -1
            tmp[:, mn[1]] = -1
        assigned = np.array(assigned)

        # Intersections over unions.
        intersection = np.array([all_intersections[m, n] for m, n in assigned])
        union = np.array([(true_areas[m] + pred_areas[n] - all_intersections[m, n])
                          for m, n in assigned])
        iou = intersection / union

        # Remove background.
        iou = iou[1:]
        assigned = assigned[1:]
        true_labels = true_labels[1:]
        pred_labels = pred_labels[1:]

        # Labels that are not assigned.
        true_not_assigned = np.setdiff1d(true_labels, assigned[:, 0])
        pred_not_assigned = np.setdiff1d(pred_labels, assigned[:, 1])

    else:
        # in case that no object is identified in one of the masks
        iou = np.array([])
        assigned = np.array([])
        true_labels = true_labels[1:]
        pred_labels = pred_labels[1:]
        true_not_assigned = true_labels
        pred_not_assigned = pred_labels

    # Returning parameters.
    params = {'iou': iou, 'assigned': assigned, 'true_not_assigned': true_not_assigned,
              'pred_not_assigned': pred_not_assigned, 'true_labels': true_labels,
              'pred_labels': pred_labels}
    return params


def get_score_summary(y_true, y_pred):
    """Compute the score for a single sample including a detailed summary."""

    y_true_labeled = get_labeled_mask(y_true)
    y_pred_labeled = get_labeled_mask(y_pred)

    params = get_iou(y_true_labeled, y_pred_labeled)
    iou = params['iou']
    assigned = params['assigned']
    true_not_assigned = params['true_not_assigned']
    pred_not_assigned = params['pred_not_assigned']
    true_labels = params['true_labels']
    pred_labels = params['pred_labels']
    n_true_labels = len(true_labels)
    n_pred_labels = len(pred_labels)

    summary = []
    for i, threshold in enumerate(np.arange(0.5, 1.0, 0.05)):
        tp = np.sum(iou > threshold)
        fn = n_true_labels - tp
        fp = n_pred_labels - tp
        if (tp + fp + fn) > 0:
            prec = tp / (tp + fp + fn)
        else:
            prec = 0
        summary.append([threshold, prec, tp, fp, fn])

    summary = np.array(summary)
    score = np.mean(summary[:, 1])  # Final score.
    params_dict = {'summary': summary, 'iou': iou, 'assigned': assigned,
                   'true_not_assigned': true_not_assigned,
                   'pred_not_assigned': pred_not_assigned, 'true_labels': true_labels,
                   'pred_labels': pred_labels, 'y_true_labeled': y_true_labeled,
                   'y_pred_labeled': y_pred_labeled}

    return score, params_dict


def get_score(y_true, y_pred):
    """Compute the score for a batch of samples."""
    scores = []
    for i in range(len(y_true)):
        score, _ = get_score_summary(y_true[i], y_pred[i])
        scores.append(score)
    return np.array(scores)


def plot_score_summary(y_true, y_pred):
    """Plot score summary for a single sample."""
    # Compute score and assign parameters.
    score, params_dict = get_score_summary(y_true, y_pred)

    assigned = params_dict['assigned']
    true_not_assigned = params_dict['true_not_assigned']
    pred_not_assigned = params_dict['pred_not_assigned']
    true_labels = params_dict['true_labels']
    pred_labels = params_dict['pred_labels']
    y_true_labeled = params_dict['y_true_labeled']
    y_pred_labeled = params_dict['y_pred_labeled']
    summary = params_dict['summary']

    n_assigned = len(assigned)
    n_true_not_assigned = len(true_not_assigned)
    n_pred_not_assigned = len(pred_not_assigned)
    n_true_labels = len(true_labels)
    n_pred_labels = len(pred_labels)

    # Summary dataframe.
    summary_df = pd.DataFrame(summary, columns=['threshold', 'precision', 'tp', 'fp', 'fn'])
    print('Final score:', score)
    print(summary_df)

    # Plots.
    fig, axs = plt.subplots(2, 3, figsize=(20, 13))

    # True mask with true objects.
    img = y_true
    axs[0, 0].imshow(img, cmap=cm.gray)
    axs[0, 0].set_title('{}.) true mask: {} true objects'.format(n, train_df['num_masks'][n]))

    # True mask with identified objects.
    # img = np.zeros(y_true.shape)
    # img[y_true_labeled > 0.5] = 255
    img, img_type = imshow_args(y_true_labeled)
    axs[0, 1].imshow(img, img_type)
    axs[0, 1].set_title('{}.) true mask: {} objects identified'.format(n, n_true_labels))

    # Predicted mask with identified objects.
    # img = np.zeros(y_true.shape)
    # img[y_pred_labeled > 0.5] = 255
    img, img_type = imshow_args(y_pred_labeled)
    axs[0, 2].imshow(img, img_type)
    axs[0, 2].set_title('{}.) predicted mask: {} objects identified'.format(
        n, n_pred_labels))

    # Prediction overlap with true mask.
    img = np.zeros(y_true.shape)
    img[y_true > 0.5] = 100
    for i, j in assigned: img[(y_true_labeled == i) & (y_pred_labeled == j)] = 255
    axs[1, 0].set_title('{}.) {} pred. overlaps (white) with true objects (gray)'.format(
        n, len(assigned)))
    axs[1, 0].imshow(img, cmap='gray', norm=None)

    # Intersection over union.
    img = np.zeros(y_true.shape)
    img[(y_pred_labeled > 0) & (y_pred_labeled < 100)] = 100
    img[(y_true_labeled > 0) & (y_true_labeled < 100)] = 100
    for i, j in assigned: img[(y_true_labeled == i) & (y_pred_labeled == j)] = 255
    axs[1, 1].set_title('{}.) {} intersections (white) over unions (gray)'.format(
        n, n_assigned))
    axs[1, 1].imshow(img, cmap='gray');

    # False positives and false negatives.
    img = np.zeros(y_true.shape)
    for i in pred_not_assigned: img[(y_pred_labeled == i)] = 255
    for i in true_not_assigned: img[(y_true_labeled == i)] = 100
    axs[1, 2].set_title('{}.) no threshold: {} fp (white), {} fn (gray)'.format(
        n, n_pred_not_assigned, n_true_not_assigned))
    axs[1, 2].imshow(img, cmap='gray');


true_mask = y_train[n,:,:,0].copy()
lab_true_mask = get_labeled_mask(true_mask)
pred_mask = true_mask.copy() # Create predicted mask from true mask.
true_mask[lab_true_mask == 7] = 0 # Remove one object => false postive
pred_mask[lab_true_mask == 10] = 0 # Remove one object => false negative
offset = 5  # Offset.
pred_mask = pred_mask[offset:, offset:]
pred_mask = np.pad(pred_mask, ((0, offset), (0, offset)), mode="constant")
plot_score_summary(true_mask, pred_mask)


# test dilation effect
kernel = np.ones((3, 3), np.uint8)
# Dilation increases object boundary to background.
# from center to nuclei region
mask_test = np.zeros((11, 11), np.uint8)
mask_test[5-3:5+3,5-3:5+3] = 1
mask_test

mask_test1 = cv2.dilate(mask_test, kernel, iterations=4)
mask_test1