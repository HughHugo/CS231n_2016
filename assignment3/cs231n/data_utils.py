import pickle
import numpy as np
import os
from scipy.misc import imread

def load_CIFAR_batch(filename):
    """load single batch of cifar"""
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding = 'bytes')
        x = datadict[b'data']
        y = datadict[b'labels']
        x = x.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float')
        y = np.array(y)
        return x, y

def load_CIFAR10(ROOT):
    """Load all of cifar"""
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        x, y = load_CIFAR_batch(f)
        xs.append(x)
        ys.append(y)
    x_train = np.concatenate(xs)
    y_train = np.concatenate(ys)
    del x, y
    x_test, y_test = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return x_train, y_train, x_test, y_test

def get_CIFAR10_data(num_training = 49000, num_validation = 1000, num_test = 1000):
    """
    Load the CIFA-10 dataset from disk and perform preprocessing to prepare it for classifiers. 
    These are the same steps as we used for the SVM, but condensed to a single function.
    """
    #Load the raw CIFAR-10 data
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    x_train, y_train, x_test, y_test = load_CIFAR10(cifar10_dir)
    
    #Subsample the data
    mask = range(num_training, num_training + num_validation)
    x_val = x_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    x_train = x_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    x_test = x_test[mask]
    y_test = y_test[mask]
    
    #Normalize the data: subtract the mean image
    mean_image = np.mean(x_train, axis = 0)
    x_train -= mean_image
    x_val -= mean_image
    x_test -= mean_image
    
    
    #Transpose so that chnnels come first.
    x_train = x_train.transpose(0, 3, 1, 2).copy()
    x_val = x_val.transpose(0, 3, 1, 2).copy()
    x_test = x_test.transpose(0, 3, 1, 2).copy()
    
    #Package data into a dictionary.
    return {
        'x_train': x_train, 'y_train': y_train, 
        'x_val': x_val, 'y_val': y_val, 
        'x_test': x_test, 'y_test': y_test
    }


def load_tiny_imagenet(path, dtype = np.float32, subtract_mean = True):
    """
    Load TinyImageNet. Each of TinyImageNet-100-A, TinyImageNet-100-B, and TinyImageNet-200 have the same directory structure, 
    so this can be used to load any of them.
    
    Inputs:
    - path: String giving path to the directory to load;
    - dtype: numpy datatype used to load the data;
    - subtract_mean: Whether to subtract the mean training image.
    
    Returns: A dictionary with the following entries:
    - class_names: A list where class_names[i] is a list of strings giving the WordNet names for class i in the loaded dataset;
    - x_train: (N_tr, 3, 64, 64) array of training images;
    - y_train: (N_tr, ) aray of training labels;
    - x_val: (N_val, 3, 64, 64) array of validation images;
    - y_val: (N_val, ) array of validation labels;
    - x_test: (N_test, 3, 64, 64) array of testing images;
    - y_test: (N_test, ) array of testing labels; if testing labels are not available (such as in student code) then y_test 
      will be None;
    - mean_image: (3, 64, 64) array giving mean training image.
    """
    # First load wnids
    with open(os.path.join(path, 'wnids.txt'), 'r') as f:
        wnids = [x.strip() for x in f]
    
    # Map wnids to integer labels
    wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}
    
    # Use words.txt to get names for each class
    with open(os.path.join(path, 'words.txt'), 'r') as f:
        wnid_to_words = dict(line.split('\t') for line in f)
        for wnid, words in wnid_to_words.items():
            wnid_to_words[wnid] = [w.strip() for w in words.split(',')]
    class_names = [wnid_to_words[wnid] for wnid in wnids]
    
    # Next load training data
    x_train = []
    y_train = []
    for i, wnid in enumerate(wnids):
        if (i + 1) % 20 == 0:
            print('loading training data for synset %d / %d' % (i + 1, len(wnids)))
        # To figure out the filenames we need to open the boxes file
        boxes_file = os.path.join(path, 'train', wnid, '%s_boxes.txt' % wnid)
        with open(boxes_file, 'r') as f:
            filenames = [x.split('\t')[0] for x in f]
        num_images = len(filenames)
        
        x_train_block = np.zeros((num_images, 3, 64, 64), dtype = dtype)
        y_train_block = wnid_to_label[wnid] * np.ones(num_images, dtype = np.int64)
        for j, img_file in enumerate(filenames):
            img_file = os.path.join(path, 'train', wnid, 'images', img_file)
            img = imread(img_file)
            if img.ndim == 2:
                ## grayscale file
                img.shape = (64, 64, 1)
            x_train_block[j] = img.transpose(2, 0, 1)
        x_train.append(x_train_block)
        y_train.append(y_train_block)
        
    # We need to concatenate all training data
    x_train = np.concatenate(x_train, axis = 0)
    y_train = np.concatenate(y_train, axis = 0)
    
    # Next load validation data
    with open(os.path.join(path, 'val', 'val_annotations.txt'), 'r') as f:
        img_files = []
        val_wnids = []
        for line in f:
            img_file, wnid = line.split('\t')[:2]
            img_files.append(img_file)
            val_wnids.append(wnid)
        num_val = len(img_files)
        y_val = np.array([wnid_to_label[wnid] for wnid in val_wnids])
        x_val = np.zeros((num_val, 3, 64, 64), dtype = dtype)
        for i, img_file in enumerate(img_files):
            img_file = os.path.join(path, 'val', 'images', img_file)
            img = imread(img_file)
            if img.ndim == 2:
                img.shape = (64, 64, 1)
            x_val[i] = img.transpose(2, 0, 1)
    
    # Next load test images
    # Studnets won't have test labels, so we need to iterate over files in the images directory.
    img_files = os.listdir(os.path.join(path, 'test', 'images'))
    x_test = np.zeros((len(img_files), 3, 64, 64), dtype = dtype)
    for i, img_file in enumerate(img_files):
        img_file = os.path.join(path, 'test', 'images', img_file)
        img = imread(img_file)
        if img.ndim == 2:
            img.shape = (64, 64, 1)
        x_test[i] = img.transpose(2, 0, 1)
    
    y_test = None
    y_test_file = os.path.join(path, 'test', 'test_annotations.txt')
    if os.path.isfile(y_test_file):
        with open(y_test_file, 'r') as f:
            img_file_to_wnid = {}
            for line in f:
                line = line.split('\t')
                img_file_to_wnid[line[0]] = line[1]
        y_test = [wnid_to_label[img_file_to_wnid[img_file]] for img_file in img_files]
        y_test = np.array(y_test)
    
    mean_image = x_train.mean(axis = 0)
    if subtract_mean:
        x_train -= mean_image
        x_val -= mean_image
        x_test -= mean_image
    
    return {
        'class_names': class_names, 
        'x_train': x_train, 
        'y_train': y_train, 
        'x_val': x_val, 
        'y_val': y_val, 
        'x_test': x_test, 
        'y_test': y_test, 
        'mean_image': mean_image
    }


def load_models(models_dir):
    """
    Load saved models from disk. This will attempt to unpickle all files in a directory; any files that give errors on 
    unpickling (such as README.txt) will be skipped.
    
    Inputs:
    - models_dir: String giving the path to a directory containing model files. Each model file is a picked dictionary with a 
      'model' field.
    
    Returns:
    A dictionary mapping model file names to models.
    """
    models = {}
    for model_file in os.listdir(models_dir):
        with open(os.path.join(models_dir, model_file), 'rb') as f:
            try: 
                models[model_file] = pickle.load(f)['model']
            except pickle.UnpicklingError:
                continue
    return models