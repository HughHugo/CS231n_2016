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
