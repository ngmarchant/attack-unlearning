import array
import gzip
import os
from os import path
import struct
import urllib.request

import numpy as np


_DATA = "/tmp/jax_example_data/"


def _download(url: str, filename: str, dirname: str = _DATA) -> str:
    """Download a url to a file in the JAX data temp directory."""
    if not path.exists(dirname):
        os.makedirs(dirname)
    dest_path = path.join(dirname, filename)
    if not path.isfile(dest_path):
        urllib.request.urlretrieve(url, dest_path)
        print("downloaded {} to {}".format(url, dest_path))
    return dest_path


def _partial_flatten(x):
    """Flatten all but the first dimension of an ndarray."""
    return np.reshape(x, (x.shape[0], -1))


def _one_hot(x, k, dtype=np.float32):
    """Create a one-hot encoding of x of size k."""
    return np.array(x[:, None] == np.arange(k), dtype)


def _mnist_raw():
    """Download and parse the raw MNIST dataset."""
    # CVDF mirror of http://yann.lecun.com/exdb/mnist/
    base = "https://storage.googleapis.com/cvdf-datasets/mnist/"

    def parse_labels(filename):
        with gzip.open(filename, "rb") as fh:
            _ = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, "rb") as fh:
            _, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()),
                            dtype=np.uint8).reshape(num_data, rows, cols)
    
    fnames = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
              "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
    
    paths = []
    for fname in fnames:
        paths.append(_download(base + fname, fname, _DATA + "mnist/"))

    train_images = parse_images(paths[0])
    train_labels = parse_labels(paths[1])
    test_images = parse_images(paths[2])
    test_labels = parse_labels(paths[3])

    return train_images, train_labels, test_images, test_labels


def mnist(dtype=np.float32):
    """Download, parse and process MNIST data to unit scale with one-hot labels
    
    Args:
        dtype (optional): Datatype to use
    
    Returns:
        tuple of ndarrays: training images, training labels, test images and 
            test labels.
    """
    train_images, train_labels, test_images, test_labels = _mnist_raw()
  
    train_images = _partial_flatten(train_images) / np.array(255, dtype=dtype)
    test_images = _partial_flatten(test_images) / np.array(255, dtype=dtype)
    train_labels = _one_hot(train_labels, 10)
    test_labels = _one_hot(test_labels, 10)

    return train_images, train_labels, test_images, test_labels


def mnist_binary(pos_class, neg_class=None, dtype=np.float32):
    """Generate a binary classification dataset based on MNIST
    
    Args:
        pos_class (int): Integer label in the set {0, 1, ..., 9}
            to regard as "positive"

        neg_class (int or list of ints, optional): Integer label(s) 
            of class(es) to regard as "negative".  Defaults to 
            the set of all labels excluding `pos_class`.
        
        dtype (optional): Datatype to use.
            
    Returns:
        tuple of ndarrays: training images, training labels, 
            test images and test labels.
    """
    train_images, train_labels, test_images, test_labels = mnist(dtype=dtype)
    
    # Ensure neg_class is a list of class labels
    if neg_class is None:
        neg_class = [label for label in range(10) if label != pos_class]
    elif isinstance(neg_class, int):
        neg_class = [neg_class]
    
    included_class = np.array([pos_class] + neg_class)
    if len(included_class) != 10:
        sel_train = np.sum(train_labels[:, included_class], axis=1) > 0
        sel_test = np.sum(test_labels[:, included_class], axis=1) > 0
        train_images, train_labels = train_images[sel_train], train_labels[sel_train]
        test_images, test_labels = test_images[sel_test], test_labels[sel_test]
    
    train_labels, test_labels = train_labels[:, pos_class], test_labels[:, pos_class]
    
    return train_images, train_labels, test_images, test_labels


def _fashion_mnist_raw():
    base = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    fnames = ["train-labels-idx1-ubyte.gz", "train-images-idx3-ubyte.gz",
              "t10k-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz"]

    paths = []
    for fname in fnames:
        paths.append(_download(base + fname, fname, _DATA + "fashion_mnist/"))

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28)

    return x_train, y_train, x_test, y_test


def fashion_mnist(dtype=np.float32):
    """Download, parse and process Fashion MNIST data to unit scale with 
    one-hot labels
    
    Args:
        dtype (optional): Datatype to use
    
    Returns:
        tuple of ndarrays: training images, training labels, test images and 
            test labels.
    """
    train_images, train_labels, test_images, test_labels = _fashion_mnist_raw()
  
    train_images = _partial_flatten(train_images) / np.array(255, dtype=dtype)
    test_images = _partial_flatten(test_images) / np.array(255, dtype=dtype)
    train_labels = _one_hot(train_labels, 10)
    test_labels = _one_hot(test_labels, 10)

    return train_images, train_labels, test_images, test_labels
    