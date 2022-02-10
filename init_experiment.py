from multi_logreg import register_pytree_node_dataclass
import datasets
import numpy as np
import jax.numpy as jnp
import jax
from preprocess_logreg import PreProcessLogReg

from argparse import Namespace
from os import path
from typing import Dict, Any
from zipfile import ZipFile

from typing import Tuple

from dataclasses import dataclass

Dataset = Tuple[jnp.ndarray, jnp.ndarray]


@jax.jit
def normalize(inputs: jnp.ndarray) -> jnp.ndarray:
    """Normalizes feature vectors so that they have unit L2-norm"""
    return inputs / jnp.linalg.norm(inputs, ord=2, axis=1, keepdims=True)


@register_pytree_node_dataclass
@dataclass(eq=True, frozen=True)
class MyLogReg(PreProcessLogReg):
    """Multi-class logistic regression, with L2 normalization applied to each input"""
    def preprocess(self, inputs: jnp.ndarray) -> jnp.ndarray:
        return normalize(inputs)


def mnist_binary(args: Namespace) -> Tuple[Namespace, Dataset, Dataset, MyLogReg]:
    """Initialize an experiment using Binary-MNIST (8 vs 3)"""
    X, y, X_test, y_test = datasets.mnist_binary(8, neg_class=3, dtype=np.float64)

    train = X, y
    test = X_test, y_test

    model = MyLogReg(lamb=args.lamb, epsilon=args.epsilon, delta=args.delta, sigma=args.sigma,
                     classes=np.array([0.0, 1.0]))

    args.feature_min = 0.0 if args.feature_min is None else args.feature_min
    args.feature_max = 1.0 if args.feature_max is None else args.feature_max 

    return args, train, test, model


def mnist(args: Namespace) -> Tuple[Namespace, Dataset, Dataset, MyLogReg]:
    """Initialize an experiment using MNIST"""
    X, y, X_test, y_test = datasets.mnist(dtype=np.float64)
    
    train = X, y
    test = X_test, y_test

    model = MyLogReg(lamb=args.lamb, epsilon=args.epsilon, delta=args.delta, sigma=args.sigma,
                     classes=np.arange(10.0))

    args.feature_min = 0.0 if args.feature_min is None else args.feature_min
    args.feature_max = 1.0 if args.feature_max is None else args.feature_max 

    return args, train, test, model


def fashion_mnist(args: Namespace) -> Tuple[Namespace, Dataset, Dataset, MyLogReg]:
    """Initialize an experiment using Fashion-MNIST"""
    X, y, X_test, y_test = datasets.fashion_mnist(dtype=np.float64)
    
    train = X, y
    test = X_test, y_test

    model = MyLogReg(lamb=args.lamb, epsilon=args.epsilon, delta=args.delta, sigma=args.sigma,
                     classes=np.arange(10.0))

    args.feature_min = 0.0 if args.feature_min is None else args.feature_min
    args.feature_max = 1.0 if args.feature_max is None else args.feature_max 

    return args, train, test, model


def har(args: Dict[str, Any]) -> Tuple[Dict[str, Any], Dataset, Dataset, MyLogReg]:
    """Initialize an experiment using HAR"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
    filename = "UCI_HAR_Dataset.zip"
    datasets._download(url, filename)

    arrays = {'X': 'UCI HAR Dataset/train/X_train.txt',
              'y': 'UCI HAR Dataset/train/y_train.txt',
              'X_test': 'UCI HAR Dataset/test/X_test.txt',
              'y_test': 'UCI HAR Dataset/test/y_test.txt'}

    with ZipFile(path.join(datasets._DATA, filename)) as z:
        for a, p in arrays.items():
            with z.open(p) as f:
                arrays[a] = np.loadtxt(f)

    train = arrays['X'], arrays['y']
    test = arrays['X_test'], arrays['y_test']

    model = MyLogReg(lamb=args.lamb, epsilon=args.epsilon, delta=args.delta, sigma=args.sigma,
                     classes=np.arange(1.0, 7.0))

    args.feature_min = -1.0 if args.feature_min is None else args.feature_min
    args.feature_max = 1.0 if args.feature_max is None else args.feature_max 

    return args, train, test, model
