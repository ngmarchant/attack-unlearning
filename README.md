# Hard to Forget: Poisoning Attacks on Certified Machine Unlearning

This repository implements a poisoning attack against machine unlearning as proposed in the following paper:

> Marchant, N. G., Rubinstein, B. I. P., & Alfeld, S. (2022). Hard to Forget: Poisoning Attacks on Certified Machine 
Unlearning. Proceedings of the AAAI Conference on Artificial Intelligence (to appear). [[arXiv:2109.08266]](https://arxiv.org/pdf/2109.08266.pdf)

The attack aims to **increase the computational cost** of processing unlearning requests for a model trained on 
user data. This is done by poisoning user data under the attacker's control and then requesting that the 
poisoned data be unlearned. 

## Requirements

* Python >=3.9
* NumPy >=1.20
* JAX >=0.2.14
* TensorFlow Probability >=0.13
* [CUDA](https://developer.nvidia.com/cuda-downloads) >=11.1
* [cuDNN](https://developer.nvidia.com/cudnn) >=8.0.5

We used the prebuilt wheels for CUDA 11.1 and cuDNN 8.0.5 as specified in the [Pipfile](Pipfile). 
Note that CUDA and cuDNN must be installed manually: they are not bundled with JAX.

## Code structure

We attack an unlearning approach for regularized linear models due to [Guo et al. (2020)](http://proceedings.mlr.press/v119/guo20c) 
called _certified removal_. Models that support this form of unlearning are implemented in the following files:

* `binary_logreg.py`: binary logistic regression
* `multi_logreg.py`: one-vs-rest (OvR) multi-class logistic regression
* `preprocess_logreg.py`: OvR multi-class logistic regression with support for a fixed (differentiable) pre-processing 
layer

Simulations of our attack are implemented primarily in `run_experiment.py`. 
This script can be called from the command line as follows:

```bash
$ python -m run_experiment \
    --expt-name mnist_binary \
    --file-prefix mnist_binary_experiment \
    --sigma 1.0 \
    --lamb 1e-3 \
    --num-poison "[500]" \
    --num-removals 1000 \
    --no-retrain \
    --pert-ball-radius 39.25 \
    --pgd-init-step 78.5 \
    --pgd-max-iter 10 \
    --adv-obj influence_norm \
    --p-norm 1
```

Please see `run_experiment.py` for documentation and the full list of possible arguments.
