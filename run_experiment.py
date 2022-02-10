import numpy as onp
import jax
import jax.numpy as jnp
from jax import random
from jax.interpreters import xla
import time
from datetime import datetime
import argparse
import jax.profiler

from typing import Tuple, Dict, Any

from multi_logreg import MultiLogReg
from grad_descent import proj_grad_descent
from projections import (linf_proj, l1_proj, l2_proj, dykstra_proj)
from adv_objectives import grad_norm, influence_norm, grnb_norm

import init_experiment

# Must operate in 64 bit mode for L-BFGS
from jax.config import config
config.update("jax_enable_x64", True)

lp_proj = {
    1.0: l1_proj,
    1: l1_proj,
    2.0: l2_proj,
    2: l2_proj,
    float('inf'): linf_proj,
    'inf': linf_proj,
}

get_adv_obj = {
    'grad_norm': grad_norm,
    'influence_norm': influence_norm,
    'grnb_norm': grnb_norm
}

get_expt = {
    'mnist': init_experiment.mnist,
    'mnist_binary': init_experiment.mnist_binary,
    'har': init_experiment.har,
    'fashion_mnist': init_experiment.fashion_mnist
}


def accuracy_score(model: MultiLogReg, data: Tuple[onp.ndarray, onp.ndarray]) -> float:
    inputs, targets = data
    if targets.ndim == 2 and targets.shape[1] > 1:
        # Targets is one-hot encoded
        targets = onp.argmax(targets, axis=1)
        if model.classes is not None:
            targets = onp.asarray(model.classes)[targets, ]
    accuracy = onp.mean(model.predict(inputs).squeeze() == targets)
    return float(accuracy)


def poison_data(model: MultiLogReg, train: Tuple[onp.ndarray, onp.ndarray], ref: Tuple[onp.ndarray, onp.ndarray],
                rng: onp.ndarray, args: Dict[str, Any]) -> Tuple[onp.ndarray, onp.ndarray]:
    """Add poisoned instances to training data and update model
    
    Args:
        model: Model to use for the attack. Need not be fitted.
        train: Initial training data.
        ref: Reference instances for poisoning. The features for each poisoned 
            instance are constrained to be within the L-p ball centered on the 
            reference features. 
        rng: Random number generator state passed to the fit method for the 
            model.
        args: Parameters passed as command line arguments.
    
    Returns:
        train: Updated training data, where the poisoned instances are 
            prepended to the initial training data in 
    """
    inputs, targets = train
    inputs_ref, targets_ref = ref
    
    model, _ = model.fit(train, rng, tolerance=args['lbfgs_grad_tol'], 
                         max_iterations=args['lbfgs_max_iter'])

    proj_ops = (lambda x, x_ref: jnp.clip(x, a_min=args['feature_min'], a_max=args['feature_max']),
                lambda x, x_ref: lp_proj[args['p_norm']](args['pert_ball_radius'], x, x_ref=x_ref, axis=1))

    def proj_op(x):
        return dykstra_proj(proj_ops, x, x_ref=inputs_ref, max_iter=args['dykstra_max_iter'], tol=args['dykstra_tol'])

    rng, rng0 = random.split(rng)
    adv_obj = get_adv_obj[args['adv_obj']](model, 
                                           ignore_model_dep=args['ignore_model_dep'],
                                           tolerance=args['lbfgs_grad_tol'],
                                           max_iterations=args['lbfgs_max_iter'])

    adv_obj_value_and_grad = jax.jit(jax.value_and_grad(adv_obj))
    def obj_value_and_grad(inputs): return adv_obj_value_and_grad(inputs, targets_ref, train, rng0)

    inputs_pois = proj_grad_descent(obj_value_and_grad, inputs_ref, args['pgd_init_step'], proj_op, ord=2, axis=1, 
                                    num_iter=args['pgd_max_iter'])

    # Update training data
    inputs = onp.concatenate((inputs_pois, inputs), axis=0)
    targets = onp.concatenate((targets_ref, targets), axis=0)
    
    return inputs, targets


def prepare_data(train: Tuple[onp.ndarray, onp.ndarray], model: MultiLogReg, rng: onp.ndarray, args: Dict[str, Any]):
    """Prepare poisoned training set

    Randomly shuffles the clean training set, then conducts a data poisoning 
    attack according to the specifications provided

    Args:
        train: Initial (clean) training data.
        model: Model to use for the attack. Need not be fitted.
        rng: Random number generator state.
        args: Parameters passed as command line arguments.
    
    Returns:
        a tuple with the following entries:
            train: the poisoned training set
            idx: permutation of the original training indices
    """
    inputs, targets = train
    idx = onp.arange(inputs.shape[0])

    # Randomly permute the training data
    rng, rng0 = random.split(rng)
    idx_perm = random.permutation(rng0, idx)
    inputs, targets = inputs[idx_perm], targets[idx_perm]
    train = inputs, targets

    # Convenience variables
    num_poison = args['num_poison']

    if num_poison:
        splits = onp.cumsum(num_poison)

        # Take the first instances in the permuted training set and set them 
        # aside to be poisoned. The remaining instances are clean.
        inputs = onp.split(inputs, splits, axis=0)
        targets = onp.split(targets, splits, axis=0)
        
        # Need to reverse the order of the ids corresponding to the poisoned 
        # batches to reflect the order they're generated
        idx_perm = onp.split(idx_perm, splits, axis=0)
        idx_perm = idx_perm[-2::-1] + [idx_perm[-1]]
        idx_perm = onp.concatenate(idx_perm)
        
        inputs_ref, inputs = inputs[0:-1], inputs[-1]
        targets_ref, targets = targets[0:-1], targets[-1]
        
        train = inputs, targets

        # Generate the poisoned instances and add them back to the training set
        for (i_ref, t_ref) in zip(inputs_ref, targets_ref):
            rng, rng0 = random.split(rng)
            train = poison_data(model, train, (i_ref, t_ref), rng0, args)
    
    return train, idx_perm


def main():
   
    print(datetime.now(), flush=True)
    parser = argparse.ArgumentParser(description="Simulate a slow-down attack on a removal-enabled model")

    parser.add_argument('--file-prefix', type=str, help="common prefix for saved files") 
    parser.add_argument('--lbfgs-grad-tol', type=float, default=1e-6,
                        help="stopping condition on the L-inf norm of the gradient for L-BFGS")
    parser.add_argument('--lbfgs-max-iter', type=int, default=1000,
                        help="maximum number of iterations for L-BFGS")
    parser.add_argument('--pgd-init-step', type=float, default=0.01,
                        help="initial step size for projected gradient descent")
    parser.add_argument('--pgd-max-iter', type=int, default=50,
                        help="maximum number of iterations for projected gradient descent")
    parser.add_argument('--dykstra-tol', type=float, default=1e-6,
                        help="tolerance for Dykstra's projection algorithm")
    parser.add_argument('--dykstra-max-iter', type=int, default=50,
                        help="maximum number of iterations for Dykstra's projection algorithm")
    parser.add_argument('--num-repeats', type=int, default=100,
                        help="number of times to repeat the experiment (unlearning a sequence of instances until "
                             "retraining is triggered)")
    parser.add_argument('--num-poison', type=str, required=True,
                        help="number of instances in the training set to poison in each batch (must be a string "
                             "representation of a Python tuple or list)")
    parser.add_argument('--num-removals', type=int, required=True,
                        help="number of instances to remove")
    parser.add_argument('--no-retrain', dest='no_retrain', action='store_true',
                        help="stop removing instances if retraining is triggered")
    parser.add_argument('--lamb', type=float, default=1e-4,
                        help="L2 regularization parameter")
    parser.add_argument('--epsilon', type=float, default=1.0,
                        help="privacy loss parameter")
    parser.add_argument('--delta', type=float, default=1e-4,
                        help="probability that the privacy guarantee is violated")
    parser.add_argument('--sigma', type=float, default=1.0,
                        help="standard deviation of the coefficients of the random linear perturbation term")
    parser.add_argument('--pert-ball-radius', type=float,
                        help="radius of the L-p ball centred on the reference image within which poisoned examples "
                             "must lie")
    parser.add_argument('--p-norm', type=float, default=1,
                        help="parameter p specifying the L-p space")
    parser.add_argument('--adv-obj', type=str, default="influence_norm",
                        help="the objective that the adversary is attempting to maximize - valid values are "
                             "'grad_norm', 'influence_norm' or 'grnb_norm'")
    parser.add_argument('--adv-use-test', dest='adv_use_train', action='store_false',
                        help="attacker uses an orthogonal training set to generate poisoned examples")
    parser.add_argument('--adv-use-train', dest='adv_use_train', action='store_true',
                        help="attacker uses the defender's training set to generate poisoned examples")
    parser.add_argument('--rng-seed', type=int, default=0,
                        help="integer seed to initialize the pseudo-random number generator")
    parser.add_argument('--feature-min', type=float, default=None,
                        help="lower bound on feature size")
    parser.add_argument('--feature-max', type=float, default=None,
                        help="upper bound on feature size")
    parser.add_argument('--expt-name', type=str,
                        help="experiment name: valid values are 'mnist', 'mnist_binary', 'har' or 'fashion_mnist'")
    parser.add_argument('--ignore-model-dep', dest='ignore_model_dep', action='store_true',
                        help="ignore the model dependence on the poisoned data in the adversarial objective")
    parser.add_argument('--account-model-dep', dest='ignore_model_dep', action='store_false',
                        help="account for the model dependence on the poisoned data in the adversarial objective")
    parser.set_defaults(adv_use_train=True, no_retrain=False, ignore_model_dep=True)
    args = parser.parse_args()
    
    args.num_poison = eval(args.num_poison)
    total_num_poison = sum(args.num_poison)
    
    grnb_history_file = args.file_prefix + "_grnb_history.npy"
    test_acc_history_file = args.file_prefix + "_test_acc_history.npy"
    retrain_history_file = args.file_prefix + "_retrain_history.npy"
    pois_examples_file = args.file_prefix + "_pois_examples.npy"
    times_pgd_file = args.file_prefix + "_times_pgd.npy"
    times_unlearn_file = args.file_prefix + "_times_unlearn.npy"
    pois_idx_file = args.file_prefix + "_pois_idx.npy"

    print("Loading data", flush=True)
    args, train_clean, test, model_orig = get_expt[args.expt_name](args)
    
    args_dict = vars(args)
    print(args_dict)

    # Model template used by adversary (they don't account for obj perturbation)
    model_adv = model_orig.set_sigma(0.0)

    print("Using random seed {}".format(args.rng_seed), flush=True)
    rng = random.PRNGKey(args.rng_seed)
    
    print("Model will be re-trained from scratch when the GRNB exceeds {}".format(model_orig.grnb_thres), flush=True)

    print("Running {} repeats of unlearning experiment".format(args.num_repeats), flush=True)
    retrain_history = []
    grnb_history = []
    test_acc_history = []
    pois_examples = []
    pois_idx = []
    times_pgd = []
    times_unlearn = []
    t0 = time.time()
    for t in range(args.num_repeats):
        
        t_pgd_start = time.time()
        rng, rng0 = random.split(rng)

        if args.adv_use_train:
            # Attacker has access to the defender's training set.
            train, idx_perm = prepare_data(train_clean, model_adv, rng0, args_dict)
        else:
            # Attacker has access to an orthogonal data set (the MNIST test set), but cannot access the defender's 
            # training set.
            train, idx_perm = prepare_data(test, model_adv, rng0, args_dict)
            train = (onp.concatenate((train[0][:total_num_poison], train_clean[0]), axis=0),
                     onp.concatenate((train[1][:total_num_poison], train_clean[1]), axis=0))
        retain_weights = onp.ones(train[0].shape[0], dtype=bool)
        delete_weights = onp.zeros_like(retain_weights, dtype=bool)
        this_pois_idx = onp.asarray(idx_perm[:total_num_poison])
        this_pois_inputs = onp.asarray(train[0][:total_num_poison])
        pois_idx.append(this_pois_idx)
        pois_examples.append(this_pois_inputs)
        
        t_pgd_end = time.time()
        times_pgd.append(t_pgd_end - t_pgd_start)

        if (t % 10) == 0 and t != 0:
            # Fix OOM issues
            xla._xla_callable.cache_clear()
            t1 = time.time()
            avg_time = (t1 - t0)/t
            msg = "Completed {} of {} repeats.".format(t, args.num_repeats)
            msg += " Average time per experiment is {} seconds.".format(avg_time)
            if args.no_retrain:
                avg_num_removals = onp.mean([len(x) for x in retrain_history])
                msg += " Average number of removals before retraining is {}".format(avg_num_removals)
            else:    
                retrain_freq = onp.mean(onp.concatenate(retrain_history))
                msg += " Frequency of retraining is {}.".format(retrain_freq)
            print(msg, flush=True)
        
        # Fit model based on (poisoned) training data received by the central authority
        rng, rng0 = random.split(rng)
        model, _ = model_orig.fit(train, rng0, tolerance=args.lbfgs_grad_tol, max_iterations=args.lbfgs_max_iter)

        this_grnb_history = []
        this_test_acc_history = []
        this_retrain_history = []
        this_unlearn_time = 0.0
        gram_matrix = None
        # Unlearn instances in order
        for i in range(0, args.num_removals):
            this_test_acc = accuracy_score(model, test)
            this_test_acc_history.append(this_test_acc)
            
            # Unlearn the next example
            t_unlearn_start = time.time()
            
            # Update training set
            delete_weights[i] = True
            retain_weights[i] = False

            # Update model
            rng, rng0 = random.split(rng)
            model, diagnostics, gram_matrix, retrained = model.unlearn(
                train, delete_weights, retain_weights, rng0, prev_gram_matrix=gram_matrix,
                enforce_grnb_constraint=not args.no_retrain, tolerance=args.lbfgs_grad_tol,
                max_iterations=args.lbfgs_max_iter, use_full_data_hess_approx=True
            )
            
            # Reset delete array
            delete_weights[i] = False

            t_unlearn_end = time.time()
            this_unlearn_time += t_unlearn_end - t_unlearn_start
            
            retrained = bool(retrained)

            if args.no_retrain:
                # Check whether model would have been retrained if GRNB constraint was enforced
                if retrained := model.grnb >= model.grnb_thres:
                    break
            
            this_retrain_history.append(retrained)
            this_grnb_history.append(float(model.grnb))
        
        grnb_history.append(this_grnb_history)
        retrain_history.append(this_retrain_history)
        test_acc_history.append(this_test_acc_history)
        times_unlearn.append(this_unlearn_time)
        # jax.profiler.save_device_memory_profile(f"memory{t}.prof")
        
    print("Saving gradient residual norm bound histories to {}".format(grnb_history_file), flush=True)
    grnb_history = onp.array(grnb_history, dtype=object)
    onp.save(grnb_history_file, grnb_history)
    
    print("Saving test accuracy histories to {}".format(test_acc_history_file), flush=True)
    test_acc_history = onp.array(test_acc_history, dtype=object)
    onp.save(test_acc_history_file, test_acc_history)
    
    print("Saving retraining histories to {}".format(retrain_history_file), flush=True)
    retrain_history = onp.array(retrain_history, dtype=object)
    onp.save(retrain_history_file, retrain_history)
    
    if args.num_poison:
        print("Saving ids of poisoned examples {}".format(pois_idx_file), flush=True)
        pois_idx = onp.array(pois_idx)
        onp.save(pois_idx_file, pois_idx)

        print("Saving poisoned examples to {}".format(pois_examples_file), flush=True)
        pois_examples = onp.array(pois_examples)
        onp.save(pois_examples_file, pois_examples)
    
        print("Saving wall clock times for projected gradient descent to {}".format(times_pgd_file), flush=True)
        times_pgd = onp.array(times_pgd)
        onp.save(times_pgd_file, times_pgd)
    
    print("Saving wall clock times for unlearning to {}".format(times_unlearn_file), flush=True)
    times_unlearn = onp.array(times_unlearn)
    onp.save(times_unlearn_file, times_unlearn)


if __name__ == '__main__':
    main()
