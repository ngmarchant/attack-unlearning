import jax
import jax.numpy as jnp

from typing import Tuple, Callable, Optional, Sequence
from preprocess_logreg import PreProcessLogReg

Dataset = Tuple[jnp.ndarray, jnp.ndarray]


def influence_norm(model: PreProcessLogReg, ignore_model_dep: bool = True, **kwargs) -> \
        Callable[[jnp.ndarray, jnp.ndarray, Dataset, jnp.ndarray, Optional[Sequence[jnp.ndarray]]], float]:
    """Adversary's surrogate objective based on the influence norm

    Args:
        model: model on which to launch the attack.
        ignore_model_dep: whether to ignore the dependence of the model on the poisoned data. Defaults to True.
        **kwargs: other keyword arguments passed to the model's `fit` method.

    Return:
        an objective function to _minimize_ with the following arguments:
            input_poison: inputs of poisoned data to optimize
            targets_poison: fixed targets for the poisoned data (not optimized)
            data_retain: clean training data that will be retained.
            rng: a PRNGKey (an array with shape (2,) and dtype uint32)
            hessians_retain (optional): hessians of the objective for each model in the ensemble, evaluated on
                `data_retain`. This will be computed if not provided.
    """
    def obj(inputs_poison: jnp.ndarray, targets_poison: jnp.ndarray, data_retain: Dataset, rng: jnp.ndarray,
            hessians_retain: Optional[Sequence[jnp.ndarray]] = None) -> float:
        inputs_retain, targets_retain = data_retain
        data_poison = inputs_poison, targets_poison

        if not ignore_model_dep:
            inputs = jnp.concatenate((inputs_poison, inputs_retain), axis=0)
            targets = jnp.concatenate((targets_poison, targets_retain), axis=0)
            data = inputs, targets
            model_star, _ = model.fit(data, rng, 
                                      tolerance=kwargs.get('tolerance', 1e-4), 
                                      max_iterations=kwargs.get('max_iterations', 100))
            hessians = model_star.hessians(data)
        else:
            model_star = model
            if hessians_retain is None:
                hessians_retain = model_star.hessians(data_retain)
            hessians = hessians_retain + model_star.hessians(data_poison)

        hessians_inv = jax.tree_map(jnp.linalg.inv, hessians)
        grads = model_star.gradients(data_poison, perturbation_term=False)
        influences = jnp.einsum('ijk,ik->ij', hessians_inv, grads)
        influence_norms = jnp.linalg.norm(influences, axis=1, ord=2)

        return -jnp.sum(influence_norms)
    
    return obj


def grad_norm(model: PreProcessLogReg, ignore_model_dep: bool = True, **kwargs) -> \
        Callable[[jnp.ndarray, jnp.ndarray, Dataset, jnp.ndarray, Optional[Sequence[jnp.ndarray]]], float]:
    """Adversary's surrogate objective based on the gradient norm contribution

    Args:
        model: model on which to launch the attack.
        ignore_model_dep: whether to ignore the dependence of the model on the poisoned data. Defaults to True.
        **kwargs: other keyword arguments passed to the model's `fit` method.

    Return:
        an objective function to _minimize_ with the following arguments:
            input_poison: inputs of poisoned data to optimize
            targets_poison: fixed targets for the poisoned data (not optimized)
            data_retain: clean training data that will be retained.
            rng: a PRNGKey (an array with shape (2,) and dtype uint32)
            hessians_retain (optional): hessians of the objective for each model in the ensemble, evaluated on
                `data_retain`. This argument is ignored.
    """
    def obj(inputs_poison: jnp.ndarray, targets_poison: jnp.ndarray, data_retain: Dataset, rng: jnp.ndarray,
            hessians_retain: Optional[Sequence[jnp.ndarray]] = None) -> float:
        inputs_retain, targets_retain = data_retain
        data_poison = inputs_poison, targets_poison

        if not ignore_model_dep:
            inputs = jnp.concatenate((inputs_poison, inputs_retain), axis=0)
            targets = jnp.concatenate((targets_poison, targets_retain), axis=0)
            data = inputs, targets
            model_star, _ = model.fit(data, rng, 
                                      tolerance=kwargs.get('tolerance', 1e-4), 
                                      max_iterations=kwargs.get('max_iterations', 100))
        else:
            model_star = model

        grads = model_star.gradients(data_poison, perturbation_term=False)
        grad_norms = jnp.linalg.norm(grads, axis=1, ord=2)

        return -jnp.sum(grad_norms)
    
    return obj


def grnb_norm(model: PreProcessLogReg, ignore_model_dep: bool = True, **kwargs) -> \
        Callable[[jnp.ndarray, jnp.ndarray, Dataset, jnp.ndarray, Optional[Sequence[jnp.ndarray]]], float]:
    """Adversary's objective based on the gradient residual norm bound

    Args:
        model: model on which to launch the attack.
        ignore_model_dep: whether to ignore the dependence of the model on the poisoned data. Defaults to True.
        **kwargs: other keyword arguments passed to the model's `fit` method.

    Return:
        an objective function to _minimize_ with the following arguments:
            input_poison: inputs of poisoned data to optimize
            targets_poison: fixed targets for the poisoned data (not optimized)
            data_retain: clean training data that will be retained.
            rng: a PRNGKey (an array with shape (2,) and dtype uint32)
            hessians_retain (optional): hessians of the objective for each model in the ensemble, evaluated on
                `data_retain`. This will be computed if not provided.
    """
    def obj(inputs_poison: jnp.ndarray, targets_poison: jnp.ndarray, data_retain: Dataset, rng: jnp.ndarray,
            hessians_retain: Optional[Sequence[jnp.ndarray]] = None) -> float:
        inputs_retain, targets_retain = data_retain
        data_poison = inputs_poison, targets_poison
        
        if not ignore_model_dep:
            inputs = jnp.concatenate((inputs_poison, inputs_retain), axis=0)
            targets = jnp.concatenate((targets_poison, targets_retain), axis=0)
            data = inputs, targets
            model_star, _ = model.fit(data, rng, 
                                      tolerance=kwargs.get('tolerance', 1e-4), 
                                      max_iterations=kwargs.get('max_iterations', 100))
            hessians = model_star.hessians(data)
        else:
            model_star = model
            if hessians_retain is None:
                hessians_retain = model_star.hessians(data_retain)
            hessians = hessians_retain + model_star.hessians(data_poison)
        
        hessians_inv = jax.tree_map(jnp.linalg.inv, hessians)
        grads = model_star.gradients(data_poison, perturbation_term=False)
        influences = jnp.einsum('ijk,ik->ij', hessians_inv, grads)
        inputs = jnp.concatenate((inputs_poison, inputs_retain), axis=0)
        inputs = model_star.preprocess(inputs)
        inputs_influences = jnp.einsum('nj,ij->in', inputs, influences[:, :-1]) + influences[:, -1]
        inputs_norm = jnp.linalg.norm(inputs, ord=2)
        grnbs = (0.25 * inputs_norm * jnp.linalg.norm(influences, axis=1, ord=2) *
                 jnp.linalg.norm(inputs_influences, axis=1, ord=2))

        return -jnp.sum(grnbs)
    
    return obj
