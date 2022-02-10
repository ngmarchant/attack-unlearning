import binary_logreg
from binary_logreg import BinaryLogReg

import jax
import jax.numpy as jnp
from jax import random
from jax.nn import sigmoid

from cr_model import EnsembleCRModel

from dataclasses import dataclass
import dataclasses
from typing import Optional, Tuple, Dict, TypeVar
import warnings
from functools import partial

Model = TypeVar('Model')
Dataset = Tuple[jnp.ndarray, jnp.ndarray]


def _is_one_hot_encoded(targets: jnp.ndarray) -> bool:
    """Check whether targets are one-hot encoded"""
    return targets.ndim == 2 and targets.shape[1] >= 1


def _one_hot_encode(targets: jnp.ndarray, classes: Optional[jnp.ndarray]) -> jnp.ndarray:
    """One-hot encode label vector
    
    Args:
        targets: a 1D array of class labels of shape (n_instances,) or a one-hot encoded representation of the labels
            of shape (n_instances, n_classes).
        classes: set of labels that could appear in targets. This is required unless targets is already one-hot
            encoded.

    Returns:
        one-hot encoded targets of shape (n_instances, n_classes) if n_classes > 2 or (n_instances, 1) if
        n_classes == 2.
    """
    if _is_one_hot_encoded(targets):
        targets_one_hot = targets
    else:
        if classes is None:
            raise ValueError("`classes` must be specified if targets is not one-hot encoded")
        
        targets = targets.ravel()
        index = jnp.argsort(classes)
        n_classes = index.size
        classes_sorted = jnp.asarray(classes)[index, ]
        sorted_index = jnp.searchsorted(classes_sorted, targets)
        targets_one_hot = jnp.eye(n_classes, dtype=int)[sorted_index]

    if targets_one_hot.shape[1] == 2:
        return targets_one_hot[:, [1]]
    else:
        return targets_one_hot


@jax.jit
def _decision_function(models: BinaryLogReg, inputs: jnp.ndarray) -> jnp.ndarray:
    """Vectorized implementation of the `decision_function` method for MultiLogReg"""
    decision_fn_vmap = jax.vmap(BinaryLogReg.decision_function, in_axes=(0, None), out_axes=1)
    scores = decision_fn_vmap(models, inputs)
    n_instances, n_features, _ = scores.shape
    return scores.reshape(n_instances, n_features)


@jax.jit
def _fit(models: BinaryLogReg, data: Dataset, rng: jnp.ndarray,
         data_weights: Optional[jnp.ndarray], tolerance: float, max_iterations: int) -> Tuple['MultiLogReg', Dict]:
    """Implementation of the `fit` method for MultiLogReg"""
    inputs, targets = data
    
    def f(x):
        m_dict = dataclasses.asdict(models)
        m = BinaryLogReg(**m_dict)
        targets, rng = x
        return BinaryLogReg.fit(m, (inputs, targets), rng, data_weights=data_weights, 
                                tolerance=tolerance, max_iterations=max_iterations)
    
    rngs = random.split(rng, targets.shape[1])
    models, diagnostics = jax.lax.map(f, (jnp.moveaxis(targets, -1, 0), rngs))

    return models, diagnostics


@partial(jax.jit, static_argnums=8)
def _unlearn(models: BinaryLogReg, data: Dataset, delete: jnp.ndarray, retain: jnp.ndarray, rng: jnp.ndarray,
             tolerance: float, max_iterations: int, prev_gram_matrix: Optional[jnp.ndarray],
             use_full_data_hess_approx: bool) -> Tuple[BinaryLogReg, float, jnp.ndarray]:
    """Implementation of the `unlearn` method for MultiLogReg"""
    inputs, targets = data

    # If Gram matrix not supplied, compute it once for all models
    if prev_gram_matrix is None:
        prev_gram_matrix = binary_logreg._init_gram_matrix(inputs, jnp.logical_or(retain, delete), add_intercept=True)

    def f(x):
        model, targets, rng = x
        return BinaryLogReg.unlearn(model, (inputs, targets), delete, retain, rng, 
                                    tolerance=tolerance, 
                                    max_iterations=max_iterations, 
                                    prev_gram_matrix=prev_gram_matrix, 
                                    enforce_grnb_constraint=False,  # enforce later
                                    use_full_data_hess_approx=use_full_data_hess_approx, 
                                    warn_retrain=False)
    
    rngs = random.split(rng, jnp.shape(models.lamb)[0])
    models, _, gram_matrix, _ = jax.lax.map(f, (models, jnp.moveaxis(targets, -1, 0), rngs))

    return models, jnp.sum(models.grnb), gram_matrix[0]


@jax.jit
def _objective(models: BinaryLogReg, data: Dataset, data_weights: Optional[jnp.ndarray]) -> float:
    """Vectorized implementation of the `objective` method for MultiLogReg"""
    objective_vmap = jax.vmap(BinaryLogReg.objective, in_axes=(0, (None, 1), None), out_axes=0)
    o_vals = objective_vmap(models, data, data_weights)
    return jnp.sum(o_vals)


@partial(jax.jit, static_argnums=3)
def _gradients(models: BinaryLogReg, data: Dataset, data_weights: Optional[jnp.ndarray],
               perturbation_term: bool) -> jnp.ndarray:
    """Vectorized implementation of the `gradients` method for MultiLogReg"""
    gradient_vmap = jax.vmap(BinaryLogReg.gradient, in_axes=(0, (None, 1), None, None), out_axes=0)
    return gradient_vmap(models, data, data_weights, perturbation_term)


@jax.jit
def _hessians(models: BinaryLogReg, data: Dataset,
              data_weights: Optional[jnp.ndarray]) -> jnp.ndarray:
    """Implementation of the `hessians` method for MultiLogReg"""
    def f(x):
        model, targets = x
        return BinaryLogReg.hessian(model, (data[0], targets), data_weights=data_weights)
    return jax.lax.map(f, (models, jnp.moveaxis(data[1], -1, 0)))


def register_pytree_node_dataclass(cls):
    def _flatten(obj): return jax.tree_flatten(dataclasses.asdict(obj))

    def _unflatten(d, children):
        class_dict = d.unflatten(children)
        models_dict = class_dict['models']
        class_dict['models'] = None if models_dict is None else BinaryLogReg(**models_dict)
        return cls(**class_dict)

    jax.tree_util.register_pytree_node(cls, _flatten, _unflatten)

    return cls


@register_pytree_node_dataclass
@dataclass(frozen=True)
class MultiLogReg(EnsembleCRModel):
    """Multi-class logistic regression model with support for certified removal

    This class implements one-vs-rest multi-class logistic regression. It maintains an ensemble of BinaryLogReg
    models.

    Attributes:
        classes: array of labels used to encode the classes.
        lamb: L2 regularization strength.
        epsilon: parameter for the certified removal guarantee, typically on the unit interval.
        delta: parameter for the certified removal guarantee, typically of order 1/(training set size).
        sigma: standard deviation of the random parameters.
        models: tuple containing BinaryLogReg models in the ensemble.
        grnb: cumulative gradient residual norm bound.
        grnb_thres: threshold on `grnb` above which full retraining is triggered.
    """
    classes: jnp.ndarray = jnp.empty(0, dtype=int)
    lamb: float = 1.0
    epsilon: float = 1.0
    delta: float = 1e-4
    sigma: float = 1.0
    models: Optional[BinaryLogReg] = None
    grnb: float = 0.0

    def __post_init__(self: Model):
        # Ensure classes is specified
        if jnp.size(self.classes) == 0:
            raise ValueError("`classes` must a non-empty array")

    @property
    def grnb_thres(self: Model) -> float:
        return self.sigma * self.epsilon / jnp.sqrt(2 * jnp.log(1.5 / self.delta))

    def decision_function(self: Model, inputs: jnp.ndarray) -> jnp.ndarray:
        """Predict confidence scores

        The confidence score for a sample is proportional to the signed distance of that sample to the hyperplane.

        Args:
            inputs: sample inputs to score. Must be a 2d array of shape (n_samples, n_features).

        Returns:
            array of shape (n_samples,) containing the confidence scores (for the positive class) for each sample.
            A positive score means the positive class would be predicted.
        """
        assert self.models is not None, "model is not fitted"
        return _decision_function(self.models, inputs)

    def predict(self: Model, inputs: jnp.ndarray) -> jnp.ndarray:
        """Predict class labels

        Args:
            inputs: samples inputs on which to make predictions. Must be a 2d array of shape (n_samples, n_features).

        Returns:
            array of shape (n_samples,) containing the predicted class labels for each sample.
        """
        scores = self.decision_function(inputs)
        if scores.shape[1] == 1:
            indices = (scores > 0).astype(int)
        else:
            indices = scores.argmax(axis=1)
        if self.classes is not None:
            return jnp.asarray(self.classes)[indices]
        else:
            return indices

    def predict_proba(self: Model, inputs: jnp.ndarray) -> jnp.ndarray:
        """Probability estimates

        Args:
            inputs: samples inputs on which to estimate predicted probabilities. Must be a 2d array of shape
                (n_samples, n_features).

        Returns:
            array of shape (n_samples,) containing the predicted probabilities that each sample is in the positive
            class.
        """
        scores = self.decision_function(inputs)
        prob = sigmoid(scores)
        if prob.ndim == 1:
            return jnp.hstack((1 - prob, prob))
        else:
            # OvR normalization, like LibLinear's predict_probability
            prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
            return prob

    def fit(self: Model, data: Dataset, rng: jnp.ndarray, data_weights: Optional[jnp.ndarray] = None,
            tolerance: float = 1e-4, max_iterations: int = 100) -> Tuple[Model, Dict]:
        """Fit the model

        Args:
            data: training data represented as a tuple, where the first element is the inputs and the second element
                is the targets (labels).
            rng: a PRNGKey (an array with shape (2,) and dtype uint32).
            data_weights: non-negative weights associated with the samples in `data`.
            tolerance: gradient tolerance for L-BFGS. If the sup-norm of the gradient vector is below this value,
                L-BFGS is terminated.
            max_iterations: maximum number of iterations before L-BFGS is terminated.

        Returns:
            a tuple where the first element is the fitted model, and the second element contains diagnostics
            associated with L-BFGS.
        """
        inputs, targets = data
        
        targets = _one_hot_encode(targets, self.classes)
        
        data = inputs, targets
        
        model = BinaryLogReg(epsilon=self.epsilon, delta=self.delta, lamb=self.lamb, sigma=self.sigma)
        models, diagnostics = _fit(model, data, rng, data_weights, tolerance, max_iterations)

        # Update model
        model_dict = dataclasses.asdict(self)
        model_dict['models'] = models
        model_dict['grnb'] = 0.0  # reset GRNB
        model = self.__class__(**model_dict)

        return model, diagnostics

    def unlearn(self: Model, data: Dataset, delete: jnp.ndarray, retain: jnp.ndarray, rng: jnp.ndarray,
                tolerance: float = 1e-4, max_iterations: int = 100, prev_gram_matrix: Optional[jnp.ndarray] = None,
                enforce_grnb_constraint: bool = True, use_full_data_hess_approx: bool = True,
                warn_retrain: bool = False) -> Tuple[Model, Optional[Dict], Optional[jnp.ndarray], bool]:
        """Unlearn training data

        Args:
            data: training data represented as a tuple, where the first element is the inputs and the second element
                is the targets (labels).
            delete: boolean array specifying samples in data to unlearn/delete.
            retain: boolean array specifying samples in data to keep post-deletion.
            rng: a PRNGKey (an array with shape (2,) and dtype uint32).
            tolerance: gradient tolerance for L-BFGS. If the sup-norm of the gradient vector is below this value,
                L-BFGS is terminated.
            max_iterations: maximum number of iterations before L-BFGS is terminated.
            prev_gram_matrix: gram matrix for the training data inputs prior to unlearning. If not provided, it is
                computed from scratch.
            enforce_grnb_constraint: boolean specifying whether to retrain from scratch if the cumulative gradient
                residual norm exceeds the bound that guarantees (epsilon, delta)-CR.
            use_full_data_hess_approx: boolean specifying whether to use the full training data (prior to deletion)
                to approximate the Hessian. If False, only the retained data (post-deletion) is used to approximate
                the Hessian as in Guo et al. (2020).
            warn_retrain: boolean specifying whether to emit a warning if retraining from scratch is triggered.

        Returns:
            a tuple where the first element is the unlearned model, the second element contains diagnostics associated
            with L-BFGS for retraining, the third element is the updated gram matrix, and the fourth element
            is a boolean indicating whether retraining was triggered.
        """
        assert self.models is not None, "model is not fitted"
        
        inputs, targets = data
        targets = _one_hot_encode(targets, self.classes)
        data = inputs, targets

        models, grnb, gram_matrix = _unlearn(self.models, data, delete, retain, rng, tolerance, max_iterations,
                                             prev_gram_matrix, use_full_data_hess_approx)

        def retrain(_):
            # Retrain from scratch
            if warn_retrain:
                warnings.warn("GRNB of {} exceeds upper bound of {}. "
                              "Retraining from scratch".format(grnb, self.grnb_thres))
            model_new, diagnostics = self.fit(data, rng, data_weights=retain, tolerance=tolerance,
                                              max_iterations=max_iterations)
            return model_new, diagnostics, gram_matrix, True
        
        def done(_):
            # Update model
            model_dict = dataclasses.asdict(self)
            model_dict['models'] = models
            model_dict['grnb'] = grnb
            n_models = jnp.shape(models.lamb)[0]
            model_new = self.__class__(**model_dict)
            diagnostics = {'converged': jnp.zeros(n_models, dtype=bool),
                           'num_iterations': jnp.zeros(n_models, dtype=jnp.int32),
                           'grad_norm': jnp.zeros(n_models, dtype=jnp.float64)}
            return model_new, diagnostics, gram_matrix, False
        
        return jax.lax.cond(
            jnp.any(grnb > self.grnb_thres) & enforce_grnb_constraint,
            retrain,
            done,
            None
        )
    
    def set_sigma(self: Model, value: float) -> Model:
        """Set sigma parameter
        
        Args:
            value: new value of sigma
        
        Returns:
            MultiLogReg: new model with sigma parameter updated
        """
        return dataclasses.replace(self, sigma=value, models=None, grnb=0.0)

    def objective(self: Model, data: Dataset, data_weights: Optional[jnp.ndarray] = None) -> float:
        """Compute the objective on the given training data"""
        assert self.models is not None, "model is not fitted"

        inputs, targets = data
        targets = _one_hot_encode(targets, self.classes)
        data = inputs, targets

        return _objective(self.models, data, data_weights)

    def gradients(self: Model, data: Dataset, data_weights: Optional[jnp.ndarray] = None,
                  perturbation_term: bool = True) -> jnp.ndarray:
        """Compute the (ravelled) gradient of the objective with respect to the model parameters on the given
        training data"""
        assert self.models is not None, "model is not fitted"

        inputs, targets = data
        targets = _one_hot_encode(targets, self.classes)
        data = inputs, targets

        return _gradients(self.models, data, data_weights, perturbation_term)

    def hessians(self: Model, data: Dataset, data_weights: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Compute the (ravelled) Hessian of the objective with respect to the model parameters on the given training
        data"""
        assert self.models is not None, "model is not fitted"

        inputs, targets = data
        targets = _one_hot_encode(targets, self.classes)
        data = inputs, targets

        return _hessians(self.models, data, data_weights)
