from cr_model import CRModel
import warnings

import jax.numpy as jnp
import jax
from jax.numpy.linalg import norm
from jax import jit, value_and_grad, random, custom_jvp
from jax.experimental import stax
from jax.experimental.stax import Dense
from jax.nn import log_sigmoid, sigmoid
from jax.nn.initializers import normal, zeros
from jax.flatten_util import ravel_pytree

from tensorflow_probability.substrates.jax.optimizer import lbfgs_minimize

from util import register_pytree_node_dataclass, tree_zeros_like
from dataclasses import dataclass
import dataclasses
from typing import Dict, Union, Any, Optional, Tuple

from functools import partial

Dataset = Tuple[jnp.ndarray, jnp.ndarray]
def _zeros_64(key, shape): return zeros(key, shape, dtype=jnp.dtype('float64'))


_init_params, _decision_function = stax.serial(
            Dense(1, W_init=_zeros_64, b_init=_zeros_64)
        )


def _init_random_params(rng: jnp.ndarray, shape: Any, sigma: float):
    """Initializer for randomized coefficients of the linear perturbation term"""
    params = []
    r1, r2 = random.split(rng)
    norm_init = normal(stddev=sigma, dtype=jnp.dtype('float64'))
    coef = norm_init(r1, (shape[-1], 1))
    intercept = norm_init(r2, (1,))
    param = coef, intercept
    params.append(param)
    return shape, params


def _init_data_weights(data: Dataset, data_weights: Optional[jnp.ndarray]) -> jnp.ndarray:
    """Helper function that returns initialized sample weights. If not specified, they are initialized to one."""
    if data_weights is None:
        return jnp.ones(data[0].shape[0], dtype=data[0].dtype)
    else:
        return data_weights


@partial(jit, static_argnums=(2,))
def _init_gram_matrix(inputs: jnp.ndarray, weights: jnp.ndarray, add_intercept: bool = True):
    """Compute the gram matrix for the inputs

    Args:
        inputs: 2d array of shape (n_samples, n_features) which contains the input features.
        weights: 1d array of shape (n_samples,) which contains non-negative weights for each sample.
        add_intercept: whether to append an extra constant feature to incorporate an intercept. Defaults to True.

    Returns:
        a 2d array of shape (n_features, n_features) if add_intercept is False, otherwise a 2d array of shape
        (n_features + 1, n_features + 1).
    """
    inputs_weighted = inputs * weights[:, None]
    gram_matrix = inputs_weighted.T @ inputs_weighted

    # Account for intercept
    if add_intercept:
        inputs_colsums = jnp.sum(inputs_weighted, axis=0)
        gram_matrix = jnp.block([[gram_matrix, inputs_colsums.reshape(-1, 1)],
                                 [inputs_colsums, jnp.sum(weights)]])

    return gram_matrix


@jit
def _objective(params: Any, random_params: Any, data: Dataset, lamb: float, pos_label: Union[str, int, float],
               data_weights: Optional[jnp.ndarray]) -> float:
    """Objective function for L2-regularized logistic regression. Includes a randomized perturbation term to
    improve privacy."""
    data_weights = _init_data_weights(data, data_weights)

    inputs = jnp.atleast_2d(data[0])
    targets = jnp.atleast_1d(data[1])
    targets = targets.ravel()  # ensure targets is a 1d vector
    scores = _decision_function(params, inputs).ravel()
    cross_entropy_terms = jnp.where(targets == pos_label, log_sigmoid(scores), log_sigmoid(-scores))
    cross_entropy_loss = -jnp.dot(data_weights, cross_entropy_terms)
    
    # Regularization penalty term
    coef, intercept = params[0]
    # Guo et al. (2020) penalize the coefficients *and* the bias
    l2_penalty = 0.5 * lamb * jnp.sum(data_weights) * (jnp.sum(coef**2) + jnp.sum(intercept**2))

    # Random linear term for indistinguishability/privacy
    b_coef, b_intercept = random_params[0]
    # Note: this term is divided by data_size in
    # https://github.com/facebookresearch/certified-removal/blob/2a3aa66e85b95d659830f944f644daaf41f63167/test_removal.py#L109
    rand_linear = (b_coef * coef).sum() + (b_intercept * intercept).sum()
    
    return cross_entropy_loss + l2_penalty + rand_linear


@partial(jit, static_argnums=3)
def _gradient(model: 'BinaryLogReg', data: Dataset, data_weights: Optional[jnp.ndarray],
              perturbation_term: bool) -> jnp.ndarray:
    """Ravelled gradient of the objective function with respect to the model parameters"""
    params_flat, unravel = ravel_pytree(model.params)
    random_params = model.random_params
    if not perturbation_term:
        random_params = tree_zeros_like(random_params)
    g = jax.grad(lambda p: _objective(unravel(p), random_params, data, model.lamb, model.pos_label, data_weights))
    return g(params_flat)


@jit
def _hessian(model: 'BinaryLogReg', data: Dataset, data_weights: Optional[jnp.ndarray]) -> jnp.ndarray:
    """Ravelled Hessian matrix of the objective function with respect to the model parameters"""
    params_flat, unravel = ravel_pytree(model.params)
    random_params = model.random_params
    h = jax.hessian(lambda p: _objective(unravel(p), random_params, data, model.lamb, model.pos_label, data_weights))
    return h(params_flat)


@custom_jvp
def _fit(init_params: Any, random_params: Any, data: Dataset, lamb: float, pos_label: Union[str, int, float],
         data_weights: jnp.ndarray, tolerance: float, max_iterations: int) -> Tuple[Any, Dict[str, Any]]:
    """Implementation of the `fit` method for BinaryLogReg"""
    # L-BFGS implementation in TFP is not compatible with PyTrees, so need to 
    # ravel parameters. Call ravelled parameters "position".
    init_position, unravel = ravel_pytree(init_params)
    
    # Wrapper for objective function which accepts position as argument.
    def obj(position):
        params = unravel(position)
        return _objective(params, random_params, data, lamb, pos_label, data_weights)
    
    obj_and_grad = value_and_grad(obj)

    def solve(f, init_position):
        return lbfgs_minimize(f, init_position, tolerance=tolerance, max_iterations=max_iterations)
    
    result = solve(obj_and_grad, init_position)
    params = unravel(result.position)

    return params, {'converged': result.converged, 
                    'num_iterations': result.num_iterations, 
                    'grad_norm': jnp.linalg.norm(result.objective_gradient, ord=2)}


# Custom implementation of jvp based on implicit function theorem
@_fit.defjvp
def _fit_jvp(primals, tangents):
    init_params, random_params, data, lamb, pos_label, data_weights, tolerance, max_iterations = primals
    _, _, data_dot, _, _, _, _, _ = tangents
    
    params, diagnostics = _fit(init_params, random_params, data, lamb, pos_label, data_weights, tolerance,
                               max_iterations)
    
    inputs, targets = data
    inputs_dot, _ = data_dot

    scores = _decision_function(params, inputs).ravel()
    preds = sigmoid(scores)
    preds_var = preds * (1 - preds)
    scores_dot = _decision_function(params, inputs_dot).ravel()
    
    first_term = jnp.einsum('i,i,ij->j', scores_dot, preds_var, inputs)
    first_term = jnp.concatenate((jnp.atleast_1d(inputs.shape[0]), first_term))
    second_term = jnp.einsum('i,ij->j', preds - targets, inputs_dot)
    second_term = jnp.concatenate((jnp.atleast_1d(0), second_term))
    jac_X_grad = first_term + second_term
    
    params_flat, unravel = ravel_pytree(params)
    hess = jax.hessian(lambda p: _objective(unravel(p), random_params, data, lamb, pos_label, data_weights))(params_flat)
    hess_inv = jnp.linalg.inv(hess)
    
    # Pre-multiplying by the inverse Jacobian of the gradient 
    # of the loss w.r.t. w (i.e. the inverse Hessian) gives 
    # the required result 
    tangent_out = unravel(hess_inv @ jac_X_grad), {k: jnp.zeros_like(jnp.asarray(v)) for k, v in diagnostics.items()}
    
    primal_out = params, diagnostics
    
    return primal_out, tangent_out


_fit_jit = jax.jit(_fit)


@partial(jit, static_argnums=5)
def _unlearn(model: 'BinaryLogReg', data: Dataset, delete: jnp.ndarray, retain: jnp.ndarray,
             prev_gram_matrix: Optional[jnp.ndarray], use_full_data_hess_approx: bool) -> \
        Tuple[jnp.ndarray, float, jnp.ndarray]:
    """Implementation of the `unlearn` method for BinaryLogReg"""
    inputs, _ = data
    
    # Gradient contribution due to deleted data
    grad_del = BinaryLogReg.gradient(model, data, delete, False)
    
    # Correct error in the quadratic approximation used by Guo et al. (2020). Namely, the Hessian should be
    # approximated on the objective prior to removal.
    if use_full_data_hess_approx:
        retain = jnp.logical_or(delete, retain)
    
    # Update the Gram matrix.
    if prev_gram_matrix is None:
        # Compute for the first time
        gram_matrix = _init_gram_matrix(inputs, retain, add_intercept=True)
    else:
        gram_matrix = prev_gram_matrix
        if not use_full_data_hess_approx:
            delta_gram_matrix = _init_gram_matrix(inputs, delete, add_intercept=True)
            gram_matrix = gram_matrix - delta_gram_matrix
    
    # Although the Hessian can be expressed as a sum over training instances, we can't update it incrementally. This
    # is because the model parameters change at each iteration.
    hess = BinaryLogReg.hessian(model, data, retain)

    hess_inv_grad_del = jnp.linalg.inv(hess) @ grad_del
    inputs_x_hess_inv_grad_del = (inputs * retain[:, None]) @ hess_inv_grad_del[0:-1] + hess_inv_grad_del[-1]
    delta_grnb = (0.25 * jnp.sqrt(norm(gram_matrix, ord=2)) * norm(hess_inv_grad_del, ord=2) *
                  norm(inputs_x_hess_inv_grad_del, ord=2))
    
    # Easier to perform a Newton update using ravelled parameters
    params_flat, unravel = ravel_pytree(model.params)
    params = unravel(params_flat + hess_inv_grad_del)
    
    # Update GRNB
    grnb = model.grnb + delta_grnb

    return params, grnb, gram_matrix


@register_pytree_node_dataclass
@dataclass(frozen=True)
class BinaryLogReg(CRModel):
    """Logistic regression model with support for certified removal

    Attributes:
        lamb: L2 regularization strength.
        params: model parameters, represented as a tuple containing a vector of coefficients and an intercept.
        random_params: random parameters, represented as a tuple containing a vector of coefficients and an
            intercept.
        epsilon: parameter for the certified removal guarantee, typically on the unit interval.
        delta: parameter for the certified removal guarantee, typically of order 1/(training set size).
        sigma: standard deviation of the random parameters.
        pos_label: label used to encode positive instances.
        neg_label: label used to encode negative instances.
        grnb: cumulative gradient residual norm bound.
        grnb_thres: threshold on `grnb` above which full retraining is triggered.
    """
    lamb: float = 1.0
    params: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None
    random_params: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None
    epsilon: float = 1.0
    delta: float = 1e-4
    sigma: float = 1.0
    pos_label: Union[str, int, float] = 1
    neg_label: Union[str, int, float] = 0
    grnb: float = 0.0
    
    @property
    def grnb_thres(self) -> float:
        return self.sigma * self.epsilon / jnp.sqrt(2 * jnp.log(1.5 / self.delta))

    def decision_function(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Predict confidence scores

        The confidence score for a sample is proportional to the signed distance of that sample to the hyperplane.

        Args:
            inputs: sample inputs to score. Must be a 2d array of shape (n_samples, n_features).

        Returns:
            array of shape (n_samples,) containing the confidence scores (for the positive class) for each sample.
            A positive score means the positive class would be predicted.
        """
        assert self.params is not None, "model is not fitted"
        return _decision_function(self.params, inputs)

    def predict(self, inputs: jnp.ndarray) -> jnp.ndarray:
        scores = self.decision_function(inputs)
        binary = (scores > 0)
        return jnp.where(binary, self.pos_label, self.neg_label)

    def predict_proba(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Probability estimates

        Args:
            inputs: samples inputs on which to estimate predicted probabilities. Must be a 2d array of shape
                (n_samples, n_features).

        Returns:
            array of shape (n_samples,) containing the predicted probabilities that each sample is in the positive
            class.
        """
        scores = self.decision_function(inputs)
        return sigmoid(scores)

    def fit(self, data: Dataset, rng: jnp.ndarray, data_weights: Optional[jnp.ndarray] = None,
            tolerance: float = 1e-4, max_iterations: int = 1000) -> Tuple['BinaryLogReg', Dict]:
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
        data_weights = _init_data_weights(data, data_weights)

        inputs, _ = data
        
        # Update random parameters
        rng1, rng2 = random.split(rng)
        _, init_params = _init_params(rng1, (-1, inputs.shape[1]))
        _, random_params = _init_random_params(rng2, (-1, inputs.shape[1]), self.sigma)

        params, diagnostics = _fit_jit(init_params, random_params, data, self.lamb, self.pos_label,
                                       data_weights, tolerance, max_iterations)

        # Update model
        model_dict = dataclasses.asdict(self)
        model_dict['params'] = params
        model_dict['random_params'] = random_params
        model_dict['grnb'] = 0.0
        model = BinaryLogReg(**model_dict)

        return model, diagnostics

    def unlearn(self, data: Dataset, delete: jnp.ndarray, retain: jnp.ndarray, rng: jnp.ndarray,
                tolerance: float = 1e-4, max_iterations: int = 100, prev_gram_matrix: Optional[jnp.ndarray] = None,
                enforce_grnb_constraint: bool = True, use_full_data_hess_approx: bool = True,
                warn_retrain: bool = False) -> Tuple['BinaryLogReg', Optional[Dict], Optional[jnp.ndarray], bool]:
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
        assert self.params is not None, "model is not fitted"
        params, grnb, gram_matrix = _unlearn(self, data, delete, retain, prev_gram_matrix, use_full_data_hess_approx)
        
        def retrain(_):
            if warn_retrain:
                warnings.warn("GRNB of {} exceeds upper bound of {}. Retraining from "
                              "scratch".format(grnb, self.grnb_thres))
            model_new, diagnostics = BinaryLogReg.fit(self, data, rng, data_weights=retain, tolerance=tolerance,
                                                      max_iterations=max_iterations)
            return model_new, diagnostics, gram_matrix, True
    
        def done(_):
            # Update model
            model_dict = dataclasses.asdict(self)
            model_dict['params'] = params
            model_dict['grnb'] = grnb
            model_new = BinaryLogReg(**model_dict)
            diagnostics = {'converged': jnp.zeros((), dtype=bool), 'num_iterations': jnp.zeros((), dtype=jnp.int32),
                           'grad_norm': jnp.zeros((), dtype=jnp.float64)}
            return model_new, diagnostics, gram_matrix, False
        
        return jax.lax.cond(
            jnp.any(grnb > self.grnb_thres) & enforce_grnb_constraint,
            retrain,
            done,
            None
        )
    
    def set_sigma(self, value: float) -> 'BinaryLogReg':
        """Set sigma parameter
        
        Args:
            value: new value of sigma
        
        Returns:
            new model with sigma parameter updated
        """
        return dataclasses.replace(self, sigma=value, params=None, random_params=None, grnb=0.0)

    def objective(self, data: Dataset, data_weights: Optional[jnp.ndarray] = None) -> float:
        """Compute the objective on the given training data"""
        assert self.params is not None, "model is not fitted"
        return _objective(self.params, self.random_params, data, self.lamb, self.pos_label, data_weights)

    def gradient(self, data: Dataset, data_weights: Optional[jnp.ndarray] = None,
                 perturbation_term: bool = True) -> jnp.ndarray:
        """Compute the (ravelled) gradient of the objective with respect to the model parameters on the given
        training data"""
        assert self.params is not None, "model is not fitted"
        return _gradient(self, data, data_weights, perturbation_term)

    def hessian(self, data: Dataset, data_weights: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Compute the (ravelled) Hessian of the objective with respect to the model parameters on the given training
        data"""
        assert self.params is not None, "model is not fitted"
        return _hessian(self, data, data_weights)
