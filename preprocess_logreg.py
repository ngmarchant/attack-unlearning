
from multi_logreg import MultiLogReg, register_pytree_node_dataclass

import jax.numpy as jnp
from jax import jit

from typing import Optional, Tuple, Dict, TypeVar
from dataclasses import dataclass
from functools import partial


Model = TypeVar('Model')
Dataset = Tuple[jnp.ndarray, jnp.ndarray]


@register_pytree_node_dataclass
@dataclass(eq=True, frozen=True)
class PreProcessLogReg(MultiLogReg):
    """Multi-class logistic regression model with a fixed pre-processing layer and support for certified removal

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

    def preprocess(self: Model, inputs: jnp.ndarray) -> jnp.ndarray:
        """Pre-processes the inputs before they are fed to the OvR multi-class logistic regression layer
        
        This method should be overriden"""
        return inputs
    
    @jit
    def decision_function(self: Model, inputs: jnp.ndarray) -> jnp.ndarray:
        processed_inputs = self.preprocess(inputs)
        return super().decision_function(processed_inputs)

    @jit
    def predict(self: Model, inputs: jnp.ndarray) -> jnp.ndarray:
        processed_inputs = self.preprocess(inputs)
        return super().predict(processed_inputs)

    def predict_proba(self: Model, inputs: jnp.ndarray):
        processed_inputs = self.preprocess(inputs)
        return super().predict_proba(processed_inputs)

    @jit
    def fit(self: Model, data: Dataset, rng: jnp.ndarray, data_weights: Optional[jnp.ndarray] = None,
            tolerance: float = 1e-4, max_iterations: int = 100) -> Tuple[Model, Dict]:
        inputs, targets = data
        processed_data = self.preprocess(inputs), targets
        return super().fit(
            processed_data, rng, 
            data_weights=data_weights,                
            tolerance=tolerance, 
            max_iterations=max_iterations
        )        

    @partial(jit, static_argnums=(8, 9, 10))
    def unlearn(self: Model, data: Dataset, delete: jnp.ndarray, retain: jnp.ndarray, rng: jnp.ndarray,
                tolerance: float = 1e-4, max_iterations: int = 100, prev_gram_matrix: Optional[jnp.ndarray] = None,
                enforce_grnb_constraint: bool = True, use_full_data_hess_approx: bool = True,
                warn_retrain: bool = False) -> Tuple[Model, Optional[Dict], Optional[jnp.ndarray], bool]:
        inputs, targets = data
        processed_data = self.preprocess(inputs), targets
        return super().unlearn(
            processed_data, delete, retain, rng, 
            tolerance=tolerance, 
            max_iterations=max_iterations, 
            prev_gram_matrix=prev_gram_matrix, 
            enforce_grnb_constraint=enforce_grnb_constraint, 
            use_full_data_hess_approx=use_full_data_hess_approx, 
            warn_retrain=warn_retrain
        )

    @jit
    def objective(self: Model, data: Dataset, data_weights: Optional[jnp.ndarray] = None) -> float:
        inputs, targets = data
        processed_data = self.preprocess(inputs), targets
        return super().objective(processed_data, data_weights=data_weights)
    
    @partial(jit, static_argnums=3)
    def gradients(self: Model, data: Dataset, data_weights: Optional[jnp.ndarray] = None,
                  perturbation_term: bool = True) -> jnp.ndarray:
        inputs, targets = data
        processed_data = self.preprocess(inputs), targets
        return super().gradients(processed_data, data_weights=data_weights, perturbation_term=perturbation_term)

    @jit
    def hessians(self: Model, data: Dataset, data_weights: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        inputs, targets = data
        processed_data = self.preprocess(inputs), targets
        return super().hessians(processed_data, data_weights=data_weights)
