from dataclasses import dataclass
import jax.numpy as jnp

from typing import Tuple, Optional, Dict, Sequence, TypeVar

Dataset = Tuple[jnp.ndarray, jnp.ndarray]
Model = TypeVar('Model')


@dataclass(eq=True, frozen=True)
class CRModel:
    """Base class for a model that supports (ε,δ)-certified removal (CR)
    """
    epsilon: float
    delta: float

    def predict(self: Model, inputs: jnp.ndarray) -> jnp.ndarray:
        """Predict class labels

        Args:
            inputs: samples inputs on which to make predictions. Must be a 2d array of shape (n_samples, n_features).

        Returns:
            array of shape (n_samples,) containing the predicted class labels for each sample.
        """
        pass

    def fit(self: Model, data: Dataset, rng: jnp.ndarray, data_weights: Optional[jnp.ndarray] = None) -> \
            Tuple[Model, Dict]:
        """Fit the model

        Args:
            data: training data represented as a tuple, where the first element is the inputs and the second element
                is the targets (labels).
            rng: a PRNGKey (an array with shape (2,) and dtype uint32).
            data_weights: non-negative weights associated with the samples in `data`.

        Returns:
            a tuple where the first element is the fitted model, and the second element contains diagnostics.
        """
        pass

    def unlearn(self, data: Dataset, delete: jnp.ndarray, retain: jnp.ndarray,
                rng: jnp.ndarray) -> Tuple[Model, Optional[Dict], Optional[jnp.ndarray], bool]:
        """Unlearn training data

        Args:
            data: training data represented as a tuple, where the first element is the inputs and the second element
                is the targets (labels).
            delete: boolean array specifying samples in data to unlearn/delete.
            retain: boolean array specifying samples in data to keep post-deletion.
            rng: a PRNGKey (an array with shape (2,) and dtype uint32).

        Returns:
            a tuple where the first element is the unlearned model, the second element contains diagnostics for
            retraining, the third element is the updated gram matrix, and the fourth element is a boolean indicating
            whether retraining was triggered.
        """
        pass


@dataclass(eq=True, frozen=True)
class EnsembleCRModel(CRModel):
    """Base class for an ensemble of models which supports (ε,δ)-certified removal (CR)
    """
    models: Optional[Sequence[CRModel]]