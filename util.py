import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import dataclasses


# Copied from https://github.com/google/jax/issues/2371#issuecomment-805361566
def register_pytree_node_dataclass(cls):
    _flatten = lambda obj: jax.tree_flatten(dataclasses.asdict(obj))
    _unflatten = lambda d, children: cls(**d.unflatten(children))
    jax.tree_util.register_pytree_node(cls, _flatten, _unflatten)
    return cls


def tree_zeros_like(tree):
    """Returns a tree with all values zeroed-out"""
    tree_flat, unravel = ravel_pytree(tree)
    tree_flat = jnp.zeros_like(tree_flat)
    return unravel(tree_flat)