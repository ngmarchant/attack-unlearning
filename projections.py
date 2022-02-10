import jax.numpy as jnp
import jax
from functools import partial

from typing import Callable, Optional, Union, Sequence


def soft_thresh(lamda: Union[float, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
    r"""Soft thresholding operator

    Performs:

    .. math::
        (| x | - \lambda)_+  \mathrm{sgn}(x)

    Args:
        lamda (float or array): Threshold parameter.
        x (array): Input array.
    """
    sign = jnp.sign(x)
    mag = jnp.maximum(jnp.abs(x) - lamda, 0)
    return sign * mag


@partial(jax.jit, static_argnums=0)
def linf_proj(eps: float, x: jnp.ndarray, 
              x_ref: Optional[jnp.ndarray] = None, 
              **kwargs) -> jnp.ndarray:
    """Projection onto L-inf ball
    
    Args:
        eps (float or array): L-inf ball scaling.
        x (array): Input array.
        x_ref (array, optional): Centre of the L-inf ball. Defaults to zero.
    
    References:
        https://math.stackexchange.com/questions/1825747/orthogonal-projection-onto-the-l-infty-unit-ball
    """
    if x_ref is not None:
        x = x - x_ref

    output = jnp.clip(x, a_min=-eps, a_max=eps)

    if x_ref is not None:
        output += x_ref
    
    return output


@partial(jax.jit, static_argnums=(0, 3))
def _l1_proj(eps: float, x: jnp.ndarray, x_ref: Optional[jnp.ndarray], 
             axis: Optional[int]):
    x_shape = x.shape
    x = x.ravel() if axis is None else x
    
    if x_ref is not None:
        x_ref = x_ref.ravel() if axis is None else x_ref
        x = x - x_ref
    
    n = jnp.size(x, axis=axis)
    abs_x = jnp.abs(x)
    s = jnp.flip(jnp.sort(abs_x, axis=axis), axis=axis)
    st = (jnp.cumsum(s, axis=axis) - eps) / (jnp.arange(n) + 1)
    diff = s - st
    idx = jnp.where(diff > 0, jnp.arange(n), 0).max(axis=axis, keepdims=True)
    proj = soft_thresh(jnp.take_along_axis(st, idx, axis=axis), x)
    
    x = jnp.where(jnp.linalg.norm(x, ord=1, axis=axis, keepdims=True) > eps, 
                  proj,
                  x)
    
    if x_ref is not None:
        x += x_ref
    
    x = x.reshape(x_shape)
    
    return x


def l1_proj(eps: float, x: jnp.ndarray, 
            x_ref: Optional[jnp.ndarray] = None, 
            axis: Optional[int] = None, **kwargs) -> jnp.ndarray:
    """Projection onto L1 ball.

    Args:
        eps (float or array): L1 ball scaling.
        x (array): Input array.
        x_ref (array, optional): Centre of the L1 ball. Defaults to zero.
        axis (int, optional): Axis along which to apply the projection. 
            Defaults to None, in which case the projection is applied to the 
            flattened array.

    References:
        J. Duchi, S. Shalev-Shwartz, and Y. Singer, "Efficient projections onto
        the l1-ball for learning in high dimensions" 2008.
    """
    return _l1_proj(eps, x, x_ref, axis)


@partial(jax.jit, static_argnums=(0, 3))
def _l2_proj(eps: float, x: jnp.ndarray, x_ref: Optional[jnp.ndarray], 
             axis: Optional[int]):
    x_shape = x.shape
    x = x.ravel() if axis is None else x
    
    if x_ref is not None:
        x_ref = x_ref.ravel() if axis is None else x_ref
        x = x - x_ref
    
    norm = jnp.linalg.norm(x, ord=2, axis=axis, keepdims=True)
    output = jnp.minimum(1, eps/norm) * x
    
    if x_ref is not None:
        output += x_ref
    
    output = output.reshape(x_shape)
    
    return output


def l2_proj(eps: float, x: jnp.ndarray, 
            x_ref: Optional[jnp.ndarray] = None, 
            axis: Optional[int] = None, **kwargs) -> jnp.ndarray:
    """Projection onto L2 ball.

    Args:
        eps (float, or array): L2 ball scaling.
        x (array): Input array.
        x_ref (array, optional): Centre of the L2 ball. Defaults to zero.
        axis (int, optional): Axis along which to apply the projection. 
            Defaults to None, in which case the projection is applied to the 
            flattened array.
    """
    return _l2_proj(eps, x, x_ref, axis)


# Adapted from https://github.com/mjhough/Dykstra/blob/main/dykstra/Dykstra.py
@partial(jax.jit, static_argnums=(0,))
def dykstra_proj(proj_ops: Sequence[Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]],
                 x: jnp.ndarray, x_ref: Optional[jnp.ndarray] = None, 
                 max_iter: int = 50, 
                 tol: float = 1e-6) -> jnp.ndarray:
    """Dykstra's algorithm for projection onto the intersection of convex sets
    
    Args:
        proj_ops (tuple of callables): a tuple containing projection operators for each convex set in the intersection.
        x: point to project onto the intersection of convex sets.
        x_ref: optional reference point to pass to proj_ops.
        max_iter (int): maximum number of iterations to perform.
        tol (float): tolerance for stopping condition. See c_I^k in
            https://www.ime.usp.br/~egbirgin/publications/br.pdf
    
    Returns:
        projection of x onto the intersection of convex sets
    
    References:
        Ernesto G. Birgin and Marcos Raydan. 2005. Robust Stopping Criteria for Dykstra's Algorithm.
        SIAM J. Sci. Comput. 26, 4 (2005), 1405–1414. DOI: https://doi.org/10.1137/03060062X
    """
    num_proj_ops = len(proj_ops)
    y = [jnp.zeros_like(x) for _ in range(num_proj_ops)]
    k = 0
    cI = float('inf')
    
    def cond_fun(val) -> bool:
        k, cI, _, _ = val
        # return k < max_iter & cI >= tol
        return jax.lax.lt(k, max_iter) & jax.lax.ge(cI, tol)
        
    def body_fun(val):
        k, _, y, x = val
        cI = 0.0
        for i in range(num_proj_ops):
            prev_y = y[i]
            
            # Update iterate
            z = x - prev_y
            x = proj_ops[i](z, x_ref)

            # Update increment
            y[i] = x - z

            # Update stop condition
            cI += jnp.sum((prev_y - y[i])**2)
        return k + 1, cI, y, x
        
    k, cI, y, x = jax.lax.while_loop(cond_fun, body_fun, (k, cI, y, x))

    return x
