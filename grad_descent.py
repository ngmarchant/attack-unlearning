import jax.numpy as jnp
import numpy as onp

from typing import Callable, Tuple, Union


def proj_grad_descent(obj_value_and_grad: Callable[[jnp.ndarray], Tuple[float, jnp.ndarray]], x_init: jnp.ndarray,
                      step_size: float, proj_op: Callable[[jnp.ndarray], jnp.ndarray], num_iter: int = 10, 
                      ord: Union[str, int, None, float] = 2, axis: Union[int, None, Tuple[int, int]] = None, 
                      armijo: bool = True, tau: float = 0.5, c: float = 0.5) -> onp.ndarray:
    """Run projected gradient descent
    
    Args:
        obj_value_and_grad: objective function which returns the value and gradient.
        x_init: starting value of the parameters to optimize.
        step_size: step size (positive float).
        proj_op: function that projects parameters onto the feasible set.
        num_iter: number of iterations to perform. Defaults to 10.
        ord: order of the gradient norm used to normalize the gradient. See documentation in `jnp.linalg.norm`.
        axis: axis along which to compute the gradient norm. See documentation in `jnp.linalg.norm`.
        armijo: whether to perform an Armijo backtracking line search to determine a good step size.
        tau: Armijo parameter.
        c: Armijo parameter.
        
    Returns:
        final value of the parameters after stopping condition is triggered.
    """
    
    x = x_init
    
    def step_direction(x, g):
        # if p == 1:
        #     abs_g = jnp.abs(g)
        #     return -(abs_g == abs_g.max()) * jnp.sign(g)
        # elif p == 2:
        #     g_norm_2 = jnp.linalg.norm(g.ravel(), ord=2)
        #     return -g / g_norm_2
        # elif p == jnp.inf or p == 'inf':
        #     return -jnp.sign(g)
        # else:
        #     raise ValueError("p = {} not supported".format(p))
        g_norm = jnp.linalg.norm(g, ord=ord, axis=axis, keepdims=True)
        return - g / g_norm # negative for gradient descent

    def search_step_size(x, direction, g):
        max_iter = 100
        g_dot_direction = jnp.sum(g * direction, axis=None)
        t = - c * g_dot_direction
        a_j = step_size
        obj_x, _ = obj_value_and_grad(x)
        
        def cond_fun(val):
            j, a_j = val
            obj_x_p, _ = obj_value_and_grad(x + a_j * direction)
            armijo_cond = (obj_x - obj_x_p) < a_j * t
            return armijo_cond & (j < max_iter)
        
        def body_fun(val):
            j, a_j = val
            return j + 1, tau * a_j
        
        val = 0, a_j
        # val = jax.lax.while_loop(cond_fun, body_fun, val)
        while cond_fun(val):
            val = body_fun(val)
        
        return val[1]
        
    for i in range(1, num_iter + 1):        
        # Take steepest descent step
        v, g = obj_value_and_grad(x)
        direction = step_direction(x, g)
        if armijo:
            eta = search_step_size(x, direction, g)
        else:
            eta = step_size

        x = x + eta * direction
        
        # Project
        x = proj_op(x)
    
    return onp.asarray(x)
