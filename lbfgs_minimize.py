from jax.flatten_util import ravel_pytree
from tensorflow_probability.substrates.jax.optimizer import lbfgs_minimize as tfp_lbfgs_minimize
from collections import namedtuple

LBfgsOptimizerResults = namedtuple('LBfgsOptimizerResults', 
                                   ['converged', 'failed', 'num_objective_evaluations', 
                                    'params', 'objective_value', 'objective_gradient', 
                                    'params_deltas', 'gradient_deltas'])

def lbfgs_minimize(
    value_and_gradients_function,
    initial_params,
    previous_optimizer_results=None,
    num_correction_pairs=10,
    tolerance=1e-08,
    x_tolerance=0,
    f_relative_tolerance=0,
    initial_inverse_hessian_estimate=None,
    max_iterations=50,
    parallel_iterations=1,
    stopping_condition=None,
    max_line_search_iterations=50,
    name=None):
    
    # Use tree flatten and unflatten to convert params initial_position from PyTrees to flat arrays
    initial_position, unravel = ravel_pytree(initial_params)
    
    # Wrap the objective function to consume flat _original_ 
    # numpy arrays and produce scalar outputs.
    def value_and_gradients_function_wrapper(initial_position):
        initial_params = unravel(initial_position)
        value, grads = value_and_gradients_function(initial_params)
        return value, ravel_pytree(grads)[0]
    
    results = tfp_lbfgs_minimize(value_and_gradients_function_wrapper, 
                                 initial_position,
                                 previous_optimizer_results=previous_optimizer_results,
                                 num_correction_pairs=num_correction_pairs,
                                 tolerance=tolerance,
                                 x_tolerance=x_tolerance,
                                 f_relative_tolerance=f_relative_tolerance,
                                 initial_inverse_hessian_estimate=initial_inverse_hessian_estimate,
                                 max_iterations=max_iterations,
                                 parallel_iterations=parallel_iterations,
                                 stopping_condition=stopping_condition,
                                 max_line_search_iterations=max_line_search_iterations,
                                 name=name)
    
    # pack the output back into a PyTree
    return LBfgsOptimizerResults(converged = results.converged,
                                 failed = results.failed,
                                 num_objective_evaluations = results.num_objective_evaluations,
                                 params = unravel(results.position),
                                 objective_value = results.objective_value,
                                 objective_gradient = unravel(results.objective_gradient),
                                 params_deltas = results.position_deltas,
                                 gradient_deltas = results.gradient_deltas)