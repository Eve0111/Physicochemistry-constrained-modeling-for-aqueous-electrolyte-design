from typing import Callable, Dict, List, Optional, Tuple, Union
from botorch.optim.parameter_constraints import ScipyConstraintDict, _validate_linear_constraints_shape_input
import torch
from torch import Tensor


def _make_f_nonlinear_inequality_constraints(
    f_np_wrapper: Callable, nlc: Callable
) -> Callable[[Tensor], Tensor]:
    """
    Create callables for objective + grad for the nonlinear inequality constraints.
    The Scipy interface requires specifying separate callables and we use caching to
    avoid evaluating the same input twice. This caching only works if
    the returned functions are evaluated on the same input in immediate
    sequence (i.e., calling `f_obj(X_1)`, `f_grad(X_1)` will result in a
    single forward pass, while `f_obj(X_1)`, `f_grad(X_2)`, `f_obj(X_1)`
    will result in three forward passes).
    """

    def f_obj(x):
        obj = f_np_wrapper(x, f=nlc)
        return obj

    return f_obj


def _make_nonlinear_constraints_no_grad_patch(
    f_np_wrapper: Callable, nlc: Callable, is_intrapoint: bool, shapeX: torch.Size
) -> List[ScipyConstraintDict]:
    """Create nonlinear constraints to be used by `scipy.minimize`.

    Args:
        f_np_wrapper: A wrapper function that given a constraint evaluates
            the value and gradient (using autograd) of a numpy input and returns both
            the objective and the gradient.
        nlc: Callable representing a constraint of the form `callable(x) >= 0`. In case
            of an intra-point constraint, `callable()`takes in an one-dimensional tensor
            of shape `d` and returns a scalar. In case of an inter-point constraint,
            `callable()` takes a two dimensional tensor of shape `q x d` and again
            returns a scalar.
        is_intrapoint: A Boolean indicating if a constraint is an intra-point or
            inter-point constraint (see the docstring of the `inequality_constraints`
            argument to `optimize_acqf()`).
        shapeX: Shape of the three-dimensional batch X, that should be optimized.

    Returns:
        A list of constraint dictionaries with the following keys

        - "type": Indicates the type of the constraint, here always "ineq".
        - "fun": A callable evaluating the constraint value on `x`, a flattened
            version of the input tensor `X`, returning a scalar.
        - "jac": A callable evaluating the constraint's Jacobian on `x`, a flattened
            version of the input tensor `X`, returning a numpy array.
    """
    shapeX = _validate_linear_constraints_shape_input(shapeX)
    b, q, _ = shapeX
    constraints = []

    def get_intrapoint_constraint(b: int, q: int, nlc: Callable) -> Callable:
        return lambda x: nlc(x[b, q])

    def get_interpoint_constraint(b: int, nlc: Callable) -> Callable:
        return lambda x: nlc(x[b])

    if is_intrapoint:
        for i in range(b):
            for j in range(q):
                f_obj = _make_f_nonlinear_inequality_constraints(
                    f_np_wrapper=f_np_wrapper,
                    nlc=get_intrapoint_constraint(b=i, q=j, nlc=nlc),
                )
                constraints.append({"type": "ineq", "fun": f_obj})
    else:
        for i in range(b):
            f_obj = _make_f_nonlinear_inequality_constraints(
                f_np_wrapper=f_np_wrapper,
                nlc=get_interpoint_constraint(b=i, nlc=nlc),
            )
            constraints.append({"type": "ineq", "fun": f_obj})

    return constraints


def patch_botorch_nonlinear_constraints_no_grad() -> None:
    import botorch
    botorch.optim.parameter_constraints._make_nonlinear_constraints = _make_nonlinear_constraints_no_grad_patch
