# -*- coding: utf-8 -*-
"""
Sympy helpers
"""
from __future__ import absolute_import, division, print_function
try:
    import sympy
except ImportError:
    pass
#import numpy as np
import numpy as np
import six
import utool as ut
(print, rrr, profile) = ut.inject2(__name__, '[symb]')


def custom_sympy_attrs(mat):
    def matmul(other, hold=True):
        if hold:
            new = sympy.MatMul(mat, other)
        else:
            new = mat.multiply(other)
        custom_sympy_attrs(new)
        return new
    def inv_():
        new = mat.inv()
        custom_sympy_attrs(new)
        return new
    setattr(mat, 'matmul', matmul)
    setattr(mat, 'inv_', inv_)
    return mat


def sympy_mat(arr):
    mat = sympy.Matrix(arr)
    mat = custom_sympy_attrs(mat)
    return mat


def evalprint(str_, globals_=None, locals_=None, simplify=False):
    if globals_ is None:
        globals_ = ut.get_parent_globals()
    if locals_ is None:
        locals_ = ut.get_parent_locals()
    if isinstance(str_, six.string_types):
        var = eval(str_, globals_, locals_)
    else:
        var = str_
        str_ = ut.get_varname_from_stack(var, N=1)
    if simplify is True:
        var = sympy.simplify(var)
    print(ut.hz_str(str_ + ' = ', repr(var)))


def check_expr_eq(expr1, expr2, verbose=True):
    """
    Does not work in general. Problem is not decidable
    Thanks Richard.

    Args:
        expr1 (?):
        expr2 (?):

    CommandLine:
        python -m vtool.symbolic --test-check_expr_eq

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool.symbolic import *  # NOQA
        >>> expr1 = sympy.Matrix([ [sx*x + 1.0*tx + w1*y], [sy*y + 1.0*ty + w2*x], [1.0]])
        >>> expr2 = sympy.Matrix([ [sx*x + tx + w1*y], [sy*y + ty + w2*x], [1]])
        >>> result = check_expr_eq(expr1, expr2)
        >>> print(result)
    """
    print(ut.hz_str('Checking if ', repr(expr1), ' == ', repr(expr2)))
    random_point_check = expr1.equals(expr2)
    print('... seems %r' % (random_point_check,))
    #return random_point_check
    expr3 = expr1 - expr2
    print(ut.hz_str('... checking 0 ', repr(expr3)))
    # Does not always work.
    print('(not gaurenteed to work) expr3.is_zero = %r' % (expr3.is_zero,))
    return expr3.is_zero


def symbolic_randcheck(expr1, expr2, domain={}, n=10):
    def get_domain(key, domain={}, rng=np.random):
        min_, max_ = domain.get(key, (-100, 100))
        range_ = max_ - min_
        return (rng.rand() * (range_)) + min_
    num_checks = n
    input_list = []
    results_list = []
    for num in range(num_checks):
        expr1_subs = {key: get_domain(key, domain) for key in expr1.free_symbols}
        expr2_subs = {key: expr1_subs[key] if key in expr1_subs else get_domain(key, domain)
                      for key in expr2.free_symbols}
        expr1_value = expr1.evalf(subs=expr1_subs)
        expr2_value = expr2.evalf(subs=expr2_subs)
        input_list.append((expr1_subs, expr2_subs))
        results_list.append((expr1_value, expr2_value))
    results_list = np.array(results_list)
    #truth_list = np.allclose(results_list.T[0], results_list.T[1])
    truth_list = results_list.T[0] == results_list.T[1]
    return truth_list, results_list, input_list


def sympy_latex_repr(expr1):
    expr1_repr = sympy.latex(expr1)
    expr1_repr = expr1_repr.replace('\\\\', '\\\\\n')
    expr1_repr = expr1_repr.replace(r'\left[\begin{smallmatrix}{}', '\\MAT{\n')
    expr1_repr = expr1_repr.replace(r'\end{smallmatrix}\right]', '\n}')
    expr1_repr = expr1_repr.replace(r'\left[\begin{matrix}', '\\BIGMAT{\n')
    expr1_repr = expr1_repr.replace(r'\end{matrix}\right]', '\n}')
    expr1_repr = expr1_repr.replace(r'\left (', '(')
    expr1_repr = expr1_repr.replace(r'\right )', ')')
    # hack of align
    expr1_repr = ut.align(expr1_repr, '&', pos=None)
    return expr1_repr
    #print(expr1_repr)

if __name__ == '__main__':
    """
    CommandLine:
        python -m vtool.symbolic
        python -m vtool.symbolic --allexamples
        python -m vtool.symbolic --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
