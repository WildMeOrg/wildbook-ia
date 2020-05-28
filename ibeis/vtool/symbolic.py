# -*- coding: utf-8 -*-
"""
Sympy helpers
"""
from __future__ import absolute_import, division, print_function
import numpy as np
import six
import utool as ut
import ubelt as ub


def custom_sympy_attrs(mat):
    import sympy
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
    import sympy
    mat = sympy.Matrix(arr)
    mat = custom_sympy_attrs(mat)
    return mat


def evalprint(str_, globals_=None, locals_=None, simplify=False):
    import sympy
    if globals_ is None:
        globals_ = ut.get_parent_frame().f_globals
    if locals_ is None:
        locals_ = ut.get_parent_frame().f_locals
    if isinstance(str_, six.string_types):
        var = eval(str_, globals_, locals_)
    else:
        var = str_
        str_ = ut.get_varname_from_stack(var, N=1)
    if simplify is True:
        var = sympy.simplify(var)
    print(ub.hzcat(str_ + ' = ', repr(var)))


def check_expr_eq(expr1, expr2, verbose=True):
    """
    Does not work in general. Problem is not decidable.
    Thanks Richard.

    Args:
        expr1 (?):
        expr2 (?):

    CommandLine:
        python -m vtool_ibeis.symbolic --test-check_expr_eq

    SeeALso:
        vt.symbolic_randcheck

    Example:
        >>> # DISABLE_DOCTEST
        >>> from vtool_ibeis.symbolic import *  # NOQA
        >>> expr1 = sympy.Matrix([ [sx*x + 1.0*tx + w1*y], [sy*y + 1.0*ty + w2*x], [1.0]])
        >>> expr2 = sympy.Matrix([ [sx*x + tx + w1*y], [sy*y + ty + w2*x], [1]])
        >>> result = check_expr_eq(expr1, expr2)
        >>> print(result)
    """
    import sympy
    if isinstance(expr1, six.string_types):
        expr1 = sympy.simplify(expr1)
    if isinstance(expr2, six.string_types):
        expr2 = sympy.simplify(expr2)
    print(ub.hzcat('Checking if ', repr(expr1), ' == ', repr(expr2)))
    random_point_check = expr1.equals(expr2)
    if random_point_check is None:
        failexpr = expr1.equals(expr2, failing_expression=True)
        print('failexpr = %r' % (failexpr,))
        random_point_check = False
    print('... seems %r' % (random_point_check,))
    #return random_point_check
    expr3 = expr1 - expr2
    if not random_point_check and True:
        common_symbols = expr1.free_symbols.intersection(expr2.free_symbols)
        if len(common_symbols):
            y = sympy.symbols('y')  # Hack, should be a new symbol
            symbol = common_symbols.pop()
            soln1 = sympy.solve(sympy.Eq(sympy.simplify(expr1), y), symbol)
            soln2 = sympy.solve(sympy.Eq(sympy.simplify(expr2), y), symbol)
            print('Solving expr1 for common symbol: ' + str(soln1))
            print('Solving expr2 for common symbol: ' + str(soln2))
            if soln1 == soln2:
                print('This seems True')
            else:
                print('This seems False')
        sympy.solve(sympy.Eq(sympy.simplify(expr2), y), 'd')
    print(ub.hzcat('... checking 0 ', repr(expr3)))
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
    import sympy
    expr1_repr = sympy.latex(expr1)
    expr1_repr = expr1_repr.replace('\\\\', '\\\\\n')
    expr1_repr = expr1_repr.replace(r'\left[\begin{smallmatrix}{}', '\\MAT{\n')
    expr1_repr = expr1_repr.replace(r'\end{smallmatrix}\right]', '\n}')
    expr1_repr = expr1_repr.replace(r'\left[\begin{matrix}', '\\BIGMAT{\n')
    expr1_repr = expr1_repr.replace(r'\end{matrix}\right]', '\n}')
    expr1_repr = expr1_repr.replace(r'\left (', '(')
    expr1_repr = expr1_repr.replace(r'\right )', ')')
    expr1_repr = expr1_repr.replace(r'\left(', '(')
    expr1_repr = expr1_repr.replace(r'\right)', ')')
    # hack of align
    expr1_repr = ut.align(expr1_repr, '&', pos=None)
    return expr1_repr
    #print(expr1_repr)


def sympy_numpy_repr(expr1):
    import re
    expr1_repr = repr(expr1)
    expr1_repr = expr1_repr.replace('Matrix', 'np.array')
    expr1_repr = re.sub('\\bsin\\b', 'np.sin', expr1_repr)
    expr1_repr = re.sub('\\bcos\\b', 'np.cos', expr1_repr)
    expr1_repr = ut.autoformat_pep8(expr1_repr)
    print(expr1_repr)
    #import autopep8
    #autopep8.fix_code(expr1_repr)


"""
Symbolic Scrap Work:
The number of negative reviews needed is usually much larger than the number of
positive reviews.

import sympy
from sympy.abc import theta
import sympy.stats
from sympy.stats import E as mean
items = sympy.symbols('a, b, c, d')

from sympy.stats import FiniteRV, P, E
density = {0: .1, 1: .2, 2: .3, 3: .4}
X = FiniteRV('X', density)

cs = sympy.stats.FiniteRV(str('X'), {0: .5, 1: .5})

cs = [[None] * np.random.randint(10) for _ in range(1)]
print(sum(len(c) - 1 for c in cs))
print(np.mean([len(c) for c in cs]) * len(cs) - len(cs))

ori = theta
x, y, iv11, iv21, iv22, patch_size = sympy.symbols('x y iv11 iv21 iv22 S')



"""


if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m vtool_ibeis.symbolic
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
