# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

try:
    from wbia.guitool import api_tree_node_cython

    if 'api_tree_node_cython' not in globals():
        raise ImportError('')
    print('[test_treenode] cython ON')
except ImportError:
    print('[test_treenode] cython OFF')
from wbia.guitool import api_tree_node


def test_build_internal_structure():
    """

    CommandLine:
        python -m wbia.guitool.tests.test_treenode --test-test_build_internal_structure

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.guitool.tests.test_treenode import *  # NOQA
        >>> test_build_internal_structure()
    """
    if 'api_tree_node_cython' in globals():
        print('test cython ON')
        _test_build_internal_structure(api_tree_node_cython, 'cython')
    else:
        print('test cython OFF')

    _test_build_internal_structure(api_tree_node, 'python')
    print('finished all tests')


def _test_build_internal_structure(_module, lang):
    import utool as ut

    # Test data
    N = 6
    # N = 2000

    def ider_level0():
        return range(N)

    def ider_level1(input_):
        def _single(x):
            return [y for y in range(x ** 2, x ** 2 + max(0, ((N // 1) - x - 1)))]

        if isinstance(input_, list):
            return [_single(x) for x in input_]
        else:
            x = input_
            return _single(x)

    # Build Structure
    ider_list = [ider_level0, ider_level1]
    num_levels = len(ider_list)
    # TEST RECURSIVE
    print('================')
    with ut.Timer(lang + ' recursive:'):
        if num_levels == 0:
            root_id_list = []
        else:
            root_id_list = ider_list[0]()
        root_node1 = _module.TreeNode(-1, None, -1)
        level = 0
        _module._populate_tree_recursive(
            root_node1, root_id_list, num_levels, ider_list, level
        )
    if N < 10:
        print('')
        print(api_tree_node.tree_node_string(root_node1, indent=' *  '))
    print('================')
    # with ut.Timer(lang + ' iterative:'):
    #    # TEST ITERATIVE
    #    # TODO: Vet this code a bit more.
    #    root_node2 = _module.TreeNode(-1, None, -1)
    #    _module._populate_tree_iterative(
    #        root_node2, num_levels, ider_list)
    # if N < 10:
    #    print('')
    #    print(api_tree_node.tree_node_string(root_node2, indent=' *  '))
    print('================')
    print('finished %s test' % lang)


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.guitool.tests.test_treenode
        python -m wbia.guitool.tests.test_treenode --allexamples
        python -m wbia.guitool.tests.test_treenode --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
