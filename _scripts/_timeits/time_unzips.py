"""
Timeit tests for various ways of unziping lists
"""
from __future__ import absolute_import, division, print_function
import timeit
import numpy as np
from six.moves import zip
from utool._internal.meta_util_six import get_funcname

number = 2000
num_data = 5000

list1 = np.random.rand(num_data).tolist()
list2 = np.random.rand(num_data).tolist()


def test_unizip():
    """ WINNER """
    list1_, list2_ = list(zip(*tup_list))
    return list1_, list2_


def test_unzip():
    list1_, list2_ = list(zip(*tup_list))
    return list1_, list2_


def test_listcomp1():
    list1_ = [tup[0] for tup in tup_list]
    list2_ = [tup[1] for tup in tup_list]
    return list1_, list2_


def test_listcomp2():
    list1_ = [item1 for (item1, item2) in tup_list]
    list2_ = [item2 for (item1, item2) in tup_list]
    return list1_, list2_


tup_list = list(zip(list1, list2))


if __name__ == '__main__':
    test_funcs = [
        test_unizip,
        test_unzip,
        test_listcomp1,
        test_listcomp2,
    ]
    func_strs = ', '.join([get_funcname(func) for func in test_funcs])
    # Timeit main trick
    setup = 'from __main__ import (list1, list2, %s) ' % (func_strs,)
    for func in test_funcs:
        funcname = get_funcname(func)
        stmt = '%s()' % funcname
        print('Running: %s' % stmt)
        total_time = timeit.timeit(stmt=stmt, setup=setup, number=number)
        print('timed: %r seconds in %s' % (total_time, funcname))


"""
RESULTS:

Running: test_unizip()
timed: 1.4483390469557733 seconds in test_unizip

Running: test_unzip()
timed: 1.504843270633586 seconds in test_unzip

Running: test_listcomp1()
timed: 2.788620085133971 seconds in test_listcomp1

Running: test_listcomp2()
timed: 2.20052683401277 seconds in test_listcomp2
"""
