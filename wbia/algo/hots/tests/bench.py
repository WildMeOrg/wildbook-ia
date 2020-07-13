# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import utool as ut

(print, rrr, profile) = ut.inject2(__name__)


def benchmark_knn():
    r"""
    CommandLine:
        python ~/code/wbia/wbia/algo/hots/tests/bench.py benchmark_knn --profile

    Example:
        >>> # DISABLE_DOCTEST
        >>> from bench import *  # NOQA
        >>> result = benchmark_knn()
        >>> print(result)
    """
    from wbia.algo.hots import _pipeline_helpers as plh
    from wbia.algo.hots.pipeline import nearest_neighbors
    import wbia

    verbose = True
    qreq_ = wbia.testdata_qreq_(
        defaultdb='PZ_PB_RF_TRAIN',
        t='default:K=3,requery=True,can_match_samename=False',
        a='default:qsize=100',
        verbose=1,
    )
    locals_ = plh.testrun_pipeline_upto(qreq_, 'nearest_neighbors')
    Kpad_list, impossible_daids_list = ut.dict_take(
        locals_, ['Kpad_list', 'impossible_daids_list']
    )
    nns_list1 = nearest_neighbors(  # NOQA
        qreq_, Kpad_list, impossible_daids_list, verbose=verbose
    )


if __name__ == '__main__':
    r"""
    CommandLine:
        export PYTHONPATH=$PYTHONPATH:/home/joncrall/code/wbia/wbia/algo/hots/tests
        python ~/code/wbia/wbia/algo/hots/tests/bench.py
        python ~/code/wbia/wbia/algo/hots/tests/bench.py --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
