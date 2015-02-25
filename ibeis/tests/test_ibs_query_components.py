#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
# Python
import six
#import sys
from collections import OrderedDict
import multiprocessing
# Tools
import utool
from plottool import draw_func2 as df2
#IBEIS
from ibeis import viz
from ibeis.model.hots import query_helpers
print, print_, printDBG, rrr, profile = utool.inject(
    __name__, '[TEST_QUERY_COMP]')


def TEST_QUERY_COMP(ibs):
    r"""
    CommandLine:
        python -m ibeis.tests.test_ibs_query_components --test-TEST_QUERY_COMP

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.tests.test_ibs_query_components import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> # execute function
        >>> TEST_QUERY_COMP(ibs)
    """
    print('[TEST_QUERY_COMP]')
    aids = ibs.get_valid_aids()
    index = 0
    index = utool.get_argval('--index', type_=int, default=index)
    qaid_list = utool.safe_slice(aids, index, index + 1)
    print('[TEST_QUERY_COMP] len(qaid_list)=%r' % (qaid_list))
    try:
        comp_locals_ = query_helpers.get_query_components(ibs, qaid_list)
        qres_dict = OrderedDict([
            ('ORIG', comp_locals_['qres_ORIG']),
            ('FILT', comp_locals_['qres_FILT']),
            ('SVER', comp_locals_['qres_SVER']),
        ])

        top_aids = qres_dict['SVER'].get_top_aids()
        aid2 = top_aids[0]
    except Exception as ex:
        if 'qres_dict' in vars():
            for name, qres in qres_dict.items():
                print(name)
                print(qres.get_inspect_str())
        utool.printex(ex, keys=['qaid_list'], pad_stdout=True)
        raise

    for px, (lbl, qres) in enumerate(six.iteritems(qres_dict)):
        print(lbl)
        fnum = df2.next_fnum()
        df2.figure(fnum=fnum, doclf=True)
        qres.ishow_top(ibs,  fnum=fnum, top_aids=top_aids, ensure=False)
        df2.set_figtitle(lbl)
        df2.adjust_subplots_safe(top=.8)

    fnum = df2.next_fnum()

    qaid2_svtups = comp_locals_['qaid2_svtups']
    qaid2_chipmatch_FILT = comp_locals_['qaid2_chipmatch_FILT']
    aid1 = qaid = comp_locals_['qaid']
    aid2_svtup  = qaid2_svtups[aid1]
    chipmatch_FILT = qaid2_chipmatch_FILT[aid1]
    viz.show_sver(ibs, aid1, aid2, chipmatch_FILT, aid2_svtup, fnum=fnum)
    return locals()


#if __name__ == '__main__':
#    """
#    CommandLine:
#        python -m ibeis.tests.test_ibs_query_components
#        python -m ibeis.tests.test_ibs_query_components --allexamples
#        python -m ibeis.tests.test_ibs_query_components --allexamples --noface --nosrc
#    """
#    multiprocessing.freeze_support()  # for win32
#    import utool as ut  # NOQA
#    ut.doctest_funcs()

if __name__ == '__main__':
    import sys
    multiprocessing.freeze_support()  # For windows
    import ibeis
    main_locals = ibeis.main(defaultdb='testdb1', gui=False)
    ibs = main_locals['ibs']
    test_locals = utool.run_test(TEST_QUERY_COMP, ibs)
    if '--noshow' not in sys.argv:
        exec(df2.present())
    #execstr = utool.execstr_dict(test_locals, 'test_locals')
    #exec(execstr)
