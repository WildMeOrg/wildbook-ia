#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
# Python
import multiprocessing
# Tools
import utool
from plottool import draw_func2 as df2
#IBEIS
from ibeis.viz import interact
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_QUERY]')


def TEST_QUERY(ibs):
    print('[TEST_QUERY]')
    daid_list = ibs.get_valid_aids()
    print('[TEST_QUERY] len(daid_list)=%r' % (len(daid_list)))
    qaid_list = daid_list[0:1]
    print('[TEST_QUERY] len(qaid_list)=%r' % (len(qaid_list)))
    qres_dict = ibs._query_chips4(qaid_list, daid_list, use_cache=False, use_bigcache=False)
    qres_dict_ = ibs._query_chips4(qaid_list, daid_list)

    try:
        vals1 = list(qres_dict.values())
        vals2 = list(qres_dict_.values())
        assert len(vals1) == 1, 'expected 1 qres in result'
        assert len(vals2) == 1, 'expected 1 qres in result'
        assert list(qres_dict.keys()) == list(qres_dict_.keys()), 'qres cache doesnt work. key error'
        qres1 = vals1[0]
        qres2 = vals2[0]
        inspect_str1 = qres1.get_inspect_str(ibs)
        inspect_str2 = qres2.get_inspect_str(ibs)
        print(inspect_str1)
        assert inspect_str1 == inspect_str2, 'qres cache inconsistency'
        assert vals1 == vals2, 'qres cache doesnt work. val error'
    except AssertionError as ex:
        utool.printex(ex, key_list=list(locals().keys()))
        raise

    for qaid in qaid_list:
        qres  = qres_dict[qaid]
        top_aids = qres.get_top_aids()
        #top_aids = utool.safe_slice(top_aids, 3)
        aid2 = top_aids[0]
        fnum = df2.next_fnum()
        df2.figure(fnum=fnum, doclf=True)
        #viz_matches.show_matches(ibs, qres, aid2, fnum=fnum, in_image=True)
        #viz.show_qres(ibs, qres, fnum=fnum, top_aids=top_aids, ensure=False)
        interact.ishow_qres(ibs, qres, fnum=fnum, top_aids=top_aids,
                            ensure=False, annot_mode=1)
        df2.set_figtitle('Query Result')
        df2.adjust_subplots_safe(top=.8)
    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    import ibeis
    main_locals = ibeis.main(defaultdb='testdb1', gui=False)
    ibs = main_locals['ibs']
    test_locals = utool.run_test(TEST_QUERY, ibs)
    execstr = utool.execstr_dict(test_locals, 'test_locals')
    exec(execstr)
    exec(utool.ipython_execstr())
