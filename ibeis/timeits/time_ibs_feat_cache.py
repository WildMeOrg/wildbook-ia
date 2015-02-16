#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
# Python
import multiprocessing
# Tools
import utool
#from plottool import draw_func2 as df2
#IBEIS
#from ibeis.viz import interact
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TIME_QUERY]')


@profile
def TIME_QUERY(ibs):
    print('[TIME_QUERY]')
    #valid_aids = ibs.get_valid_aids()  # [0:20]
    valid_aids = ibs.get_valid_aids()[0:10]  # [0:20]
    qaid_list = valid_aids
    daid_list = valid_aids

    # Query without using the query cache
    querykw = {
        'use_bigcache': False,
        'use_cache': False,
    }
    with utool.Timer('timing all vs all query'):
        qres_dict = ibs._query_chips4(qaid_list, daid_list, **querykw)

    print('[/TIME_QUERY]')
    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    import ibeis
    main_locals = ibeis.main(defaultdb='PZ_MOTHERS', gui=False)
    ibs = main_locals['ibs']
    time_locals = TIME_QUERY(ibs)
    execstr = utool.execstr_dict(time_locals, 'time_locals')
    exec(execstr)
    exec(utool.ipython_execstr())
