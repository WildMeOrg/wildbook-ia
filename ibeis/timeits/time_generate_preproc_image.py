#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
# Python
import multiprocessing
# Tools
import utool
from ibeis.model.preproc.preproc_image import add_images_params_gen
#from plottool import draw_func2 as df2
#IBEIS
#from ibeis.viz import interact
print, print_, printDBG, rrr, profile = utool.inject(__name__,
                                                     '[TIME_GEN_PREPROC_IMG]')


def timeit2(stmt_, setup_, number=1000):
    import timeit
    stmt = utool.unindent(stmt_)
    setup = utool.unindent(setup_)
    print('----------')
    print('TIMEIT: \n' + stmt_)
    try:
        total_time = timeit.timeit(stmt, setup, number=number)
    except Exception as ex:
        utool.printex(ex, iswarning=False)
        raise
    print(' * timed: %r seconds' % (total_time))


@profile
def TIME_GEN_PREPROC_IMG(ibs):
    print('[TIME_GEN_PREPROC_IMG]')
    gid_list = ibs.get_valid_gids()
    gpath_list = ibs.get_image_paths(gid_list)

    setup = utool.unindent(
        '''
        from ibeis.model.preproc.preproc_image import add_images_params_gen
        genkw = dict(prog=False, verbose=True)
        gpath_list = %r
        ''' % (gpath_list,))

    print(utool.truncate_str(str(gpath_list), 80))

    print('Processing %d images' % (len(gpath_list),))

    timeit2('list(add_images_params_gen(gpath_list, chunksize=None, **genkw))', setup, 5)
    timeit2('list(add_images_params_gen(gpath_list, chunksize=None, **genkw))', setup, 5)
    timeit2('list(add_images_params_gen(gpath_list, chunksize=1, **genkw))', setup, 5)
    timeit2('list(add_images_params_gen(gpath_list, chunksize=2, **genkw))', setup, 5)
    timeit2('list(add_images_params_gen(gpath_list, chunksize=4, **genkw))', setup, 5)
    timeit2('list(add_images_params_gen(gpath_list, chunksize=8, **genkw))', setup, 5)
    timeit2('list(add_images_params_gen(gpath_list, chunksize=16, **genkw))', setup, 5)
    timeit2('list(add_images_params_gen(gpath_list, chunksize=32, **genkw))', setup, 5)
    timeit2('list(add_images_params_gen(gpath_list, chunksize=64, **genkw))', setup, 5)
    #timeit2('list(add_images_params_gen(gpath_list, chunksize=3, **genkw))', setup, 5)
    #timeit2('list(add_images_params_gen(gpath_list, chunksize=5, **genkw))', setup, 5)
    #timeit2('list(add_images_params_gen(gpath_list, chunksize=6, **genkw))', setup, 5)

    # TEST 1
    with utool.Timer('parallel chunksize=1'):
        output1 = list(add_images_params_gen(gpath_list, chunksize=1))
    print(utool.truncate_str(str(output1), 80))
    assert len(output1) == len(gpath_list), 'chuncksize changes output'

    # TEST 2
    with utool.Timer('parallel chunksize=2'):
        output2 = list(add_images_params_gen(gpath_list, chunksize=2))
    print(utool.truncate_str(str(output2), 80))
    assert output1 == output2, 'chuncksize changes output'

    # TEST N
    with utool.Timer('parallel chunksize=None'):
        outputN = list(add_images_params_gen(gpath_list, chunksize=None))
    print(utool.truncate_str(str(output2), 80))
    assert outputN == output2, 'chuncksize changes output'

    print('[/TIME_GEN_PREPROC_IMG]')
    return locals()


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    import ibeis
    main_locals = ibeis.main(defaultdb='GZ_ALL', gui=False)
    ibs = main_locals['ibs']
    time_locals = TIME_GEN_PREPROC_IMG(ibs)
    execstr = utool.execstr_dict(time_locals, 'time_locals')
    exec(execstr)
    exec(utool.ipython_execstr())
