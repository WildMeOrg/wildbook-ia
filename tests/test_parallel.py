#!/usr/bin/env python
'''
Tests IBEIS parallel
'''
from __future__ import division, print_function
import import_sysreq  # NOQA
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()  # for win32
    from ibeis.dev import main_api
    main_locals = main_api.main()
    from os.path import dirname, join
    from utool import util_parallel
    from utool import util_time
    from utool import util_list
    from utool import util_arg
    from utool import util_print
    import pyhesaff
    from utool.util_inject import inject, inject_all
    print, print_, printDBG, rrr, profile = inject(__name__, '[test_parallel]')
    inject_all()

    #from utool import util_time

    imgdir = dirname(pyhesaff.__file__)
    gname_list = [
        'lena.png',
        'zebra.png',
        'lena.png',
        'lena.png',
        'lena.png',
        'zebra.png',
        'zebra.png',
        'zebra.png',
        'test.png',
    ]
    # Increase data size
    print('use --ndata to specify bigger data')
    ndata = util_arg.get_arg('--ndata', type_=int, default=1)

    # Build gpath_list
    if ndata == 0:
        gname_list = ['test.png']
    else:
        gname_list = util_list.flatten([gname_list] * ndata)
    gpath_list = [join(imgdir, path) for path in gname_list]
    args_list  = [(gpath,) for gpath in gpath_list]

    @util_arg.argv_flag_dec
    def print_test_results(result_list):
        for kpts, desc in result_list:
            print('[test] kpts.shape=(%4d, %d), desc.sum=%8d' % (kpts.shape[0],
                                                                 kpts.shape[1],
                                                                 desc.sum()))

    # Run parallel tasks
    @util_print.indent_decor('[test_task]')
    def test_par_task(num_procs=None, dict_args={}):
        if num_procs is not None:
            util_parallel.close_pool()
            util_parallel.init_pool(num_procs)
        with util_time.Timer('processing tasks in parallel'):
            print(dict_args)
            result_list = util_parallel.process(pyhesaff.detect_kpts, args_list, dict_args)
        print_test_results(result_list)
        return result_list

    hesaff_kwargs = {
        'scale_min': 0,
        'scale_max': -1,
        'nogravity_hack': True
    }

    test_par_task(None, hesaff_kwargs)

    # Compare to serial if needed
    compare_serial = util_arg.get_flag('--compare')
    if compare_serial:
        test_par_task(1, hesaff_kwargs)

    main_api.main_loop(main_locals, loop=False)
