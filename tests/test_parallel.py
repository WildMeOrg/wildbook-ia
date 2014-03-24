#!/usr/bin/env python
'''
Tests IBEIS parallel
'''
from __future__ import division, print_function
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
    #from utool import util_time
    import pyhesaff

    imgdir = dirname(pyhesaff.__file__)
    gname_list = [
        'lena.png',
        'zebra.png',
        'test.png',
    ]
    # Increase data size
    print('use --ndata to specify bigger data')
    ndata = util_arg.get_arg('--ndata', type_=int, default=10)

    # Build gpath_list
    gname_list = util_list.flatten([gname_list] * ndata)
    gpath_list = [join(imgdir, path) for path in gname_list]

    # Run parallel tasks
    with util_time.Timer('processing tasks in parallel') as t:
        result_parallel = util_parallel.process(pyhesaff.detect_kpts,
                                                gpath_list)

    for kpts, desc in result_parallel:
        print('[test] kpts.shape=%r, desc.sum=%r' % (kpts.shape, desc.sum()))

    # Compare to serial if needed
    compare_serial = util_arg.get_flag('--compare')
    if compare_serial:
        with util_time.Timer('processing tasks in serial') as t:
            result_serial = util_parallel._process_serial(pyhesaff.detect_kpts,
                                                          gpath_list)

    main_api.main_loop(main_locals, loop=False)
