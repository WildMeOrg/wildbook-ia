#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import ibeis
from os.path import join, expanduser
from ibeis.injest.injest_named_images import injest_named_images

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # win32
    ibeis._preload()
    from ibeis.dev import ibsfuncs
    img_dir = expanduser('~/data/raw/snails_drop1')
    dbdir = join(ibeis.params.get_workdir(), 'snails_drop1')
    ibsfuncs.delete_ibeis_database(dbdir)
    main_locals = ibeis.main(dbdir=dbdir)
    ibs = main_locals['ibs']
    back = main_locals.get('back', None)
    fmtkey = 'snails'
    injest_named_images(ibs, img_dir, fmtkey, adjust_percent=.20)
    ibsfuncs.localize_images(ibs)

    # Print to show success
    ibs.print_name_table()
    ibs.print_image_table()
    ibs.print_roi_table()

