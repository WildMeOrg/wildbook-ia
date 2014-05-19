#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
from ibeis.injest import injest_database
from ibeis.dev import ibsfuncs
import ibeis

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # win32
    injestable = injest_database.get_std_injestable('snails_drop1')
    dbdir = ibeis.sysres.db_to_dbdir(injestable.db, allow_newdir=True)
    ibsfuncs.delete_ibeis_database(dbdir)
    ibs   = ibeis.IBEISController(dbdir)
    injest_database.injest_rawdata(ibs, injestable)

    #from ibeis.injest.injest_named_images import injest_named_images
    #ibeis._preload()
    #from ibeis.dev import ibsfuncs
    #img_dir = expanduser('~/data/raw/snails_drop1')
    #dbdir = join(ibeis.sysres.get_workdir(), 'snails_drop1')
    #ibsfuncs.delete_ibeis_database(dbdir)
    #main_locals = ibeis.main(dbdir=dbdir)
    #ibs = main_locals['ibs']
    #back = main_locals.get('back', None)
    #fmtkey = 'snails'
    #injest_named_images(ibs, img_dir, fmtkey, adjust_percent=.20)
    #ibsfuncs.localize_images(ibs)

    ## Print to show success
    #ibs.print_name_table()
    #ibs.print_image_table()
    #ibs.print_roi_table()
