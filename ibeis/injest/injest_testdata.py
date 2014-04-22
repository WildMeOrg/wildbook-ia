#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import ibeis
from os.path import join
from ibeis.injest.injest_named_images import injest_named_images
import utool

img_dir = join(ibeis.params.get_workdir(), 'testdata')
utool.remove_files_in_dir(img_dir)

main_locals = ibeis.main(dbdir=img_dir)
ibs = main_locals['ibs']
back = main_locals.get('back', None)
fmtkey = 'testdata'
injest_named_images(ibs, img_dir, fmtkey)

ibs.print_name_table()
ibs.print_name_table()
ibs.print_roi_table()
