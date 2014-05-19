#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
"""
This module lists known raw databases and how to injest them.
"""
from __future__ import absolute_import, division, print_function
import ibeis
from os.path import exists
from ibeis.dev import ibsfuncs
from ibeis.control import IBEISControl
import utool


class Injestable(object):
    """ Temp struct for storing info to injest """
    def __init__(self, db, img_dir=None, injest_type=None, fmtkey=None,
                 adjust_percent=0.0, postinjest_func=None):
        self.db              = db
        self.img_dir         = img_dir
        self.injest_type     = injest_type
        self.fmtkey          = fmtkey
        self.adjust_percent  = adjust_percent
        self.postinjest_func = postinjest_func
        self.ensure_feasibility()

    def ensure_feasibility(self):
        rawdir  = ibeis.sysres.get_rawdir()
        if self.img_dir is None:
            # Try to find data either the raw or work dir
            self.img_dir = ibeis.sysres.db_to_dbdir(self.db, extra_workdirs=[rawdir])
        msg = 'Cannot find img_dir for db=%r, img_dir=%r' % (self.db, self.img_dir)
        assert self.img_dir is not None, msg
        assert exists(self.img_dir), msg
        if self.injest_type == 'named_folders':
            assert self.fmtkey == 'name'


def get_standard_injestable(db):
    if db == 'polar_bears':
        injestable = Injestable(db, injest_type='named_folders',
                                adjust_percent=0.00,
                                fmtkey='name')
    elif db == 'testdb1':
        from vtool.tests import grabdata
        def postinjest_tesdb1_func(ibs):
            import numpy as np
            gid_list = np.array(ibs.get_valid_gids())
            unixtimes_even = (gid_list[0::2] + 100).tolist()
            unixtimes_odd  = (gid_list[1::2] + 9001).tolist()
            unixtime_list = unixtimes_even + unixtimes_odd
            ibs.set_image_unixtime(gid_list, unixtime_list)
            return None
        injestable = Injestable(db, injest_type='named_images',
                                fmtkey=ibsfuncs.FMT_KEYS.name_fmt,
                                img_dir=grabdata.get_testdata_dir(),
                                adjust_percent=0.00,
                                postinjest_func=postinjest_tesdb1_func)
    elif db == 'snails_drop1':
        injestable = Injestable(db,
                                injest_type='named_images',
                                fmtkey=ibsfuncs.FMT_KEYS.snails_fmt,
                                adjust_percent=.20)
    elif db == 'JAG_Kieryn':
        injestable = Injestable(db,
                                injest_type='unknown',
                                adjust_percent=0.00)
    else:
        raise AssertionError('Unknown db=%r' % (db,))
    return injestable


def list_injestable_images(img_dir, fullpath=True, recursive=True):
    ignore_list = ['_hsdb', '.hs_internals', '_ibeis_cache', '_ibsdb']
    gpath_list = utool.list_images(img_dir,
                                   fullpath=fullpath,
                                   recursive=recursive,
                                   ignore_list=ignore_list)
    # Ensure in unix format
    gpath_list = map(utool.unixpath, gpath_list)
    return gpath_list


def injest_rawdata(ibs, injestable, localize=False):
    """
    Injests rawdata into an ibeis database.

    if injest_type == 'named_folders':
        Converts folder structure where folders = name, to ibsdb
    if injest_type == 'named_images':
        Converts imgname structure where imgnames = name_id.ext, to ibsdb
    """
    img_dir         = injestable.img_dir
    injest_type     = injestable.injest_type
    fmtkey          = injestable.fmtkey
    adjust_percent  = injestable.adjust_percent
    postinjest_func = injestable.postinjest_func

    # Get images in the image directory
    gpath_list  = list_injestable_images(img_dir, recursive=True)
    # Parse structure for image names
    if injest_type == 'named_folders':
        name_list = ibsfuncs.get_names_from_parent_folder(gpath_list, img_dir, fmtkey)
        pass
    if injest_type == 'named_images':
        name_list = ibsfuncs.get_names_from_gnames(gpath_list, img_dir, fmtkey)
    if injest_type == 'unknown':
        name_list = [ibsfuncs.UNKNOWN_NAME for _ in xrange(len(gpath_list))]

    # Add Images
    gid_list = utool.filter_Nones(ibs.add_images(gpath_list))
    #DEBUG = True
    #if DEBUG:
    #    invalid_list = [gid is None for gid in gid_list]
    #    none_indexes = utool.filter_items(range(len(gid_list)), invalid_list)
    #    none_gpaths  = utool.filter_items(gpath_list, invalid_list)
    #    print(none_gpaths)
    #    print(none_indexes)
    #    for gpath in none_gpaths:
    #        from ibeis.model.preproc import preproc_image
    #        utool.checkpath(gpath, verbose=True)
    #        imgtup = preproc_image.preprocess_image(gpath)
    #        guuid_list = [imgtup[0]]
    #    utool.embed()
    # Resolve conflicts
    unique_gids, unique_names, unique_notes = ibsfuncs.resolve_name_conflicts(
        gid_list, name_list)
    # Add ROIs with names and notes
    rid_list = ibsfuncs.use_images_as_rois(ibs, unique_gids,
                                           name_list=unique_names,
                                           notes_list=unique_notes,
                                           adjust_percent=adjust_percent)
    if localize:
        ibsfuncs.localize_images(ibs)
    if postinjest_func is not None:
        postinjest_func(ibs)
    # Print to show success
    ibs.print_name_table()
    ibs.print_image_table()
    ibs.print_roi_table()
    return rid_list


def injest_standard_database(db, force_delete=False):
    print('[injest] Injest Standard Database: db=%r' % (db,))
    injestable = get_standard_injestable(db)
    dbdir = ibeis.sysres.db_to_dbdir(injestable.db, allow_newdir=True, use_sync=False)
    utool.ensuredir(dbdir, verbose=True)
    if force_delete:
        ibsfuncs.delete_ibeis_database(dbdir)
    ibs = IBEISControl.IBEISController(dbdir)
    injest_rawdata(ibs, injestable)


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # win32
    print('__main__ = injest_database.py')
    db = utool.get_arg('--db', str, None)
    ibs = injest_standard_database(db)
    #img_dir = join(ibeis.sysres.get_workdir(), 'polar_bears')
    #main_locals = ibeis.main(dbdir=img_dir, gui=False)
    #ibs = main_locals['ibs']
    #injest_rawdata(ibs, img_dir)
