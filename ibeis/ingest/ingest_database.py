#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
"""
This module lists known raw databases and how to ingest them.
"""
from __future__ import absolute_import, division, print_function
import ibeis
from os.path import exists
from itertools import izip
from ibeis.dev import ibsfuncs
from ibeis.control import IBEISControl
import utool

#
#
### <STANDARD DATABASES> ###

STANDARD_INGEST_FUNCS = {}


def __standard(dbname):
    """  Decorates a function as a standard ingestable database """
    def __registerdb(func):
        STANDARD_INGEST_FUNCS[dbname] = func
        return func
    return __registerdb


@__standard('polar_bears')
def ingest_polar_bears(db):
    return Ingestable(db, ingest_type='named_folders',
                      adjust_percent=0.00,
                      fmtkey='name')


@__standard('testdb1')
def ingest_testdb1(db):
    from vtool.tests import grabdata
    def postingest_tesdb1_func(ibs):
        print('postingest_tesdb1_func')
        # Adjust data as we see fit
        import numpy as np
        gid_list = np.array(ibs.get_valid_gids())
        unixtimes_even = (gid_list[0::2] + 100).tolist()
        unixtimes_odd  = (gid_list[1::2] + 9001).tolist()
        unixtime_list = unixtimes_even + unixtimes_odd
        ibs.set_image_unixtime(gid_list, unixtime_list)
        # Unname first aid in every name
        aid_list = ibs.get_valid_aids()
        nid_list = ibs.get_annotation_labelids(aid_list, 'INDIVIDUAL_KEY')
        nid_list = [ (nid[0] if len(nid) > 0 else None) for nid in nid_list]
        unique_flag = utool.flag_unique_items(nid_list)
        unique_nids = utool.filter_items(nid_list, unique_flag)
        none_nids = [ nid is not None for nid in nid_list]
        flagged_nids = [nid for nid in unique_nids if nid_list.count(nid) > 1]
        plural_flag = [nid in flagged_nids for nid in nid_list]
        flag_list = map(all, izip(plural_flag, unique_flag, none_nids))
        flagged_aids = utool.filter_items(aid_list, flag_list)
        if utool.VERYVERBOSE:
            def print2(*args):
                print('[post_testdb1] ' + ', '.join(args))
            print2('aid_list=%r' % aid_list)
            print2('nid_list=%r' % nid_list)
            print2('unique_flag=%r' % unique_flag)
            print2('plural_flag=%r' % plural_flag)
            print2('unique_nids=%r' % unique_nids)
            print2('none_nids=%r' % none_nids)
            print2('flag_list=%r' % flag_list)
            print2('flagged_nids=%r' % flagged_nids)
            print2('flagged_aids=%r' % flagged_aids)
            # print2('new_nids=%r' % new_nids)
        # Unname, some annotations for testing
        delete_aids = utool.filter_items(aid_list, flag_list)
        ibs.delete_annotation_nids(delete_aids, 'INDIVIDUAL_KEY')
        # Add all annotations with names as exemplars
        from ibeis.control.IBEISControl import IBEISController
        assert isinstance(ibs, IBEISController)
        unflagged_aids = utool.get_dirty_items(aid_list, flag_list)
        exemplar_flags = [True] * len(unflagged_aids)
        ibs.set_annotation_exemplar_flag(unflagged_aids, exemplar_flags)

        return None
    return Ingestable(db, ingest_type='named_images',
                      fmtkey=ibsfuncs.FMT_KEYS.name_fmt,
                      img_dir=grabdata.get_testdata_dir(),
                      adjust_percent=0.00,
                      postingest_func=postingest_tesdb1_func)


@__standard('snails_drop1')
def ingest_snails_drop1(db):
    return Ingestable(db,
                      ingest_type='named_images',
                      fmtkey=ibsfuncs.FMT_KEYS.snails_fmt,
                      adjust_percent=.20)


@__standard('JAG_Kieryn')
def ingest_JAG_Kieryn(db):
    return Ingestable(db,
                      ingest_type='unknown',
                      adjust_percent=0.00)


def get_standard_ingestable(db):
    if db in STANDARD_INGEST_FUNCS:
        return STANDARD_INGEST_FUNCS[db](db)
    else:
        raise AssertionError('Unknown db=%r' % (db,))


def ingest_standard_database(db, force_delete=False):
    print('[ingest] Ingest Standard Database: db=%r' % (db,))
    ingestable = get_standard_ingestable(db)
    dbdir = ibeis.sysres.db_to_dbdir(ingestable.db, allow_newdir=True, use_sync=False)
    utool.ensuredir(dbdir, verbose=True)
    if force_delete:
        ibsfuncs.delete_ibeis_database(dbdir)
    ibs = IBEISControl.IBEISController(dbdir)
    ingest_rawdata(ibs, ingestable)

### </STANDARD DATABASES> ###
#
#


class Ingestable(object):
    """ Temporary structure representing how to ingest a databases """
    def __init__(self, db, img_dir=None, ingest_type=None, fmtkey=None,
                 adjust_percent=0.0, postingest_func=None):
        self.db              = db
        self.img_dir         = img_dir
        self.ingest_type     = ingest_type
        self.fmtkey          = fmtkey
        self.adjust_percent  = adjust_percent
        self.postingest_func = postingest_func
        self.ensure_feasibility()

    def ensure_feasibility(self):
        rawdir  = ibeis.sysres.get_rawdir()
        if self.img_dir is None:
            # Try to find data either the raw or work dir
            self.img_dir = ibeis.sysres.db_to_dbdir(self.db, extra_workdirs=[rawdir])
        msg = 'Cannot find img_dir for db=%r, img_dir=%r' % (self.db, self.img_dir)
        assert self.img_dir is not None, msg
        assert exists(self.img_dir), msg
        if self.ingest_type == 'named_folders':
            assert self.fmtkey == 'name'


def ingest_rawdata(ibs, ingestable, localize=False):
    """
    Ingests rawdata into an ibeis database.

    if ingest_type == 'named_folders':
        Converts folder structure where folders = name, to ibsdb
    if ingest_type == 'named_images':
        Converts imgname structure where imgnames = name_id.ext, to ibsdb
    """
    img_dir         = ingestable.img_dir
    ingest_type     = ingestable.ingest_type
    fmtkey          = ingestable.fmtkey
    adjust_percent  = ingestable.adjust_percent
    postingest_func = ingestable.postingest_func
    print('[ingest] ingesting rawdata: img_dir=%r, injest_type=%r' % (img_dir, ingest_type))
    # Get images in the image directory
    gpath_list  = list_ingestable_images(img_dir, recursive=True)
    # Parse structure for image names
    if ingest_type == 'named_folders':
        name_list = ibsfuncs.get_names_from_parent_folder(gpath_list, img_dir, fmtkey)
        pass
    if ingest_type == 'named_images':
        name_list = ibsfuncs.get_names_from_gnames(gpath_list, img_dir, fmtkey)
    if ingest_type == 'unknown':
        name_list = [ibs.key_defaults['INDIVIDUAL_KEY'] for _ in xrange(len(gpath_list))]

    # Add Images
    gid_list_ = ibs.add_images(gpath_list)
    # <DEBUG>
    #print('added: ' + utool.indentjoin(map(str, zip(gid_list_, gpath_list))))
    unique_gids = list(set(gid_list_))
    print("[ingest] Length gid list: %d" % len(gid_list_))
    print("[ingest] Length unique gid list: %d" % len(unique_gids))
    assert len(gid_list_) == len(gpath_list)
    for gid in gid_list_:
        if gid is None:
            print('[ingest] big fat warning')
    # </DEBUG>
    gid_list = utool.filter_Nones(gid_list_)
    unique_gids, unique_names, unique_notes = ibsfuncs.resolve_name_conflicts(
        gid_list, name_list)
    # Add ANNOTATIONs with names and notes
    aid_list = ibsfuncs.use_images_as_annotations(ibs, unique_gids,
                                                  name_list=unique_names,
                                                  notes_list=unique_notes,
                                                  adjust_percent=adjust_percent)
    if localize:
        ibsfuncs.localize_images(ibs)
    if postingest_func is not None:
        postingest_func(ibs)
    # Print to show success
    #ibs.print_image_table()
    #ibs.print_tables()
    #ibs.print_annotation_table()
    #ibs.print_alr_table()
    #ibs.print_label_table()
    #ibs.print_image_table()
    return aid_list


def list_ingestable_images(img_dir, fullpath=True, recursive=True):
    ignore_list = ['_hsdb', '.hs_internals', '_ibeis_cache', '_ibsdb']
    gpath_list = utool.list_images(img_dir,
                                   fullpath=fullpath,
                                   recursive=recursive,
                                   ignore_list=ignore_list)
    # Ensure in unix format
    gpath_list = map(utool.unixpath, gpath_list)
    return gpath_list


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # win32
    print('__main__ = ingest_database.py')
    print(utool.unindent(
        '''
        usage:
        ./ibeis/ingest/ingest_database.py --db [dbname]

        Valid dbnames:''') + utool.indentjoin(STANDARD_INGEST_FUNCS.keys(), '\n  * '))
    db = utool.get_arg('--db', str, None)
    ibs = ingest_standard_database(db)
    #img_dir = join(ibeis.sysres.get_workdir(), 'polar_bears')
    #main_locals = ibeis.main(dbdir=img_dir, gui=False)
    #ibs = main_locals['ibs']
    #ingest_rawdata(ibs, img_dir)
