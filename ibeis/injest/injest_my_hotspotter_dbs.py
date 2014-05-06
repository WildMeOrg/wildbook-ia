#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import ibeis
ibeis._preload()
from ibeis.dev import params
from ibeis.dev import ibsfuncs
from ibeis.control.IBEISControl import PATH_NAMES
from os.path import join, exists
import os
import numpy as np
from vtool import linalg as ltool
import utool
from ibeis.injest import injest_hsdb


def is_hsdb(dbdir):
    return (exists(join(dbdir, '_hsdb')) and
            exists(join(dbdir, '_hsdb', 'name_table.csv')) and
            exists(join(dbdir, '_hsdb', 'image_table.csv')) and
            exists(join(dbdir, '_hsdb', 'chip_table.csv')))


def is_hsinternal(dbdir):
    return exists(join(dbdir, '.hs_internals'))


def is_ibeisdb(dbdir):
    return exists(join(dbdir, PATH_NAMES._ibsdb))


def is_succesful_convert(dbdir):
    return exists(join(dbdir, PATH_NAMES._ibsdb, injest_hsdb.SUCCESS_FLAG_FNAME))


#is_hsinternals_list = np.array(map(is_hsinternal, dbpath_list))
#is_ibeis_lsit       = np.array(map(is_ibeisdb, dbpath_list))
#is_only_hsdb_list = ltool.and_lists(is_hsdb_list, True - is_ibeis_lsit)
#is_only_hsinternal_list = ltool.and_lists(True - is_hsdb_list, True - is_ibeis_lsit, is_hsinternals_list)
# TEST
#only_hsintern_paths = dbpath_list[is_only_hsinternal_list].tolist()

def get_unconverted_hsdbs(workdir=None):
    if workdir is None:
        workdir = ibeis.sysres.get_workdir()
    dbname_list = os.listdir(workdir)
    dbpath_list = np.array([join(workdir, name) for name in dbname_list])
    is_hsdb_list        = np.array(map(is_hsdb, dbpath_list))
    is_ibs_cvt_list     = np.array(map(is_succesful_convert, dbpath_list))
    if utool.get_flag('--force-hsdb-convert'):
        needs_convert = is_hsdb_list
    else:
        needs_convert =  ltool.and_lists(is_hsdb_list, True - is_ibs_cvt_list)
    needs_convert_hsdb  = dbpath_list[needs_convert].tolist()
    print('NEEDS CONVERSION:')
    print('\n'.join(needs_convert_hsdb))
    return needs_convert_hsdb


def injest_unconverted_hsdbs_in_workdir(workdir=None):
    if workdir is None:
        workdir = ibeis.sysres.get_workdir()
    needs_convert_hsdb = get_unconverted_hsdbs(workdir)

    for hsdb in needs_convert_hsdb:
        try:
            if utool.get_flag('--force-hsdb-convert'):
                ibsfuncs.delete_ibeis_database(hsdb)
            injest_hsdb.convert_hsdb_to_ibeis(hsdb)
        except Exception as ex:
            print(ex)
            raise

if __name__ == '__main__':
    multiprocessing.freeze_support()  # win32
    workdir = sysres.get_workdir()
    injest_unconverted_hsdbs_in_workdir(workdir)
