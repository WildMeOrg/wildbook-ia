# This is supposed to be pristine, it turns out to be mostly clutter
from __future__ import absolute_import, division, print_function
from _devscript import devcmd
from itertools import izip
from os.path import split, join, expanduser
from plottool import draw_func2 as df2
import numpy as np
import utool
import vtool.keypoint as ktool
from ibeis import sysres
from ibeis.dev import ibsfuncs
from ibeis.viz import interact
from ibeis.injest import injest_hsdb


@devcmd
def vdd(ibs=None, qrid_list=None):
    utool.view_directory(ibs.get_dbdir())


@devcmd('show')
def show_rids(ibs, qrid_list):
    for rid in qrid_list:
        interact.ishow_chip(ibs, rid, fnum=df2.next_fnum())


@devcmd()
def change_names(ibs, qrid_list):
    #new_name = utool.get_arg('--name', str, default='<name>_the_<species>')
    new_name = utool.get_arg('--name', str, default='glob')
    for rid in qrid_list:
        ibs.print_name_table()
        #(nid,) = ibs.add_names((new_name,))
        ibs.set_roi_names(rid, new_name)
        ibs.print_name_table()
        ibs.print_roi_table()
    # FIXME:
    #new_nid = ibs.get_name_nids(new_name, ensure=False)
    #if back is not None:
        #back.select_nid(new_nid)


@devcmd('query')
def query_rids(ibs, qrid_list):
    qrid2_qres = ibs.query_database(qrid_list)
    for qrid in qrid_list:
        qres = qrid2_qres[qrid]
        interact.ishow_qres(ibs, qres, fnum=df2.next_fnum(), annote_mode=1)
    return qrid2_qres


@devcmd('sver')
def sver_rids(ibs, qrid_list):
    qrid2_qres = ibs.query_database(qrid_list)
    for qrid in qrid_list:
        qres = qrid2_qres[qrid]
        rid2 = qres.get_top_rids()[0]
        interact.ishow_sver(ibs, qrid, rid2, fnum=df2.next_fnum(), annote_mode=1)
    return qrid2_qres


@devcmd('cfg')
def printcfg(ibs, qrid_list):
    ibs.cfg.printme3()
    print(ibs.cfg.query_cfg.get_uid())


@devcmd('list_dbs')
def list_dbs(*args):
    ibsdb_list = sysres.get_ibsdb_list()
    print('IBEIS Databases:')
    print('\n'.join(ibsdb_list))


@devcmd('list_hsdbs')
def list_unconverted_hsdbs(*args):
    needs_convert_hsdbs = injest_hsdb.get_unconverted_hsdbs()
    print('NEEDS CONVERSION:')
    print('\n'.join(needs_convert_hsdbs))


@devcmd('convert')
def convert_hsdbs(*args):
    injest_hsdb.injest_unconverted_hsdbs_in_workdir()


@devcmd
def delete_all_feats(ibs, *args):
    ibsfuncs.delete_all_features(ibs)


@devcmd
def delete_all_chips(ibs, *args):
    ibsfuncs.delete_all_chips(ibs)


MOTHERS_VIEWPOINT_EXPORT_PAIRS = [
    [117, 115],
    [72,   70],
    [45,   43],
]

GZ_VIEWPOINT_EXPORT_PAIRS = [
    [495, 559],
    [558, 152],
]


def export(ibs, rid_pairs=None):
    """
    3 - 4 different animals
    2 views of each
    matching keypoint coordinates on each roi
    """
    if rid_pairs is None:
        if ibs.get_dbname() == 'PZ_MOTHERS':
            rid_pair_list = MOTHERS_VIEWPOINT_EXPORT_PAIRS
        if ibs.get_dbname() == 'GZ_ALL':
            rid_pair_list = GZ_VIEWPOINT_EXPORT_PAIRS
    ibs.update_cfg(ratio_thresh=1.6)
    export_path = expanduser('~/Dropbox/Assignments/dataset')
    #utool.view_directory(export_path)
    # MOTHERS EG:
    for rid_pair in rid_pair_list:
        qrid2_qres = ibs.query_intra_encounter(rid_pair)
        #ibeis.viz.show_qres(ibs, qrid2_qres.values()[1]); df2.iup()
        mrids_list = []
        mkpts_list = []
        for qrid, qres in qrid2_qres.iteritems():
            print('Getting kpts from %r' % qrid)
            #qres.show_top(ibs)
            posrid_list = utool.ensure_iterable(qres.get_classified_pos())
            mrids_list.extend([(qrid, posrid) for posrid in posrid_list])
            mkpts_list.extend(qres.get_matching_keypoints(ibs, posrid_list))

        mkey2_kpts = {}
        for mrids_tup, mkpts_tup in izip(mrids_list, mkpts_list):
            assert len(mrids_tup) == 2, 'must be a match tuple'
            mrids_ = np.array(mrids_tup)
            sortx = mrids_.argsort()
            mrids_ = mrids_[sortx]
            mkpts_ = np.array(mkpts_tup)[sortx]
            if sortx[0] == 0:
                pass
            mkey = tuple(mrids_.tolist())
            try:
                kpts_list = mkey2_kpts[mkey]
                print('append to mkey=%r' % (mkey,))
            except KeyError:
                print('new mkey=%r' % (mkey,))
                kpts_list = []
            kpts_list.append(mkpts_)
            mkey2_kpts[mkey] = kpts_list

        mkeys_list = mkey2_kpts.keys()
        mkeys_keypoints = mkey2_kpts.values()

        for mkeys, mkpts_list in izip(mkeys_list, mkeys_keypoints):
            print(mkeys)
            print(len(kpts_list))
            kpts1_m = np.vstack([mkpts[0] for mkpts in mkpts_list])
            kpts2_m = np.vstack([mkpts[1] for mkpts in mkpts_list])
            match_lines = [
                repr(
                    (
                        tuple(kp1[ktool.LOC_DIMS].tolist()),
                        tuple(kp2[ktool.LOC_DIMS].tolist()),
                    )
                ) + ', '
                for kp1, kp2 in izip(kpts1_m, kpts2_m)]

            mcpaths_list = ibs.get_roi_cpaths(mkeys)
            fnames_list = map(lambda x: split(x)[1], mcpaths_list)
            for path in mcpaths_list:
                utool.copy(path, export_path)

            header_lines = ['# Exported keypoint matches (might be duplicates matches)',
                            '# matching_rids = %r' % (mkey,)]
            header_lines += ['# img%d = %r' % (count, fname) for count, fname in enumerate(fnames_list)]
            header_lines += ['# LINE FORMAT: match_pts = [(img1_xy, img2_xy) ... ]']
            header_text = '\n'.join(header_lines)
            match_text  = '\n'.join(['match_pts = ['] + match_lines + [']'])
            matchfile_text = '\n'.join([header_text, match_text])
            matchfile_name = ('match_rids(%d,%d).txt' % mkey)
            matchfile_path = join(export_path, matchfile_name)
            utool.write_to(matchfile_path, matchfile_text)
            print(header_text)
            print(utool.truncate_str(match_text, maxlen=500))
