# -*- coding: utf-8 -*-
# This is supposed to be pristine, it turns out to be mostly clutter
"""
DEPRICATE MOST OF THIS FILE IN FAVOR OF DOCTEST SCRIPTS
"""
from __future__ import absolute_import, division, print_function
import six  # NOQA
from wbia._devscript import devcmd, devprecmd
from six.moves import zip
from os.path import split, join, expanduser
from wbia.plottool import draw_func2 as df2
import numpy as np
import utool
import vtool.keypoint as ktool
from wbia import sysres
from wbia.other import ibsfuncs
from wbia.dbio import ingest_hsdb

(print, rrr, profile) = utool.inject2(__name__)


def openworkdirs_test():
    """
    problems:
        PZ_DanExt_All
        PZ_DanExt_Test
        GZ_March2012
        Wildebeest_ONLY_MATCHES

    python dev.py --convert --dbdir /raid/work/PZ_Marianne --force-delete
    python dev.py --convert --dbdir /raid/work/SL_Siva --force-delete
    python dev.py --convert --dbdir /raid/work/PZ_SweatwaterSmall --force-delete
    """
    canskip = [
        '/raid/work/NAUT_test2',
        '/raid/work/WD_Siva',
        '/raid/work/PZ_FlankHack',
        '/raid/work/PZ_Mothers',
        '/raid/work/GZ_Foals',
        '/raid/work/PZ_MTEST',
        '/raid/work/GIR_Tanya',
        '/raid/work/GZ_Siva',
        '/raid/work/Wildebeest',
        '/raid/work/sonograms',
        '/raid/work/MISC_Jan12',
        '/raid/work/GZ_Master0',
        '/raid/work/LF_OPTIMIZADAS_NI_V_E',
        '/raid/work/LF_Bajo_bonito',
        '/raid/work/Frogs',
        '/raid/work/GZ_ALL',
        '/raid/work/JAG_Kelly',
        '/raid/work/NAUT_test (copy)',
        '/raid/work/WS_hard',
        '/raid/work/WY_Toads',
        '/raid/work/NAUT_Dan',
        '/raid/work/LF_WEST_POINT_OPTIMIZADAS',
        '/raid/work/Seals',
        '/raid/work/Rhinos_Stewart',
        '/raid/work/Elephants_Stewart',
        '/raid/work/NAUT_test',
    ]
    import wbia
    from wbia.init import sysres
    import os
    import utool as ut  # NOQA
    from os.path import join
    from wbia.dbio import ingest_hsdb
    import wbia.other.dbinfo

    wbia.other.dbinfo.rrr()
    workdir = sysres.get_workdir()
    dbname_list = os.listdir(workdir)
    dbpath_list = [join(workdir, name) for name in dbname_list]
    is_hsdb_list = list(map(ingest_hsdb.is_hsdb, dbpath_list))
    hsdb_list = ut.compress(dbpath_list, is_hsdb_list)
    # is_ibs_cvt_list = np.array(list(map(is_succesful_convert, dbpath_list)))
    regen_cmds = []
    for hsdb_dpath in hsdb_list:
        if hsdb_dpath in canskip:
            continue
        try:
            ibs = wbia.opendb(hsdb_dpath)  # NOQA
            print('Succesfully opened hsdb: ' + hsdb_dpath)
            print(ibs.get_dbinfo_str())
        except Exception as ex:
            ut.printex(ex, 'Failed to convert hsdb: ' + hsdb_dpath)
            regen_cmd = 'python dev.py --convert --dbdir ' + hsdb_dpath
            regen_cmds.append(regen_cmd)
    print('\n'.join(regen_cmds))


@devcmd
def vdd(ibs=None, qaid_list=None):
    utool.view_directory(ibs.get_dbdir())


@devcmd('show')
def show_aids(ibs, qaid_list):
    from wbia.viz import interact

    for aid in qaid_list:
        interact.ishow_chip(ibs, aid, fnum=df2.next_fnum())


@devcmd()
def change_names(ibs, qaid_list):
    """ Test to changes names """
    # next_name = utool.get_argval('--name', str, default='<name>_the_<species>')
    next_name = utool.get_argval('--name', str, default='glob')
    for aid in qaid_list:
        ibs.print_name_table()
        # (nid,) = ibs.add_names((next_name,))
        ibs.set_annot_names(aid, next_name)
        ibs.print_name_table()
        ibs.print_annotation_table()
    # FIXME:
    # new_nid = ibs.get_name_rowids_from_text(next_name, ensure=False)
    # if back is not None:
    # back.select_nid(new_nid)


@devcmd('query')
def query_aids(ibs, qaid_list, daid_list=None):
    """
    CommandLine:
        python dev.py -w --show -t query --db PZ_MTEST --qaid 72

    """
    import wbia

    if daid_list is None:
        daid_list = ibs.get_valid_aids()
    cm_list = ibs.query_chips(qaid_list, daid_list)
    for cm in cm_list:
        assert isinstance(cm, wbia.algo.hots.hots_query_result.QueryResult)
        cm.ishow_top(ibs, fnum=df2.next_fnum(), annot_mode=1, make_figtitle=True)


@devcmd('sver')
def sver_aids(ibs, qaid_list, daid_list=None):
    """
    CommandLine:
        python dev.py -w --show -t sver --db PZ_MTEST --qaid 72
        python dev.py -w --show -t sver --db PZ_MTEST --qaid 1

    """
    from wbia.viz import interact

    if daid_list is None:
        daid_list = ibs.get_valid_aids()
    cm_list = ibs.query_chips(qaid_list, daid_list)
    for cm in cm_list:
        aid2 = cm.get_top_aids()[0]
        interact.ishow_sver(ibs, cm.qaid, aid2, fnum=df2.next_fnum(), annot_mode=1)


@devcmd('listdbs', 'list_dbs')
def list_dbs(*args):
    ibsdb_list = sorted(sysres.get_ibsdb_list())
    print('IBEIS Databases:')
    print('\n'.join(ibsdb_list))


@devcmd('list_hsdbs')
def list_unconverted_hsdbs(*args):
    needs_convert_hsdbs = ingest_hsdb.get_unconverted_hsdbs()
    print('NEEDS CONVERSION:')
    print('\n'.join(needs_convert_hsdbs))


@devcmd('convertall')
def convert_hsdbs(*args):
    ingest_hsdb.ingest_unconverted_hsdbs_in_workdir()


@devcmd
def delete_cache(ibs, *args):
    ibs.delete_cache()


@devcmd
def delete_all_feats(ibs, *args):
    ibsfuncs.delete_all_features(ibs)


@devcmd
def delete_all_chips(ibs, *args):
    ibsfuncs.delete_all_chips(ibs)


@devprecmd('mtest')
def ensure_mtest():
    """
    CommandLine:
        python dev.py -t mtest
    """
    import wbia

    wbia.ensure_pz_mtest()


@devprecmd('nauts')
def ensure_nauts():
    """
    CommandLine:
        python dev.py -t nauts
    """
    import wbia

    wbia.ensure_nauts()


@devprecmd('wds')
def ensure_wilddogs():
    """
    CommandLine:
        python dev.py -t wds
    """
    import wbia

    wbia.ensure_wilddogs()


MOTHERS_VIEWPOINT_EXPORT_PAIRS = [
    [117, 115],
    [72, 70],
    [45, 43],
]

GZ_VIEWPOINT_EXPORT_PAIRS = [
    [495, 559],
    [558, 152],
]


def export(ibs, aid_pairs=None):
    """
    3 - 4 different animals
    2 views of each
    matching keypoint coordinates on each annotation
    """
    if aid_pairs is None:
        if ibs.get_dbname() == 'PZ_MOTHERS':
            aid_pair_list = MOTHERS_VIEWPOINT_EXPORT_PAIRS
        if ibs.get_dbname() == 'GZ_ALL':
            aid_pair_list = GZ_VIEWPOINT_EXPORT_PAIRS
    ibs.update_query_cfg(ratio_thresh=1.6)
    export_path = expanduser('~/Dropbox/Assignments/dataset')
    # utool.view_directory(export_path)
    # MOTHERS EG:
    for aid_pair in aid_pair_list:
        cm_list, qreq_ = ibs.query_chips(aid_pair, aid_pair)
        # wbia.viz.show_qres(ibs, qaid2_qres.values()[1]); df2.iup()
        mrids_list = []
        mkpts_list = []
        for cm in cm_list:
            qaid = cm.qaid
            print('Getting kpts from %r' % qaid)
            # cm.show_top(ibs)
            posrid_list = utool.ensure_iterable(cm.get_classified_pos())
            mrids_list.extend([(qaid, posrid) for posrid in posrid_list])
            mkpts_list.extend(cm.get_matching_keypoints(ibs, posrid_list))

        mkey2_kpts = {}
        for mrids_tup, mkpts_tup in zip(mrids_list, mkpts_list):
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

        for mkeys, mkpts_list in zip(mkeys_list, mkeys_keypoints):
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
                )
                + ', '
                for kp1, kp2 in zip(kpts1_m, kpts2_m)
            ]

            mcpaths_list = ibs.get_annot_chip_fpath(mkeys)
            fnames_list = list(map(lambda x: split(x)[1], mcpaths_list))
            for path in mcpaths_list:
                utool.copy(path, export_path)

            header_lines = [
                '# Exported keypoint matches (might be duplicates matches)',
                '# matching_aids = %r' % (mkey,),
            ]
            header_lines += [
                '# img%d = %r' % (count, fname) for count, fname in enumerate(fnames_list)
            ]
            header_lines += ['# LINE FORMAT: match_pts = [(img1_xy, img2_xy) ... ]']
            header_text = '\n'.join(header_lines)
            match_text = '\n'.join(['match_pts = ['] + match_lines + [']'])
            matchfile_text = '\n'.join([header_text, match_text])
            matchfile_name = 'match_aids(%d,%d).txt' % mkey
            matchfile_path = join(export_path, matchfile_name)
            utool.write_to(matchfile_path, matchfile_text)
            print(header_text)
            print(utool.truncate_str(match_text, maxlen=500))
