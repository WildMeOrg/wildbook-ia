# This is supposed to be pristine, it turns out to be mostly clutter
from __future__ import absolute_import, division, print_function
import six
from _devscript import devcmd
from six.moves import zip
from os.path import split, join, expanduser
from plottool import draw_func2 as df2
import numpy as np
import utool
import vtool.keypoint as ktool
from ibeis import sysres
from ibeis.dev import ibsfuncs
from ibeis.viz import interact
from ibeis.ingest import ingest_hsdb


@devcmd
def vdd(ibs=None, qaid_list=None):
    utool.view_directory(ibs.get_dbdir())


@devcmd('show')
def show_aids(ibs, qaid_list):
    for aid in qaid_list:
        interact.ishow_chip(ibs, aid, fnum=df2.next_fnum())


@devcmd()
def change_names(ibs, qaid_list):
    #next_name = utool.get_arg('--name', str, default='<name>_the_<species>')
    next_name = utool.get_arg('--name', str, default='glob')
    for aid in qaid_list:
        ibs.print_name_table()
        #(nid,) = ibs.add_names((next_name,))
        ibs.set_annot_names(aid, next_name)
        ibs.print_name_table()
        ibs.print_annotation_table()
    # FIXME:
    #new_nid = ibs.get_name_nids(next_name, ensure=False)
    #if back is not None:
        #back.select_nid(new_nid)


@devcmd('query')
def query_aids(ibs, qaid_list):
    qaid2_qres = ibs.query_all(qaid_list)
    for qaid in qaid_list:
        qres = qaid2_qres[qaid]
        interact.ishow_qres(ibs, qres, fnum=df2.next_fnum(), annote_mode=1)
    return qaid2_qres


@devcmd('sver')
def sver_aids(ibs, qaid_list):
    qaid2_qres = ibs.query_all(qaid_list)
    for qaid in qaid_list:
        qres = qaid2_qres[qaid]
        aid2 = qres.get_top_aids()[0]
        interact.ishow_sver(ibs, qaid, aid2, fnum=df2.next_fnum(), annote_mode=1)
    return qaid2_qres


@devcmd('cfg')
def printcfg(ibs, qaid_list):
    ibs.cfg.printme3()
    print(ibs.cfg.query_cfg.get_cfgstr())


@devcmd('list_dbs')
def list_dbs(*args):
    ibsdb_list = sysres.get_ibsdb_list()
    print('IBEIS Databases:')
    print('\n'.join(ibsdb_list))


@devcmd('list_hsdbs')
def list_unconverted_hsdbs(*args):
    needs_convert_hsdbs = ingest_hsdb.get_unconverted_hsdbs()
    print('NEEDS CONVERSION:')
    print('\n'.join(needs_convert_hsdbs))


@devcmd('convert')
def convert_hsdbs(*args):
    ingest_hsdb.ingest_unconverted_hsdbs_in_workdir()


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
    #utool.view_directory(export_path)
    # MOTHERS EG:
    for aid_pair in aid_pair_list:
        qaid2_qres = ibs.query_intra_encounter(aid_pair)
        #ibeis.viz.show_qres(ibs, qaid2_qres.values()[1]); df2.iup()
        mrids_list = []
        mkpts_list = []
        for qaid, qres in six.iteritems(qaid2_qres):
            print('Getting kpts from %r' % qaid)
            #qres.show_top(ibs)
            posrid_list = utool.ensure_iterable(qres.get_classified_pos())
            mrids_list.extend([(qaid, posrid) for posrid in posrid_list])
            mkpts_list.extend(qres.get_matching_keypoints(ibs, posrid_list))

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
                ) + ', '
                for kp1, kp2 in zip(kpts1_m, kpts2_m)]

            mcpaths_list = ibs.get_annot_cpaths(mkeys)
            fnames_list = list(map(lambda x: split(x)[1], mcpaths_list))
            for path in mcpaths_list:
                utool.copy(path, export_path)

            header_lines = ['# Exported keypoint matches (might be duplicates matches)',
                            '# matching_aids = %r' % (mkey,)]
            header_lines += ['# img%d = %r' % (count, fname) for count, fname in enumerate(fnames_list)]
            header_lines += ['# LINE FORMAT: match_pts = [(img1_xy, img2_xy) ... ]']
            header_text = '\n'.join(header_lines)
            match_text  = '\n'.join(['match_pts = ['] + match_lines + [']'])
            matchfile_text = '\n'.join([header_text, match_text])
            matchfile_name = ('match_aids(%d,%d).txt' % mkey)
            matchfile_path = join(export_path, matchfile_name)
            utool.write_to(matchfile_path, matchfile_text)
            print(header_text)
            print(utool.truncate_str(match_text, maxlen=500))
