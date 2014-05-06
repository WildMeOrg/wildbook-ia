from _devscript import devcmd
from itertools import izip
from os.path import split, join
from plottool import draw_func2 as df2
import numpy as np
import utool
import vtool.keypoint as ktool
from ibeis.dev import ibsfuncs
from ibeis.viz import interact


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
        ibs.set_roi_props((rid,), 'name', (new_name,))
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


@devcmd('hsdbs')
def list_hsdbs(*args):
    from ibeis.injest.injest_my_hotspotter_dbs import get_unconverted_hsdbs
    get_unconverted_hsdbs()


@devcmd('convert')
def convert_hsdbs(*args):
    from ibeis.injest.injest_my_hotspotter_dbs import injest_unconverted_hsdbs_in_workdir
    injest_unconverted_hsdbs_in_workdir()


@devcmd
def delete_all_feats(ibs, *args):
    ibsfuncs.delete_all_features(ibs)


@devcmd
def delete_all_chips(ibs, *args):
    ibsfuncs.delete_all_chips(ibs)


def export(ibs, rid_list=[], nid_list=[], gid_list=[]):
    """
    3 - 4 different animals
    2 views of each
    matching keypoint coordinates on each roi
    """
    # MOTHERS EG:
    enctr1_rids = [162, 163]  # Mothers encounter 1
    qrid2_qres = ibs.query_intra_encounter(enctr1_rids)
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
        mkpts = utool.flatten(mkpts_list)
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

        export_path = r'C:\Users\jon.crall\Dropbox\Assignments\dataset'

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
        utool.view_directory(export_path)
