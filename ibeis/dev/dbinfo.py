# This is not the cleanest module
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import utool
# Science
from PIL import Image
import numpy as np
from collections import OrderedDict
from utool import util_latex as latex_formatter
from vtool import keypoint as ktool
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[dbinfo]')


def db_info(ibs):
    # Name Info
    rid_list = ibs.get_valid_rids()
    nid_list = ibs.get_valid_nids()
    rids_in_nids_list = ibs.get_rids_in_nids(nid_list)
    unknown_rids = ibs.get_rids_in_nids(ibs.UNKNOWN_NID)

    gid_list = ibs.get_valid_gids()
    gname_list = ibs.get_image_gnames(gid_list)
    cx2_gx = ibs.tables.cx2_gx

    nx2_cxs    = np.array(ibs.get_nx2_cxs())
    nx2_nRois = np.array(map(len, nx2_cxs))
    num_uniden = len(unknown_rids)
    # Seperate singleton / multitons
    multiton_nxs,  = np.where(nx2_nRois > 1)
    singleton_nxs, = np.where(nx2_nRois == 1)
    valid_nxs      = np.hstack([multiton_nxs, singleton_nxs])
    num_names_with_gt = len(multiton_nxs)
    # Chip Info
    cx2_roi = ibs.tables.cx2_roi
    multiton_cx_lists = nx2_cxs[multiton_nxs]
    multiton_cxs = np.hstack(multiton_cx_lists)
    singleton_cxs = nx2_cxs[singleton_nxs]
    multiton_nx2_nchips = map(len, multiton_cx_lists)
    valid_cxs = ibs.get_valid_cxs()
    num_chips = len(valid_cxs)
    # Image info
    num_images = len(gid_list)
    gpath_list = utool.list_images(ibs.imgdir, fullpath=True)

    def wh_print_stats(wh_list):
        if len(wh_list) == 0:
            return '{empty}'
        stat_dict = OrderedDict(
            [( 'max', wh_list.max(0)),
             ( 'min', wh_list.min(0)),
             ('mean', wh_list.mean(0)),
             ( 'std', wh_list.std(0))])
        arr2str = lambda var: '[' + (', '.join(map(lambda x: '%.1f' % x, var))) + ']'
        ret = (',\n    '.join(['%r:%s' % (key, arr2str(val)) for key, val in stat_dict.items()]))
        return '{\n    ' + ret + '}'

    def get_img_size_list(gpath_list):
        ret = []
        for img_fpath in gpath_list:
            try:
                size = Image.open(img_fpath).size
                ret.append(size)
            except Exception as ex:
                print(repr(ex))
                pass
        return ret

    print('reading image sizes')
    if len(cx2_roi) == 0:
        roi_size_list = []
    else:
        roi_size_list = cx2_roi[:, 2:4]
    img_size_list  = np.array(get_img_size_list(gpath_list))
    img_size_stats  = wh_print_stats(img_size_list)
    chip_size_stats = wh_print_stats(roi_size_list)
    multiton_stats  = utool.printable_mystats(multiton_nx2_nchips)

    num_names = len(valid_nxs)
    # print
    info_str = '\n'.join([
        (' DB Info: ' + ibs.get_db_name()),
        (' * #Img   = %d' % num_images),
        (' * #Chips = %d' % num_chips),
        (' * #Names = %d' % len(valid_nxs)),
        (' * #Unidentified Chips = %d' % num_uniden),
        (' * #Singleton Names    = %d' % len(singleton_nxs)),
        (' * #Multiton Names     = %d' % len(multiton_nxs)),
        (' * #Multiton Chips     = %d' % len(multiton_cxs)),
        (' * Chips per Multiton Names = %s' % (multiton_stats,)),
        (' * #Img in dir = %d' % len(gpath_list)),
        (' * Image Size Stats = %s' % (img_size_stats,)),
        (' * Chip Size Stats = %s' % (chip_size_stats,)), ])
    print(info_str)
    return locals()


def get_keypoint_stats(ibs):
    from hscom import latex_formater as pytex
    from hsdev import dev_consistency
    dev_consistency.check_keypoint_consistency(ibs)
    # Keypoint stats
    ibs.refresh_features()
    cx2_kpts = ibs.feats.cx2_kpts
    # Check cx2_kpts
    cx2_nFeats = map(len, cx2_kpts)
    kpts = np.vstack(cx2_kpts)
    print('[dbinfo] --- LaTeX --- ')
    _printopts = np.get_printoptions()
    np.set_printoptions(precision=3)
    scales = ktool.get_scales(kpts)
    scales = np.array(sorted(scales))
    tex_scale_stats = pytex.latex_mystats(r'kpt scale', scales)
    tex_nKpts       = pytex.latex_scalar(r'\# kpts', len(kpts))
    tex_kpts_stats  = pytex.latex_mystats(r'\# kpts/chip', cx2_nFeats)
    print(tex_nKpts)
    print(tex_kpts_stats)
    print(tex_scale_stats)
    np.set_printoptions(**_printopts)
    print('[dbinfo] ---/LaTeX --- ')
    return (tex_nKpts, tex_kpts_stats, tex_scale_stats)


def dbstats(ibs):
    # Chip / Name / Image stats
    dbinfo_locals = db_info(ibs)
    db_name = ibs.get_db_name()
    #num_images = dbinfo_locals['num_images']
    num_chips = dbinfo_locals['num_chips']
    num_names = len(dbinfo_locals['valid_nxs'])
    num_singlenames = len(dbinfo_locals['singleton_nxs'])
    num_multinames = len(dbinfo_locals['multiton_nxs'])
    num_multichips = len(dbinfo_locals['multiton_cxs'])
    multiton_nx2_nchips = dbinfo_locals['multiton_nx2_nchips']

    #tex_nImage = latex_formater.latex_scalar(r'\# images', num_images)
    tex_nChip = latex_formatter.latex_scalar(r'\# chips', num_chips)
    tex_nName = latex_formatter.latex_scalar(r'\# names', num_names)
    tex_nSingleName = latex_formatter.latex_scalar(r'\# singlenames', num_singlenames)
    tex_nMultiName  = latex_formatter.latex_scalar(r'\# multinames', num_multinames)
    tex_nMultiChip  = latex_formatter.latex_scalar(r'\# multichips', num_multichips)
    tex_multi_stats = latex_formatter.latex_mystats(r'\# multistats', multiton_nx2_nchips)

    tex_kpts_scale_thresh = latex_formatter.latex_multicolumn('Scale Threshold (%d %d)' %
                                                             (ibs.prefs.feat_cfg.scale_min,
                                                              ibs.prefs.feat_cfg.scale_max)) + r'\\' + '\n'

    (tex_nKpts, tex_kpts_stats, tex_scale_stats) = get_keypoint_stats(ibs)
    tex_title = latex_formatter.latex_multicolumn(db_name + ' database statistics') + r'\\' + '\n'
    tabular_body_list = [
        tex_title,
        tex_nChip,
        tex_nName,
        tex_nSingleName,
        tex_nMultiName,
        tex_nMultiChip,
        tex_multi_stats,
        '',
        tex_kpts_scale_thresh,
        tex_nKpts,
        tex_kpts_stats,
        tex_scale_stats,
    ]
    tabular = latex_formatter.tabular_join(tabular_body_list)
    print('[dev stats]')
    print(tabular)


def cache_memory_stats(ibs, cid_list, fnum=None):
    from hscom import latex_formater
    print('[dev stats] cache_memory_stats()')
    kpts_list = ibs.get_kpts(cid_list)
    desc_list = ibs.get_desc(cid_list)
    nFeats_list = map(len, kpts_list)
    gx_list = np.unique(ibs.cx2_gx(cid_list))

    bytes_map = {
        'chip dbytes': [utool.file_bytes(fpath) for fpath in ibs.get_rchip_path(cid_list)],
        'img dbytes':  [utool.file_bytes(gpath) for gpath in ibs.gx2_gname(gx_list, full=True)],
        #'flann dbytes':  utool.file_bytes(flann_fpath),
    }

    byte_units = {
        'GB': 2 ** 30,
        'MB': 2 ** 20,
        'KB': 2 ** 10,
    }

    tabular_body_list = [
    ]

    convert_to = 'KB'
    for key, val in bytes_map.iteritems():
        key2 = key.replace('bytes', convert_to)
        if isinstance(val, list):
            val2 = [bytes_ / byte_units[convert_to] for bytes_ in val]
            tex_str = latex_formater.latex_mystats(key2, val2)
        else:
            val2 = val / byte_units[convert_to]
            tex_str = latex_formater.latex_scalar(key2, val2)
        tabular_body_list.append(tex_str)

    tabular = latex_formater.tabular_join(tabular_body_list)

    print(tabular)
    latex_formater.render(tabular)

    if fnum is None:
        fnum = 0

    return fnum + 1
