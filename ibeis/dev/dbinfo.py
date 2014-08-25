# This is not the cleanest module
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import utool
# Science
import six
import numpy as np
from collections import OrderedDict
from utool import util_latex as util_latex
from vtool import keypoint as ktool
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[dbinfo]')


def get_dbinfo(ibs):
    """ Returns dictionary of digestable database information
    Infostr is a string summary of all the stats. Prints infostr in addition to
    returning locals
    """
    # Name Info
    #rrr()
    valid_aids = ibs.get_valid_aids()
    valid_nids = ibs.get_valid_nids()
    valid_gids = ibs.get_valid_gids()

    name_aids_list = ibs.get_name_aids(valid_nids)
    nx2_aids = np.array(name_aids_list)
    unknown_aids = utool.filter_items(valid_aids, ibs.is_aid_unknown(valid_aids))

    gname_list = ibs.get_image_gnames(valid_gids)
    nx2_nRois = np.asarray(list(map(len, nx2_aids)))
    # Seperate singleton / multitons
    multiton_nids  = np.where(nx2_nRois > 1)[0]
    singleton_nids = np.where(nx2_nRois == 1)[0]
    valid_nids      = np.hstack([multiton_nids, singleton_nids])
    num_names_with_gt = len(multiton_nids)
    # Chip Info
    multiton_aids_list = nx2_aids[multiton_nids]
    #print('multiton_nids = %r' % (multiton_nids,))
    #print('multiton_aids_list = %r' % (multiton_aids_list,))
    if len(multiton_aids_list) == 0:
        multiton_aids = np.array([], dtype=np.int)
    else:
        multiton_aids = np.hstack(multiton_aids_list)
    singleton_aids = nx2_aids[singleton_nids]
    multiton_nid2_nannots = list(map(len, multiton_aids_list))
    # Image info
    gpath_list = ibs.get_image_paths(valid_gids)
    #gpaths_incache = utool.list_images(ibs.imgdir, fullpath=True, recursive=True)

    def wh_print_stats(wh_list):
        if len(wh_list) == 0:
            return '{empty}'
        wh_list = np.asarray(wh_list)
        stat_dict = OrderedDict(
            [( 'max', wh_list.max(0)),
             ( 'min', wh_list.min(0)),
             ('mean', wh_list.mean(0)),
             ( 'std', wh_list.std(0))])
        arr2str = lambda var: '[' + (', '.join(list(map(lambda x: '%.1f' % x, var)))) + ']'
        ret = (',\n    '.join(['%r:%s' % (key, arr2str(val)) for key, val in stat_dict.items()]))
        return '{\n    ' + ret + '}'

    print('reading image sizes')
    annotation_bbox_list = ibs.get_annot_bboxes(valid_aids)
    annotation_bbox_arr = np.array(annotation_bbox_list)
    if len(annotation_bbox_arr) == 0:
        annotation_size_list = []
    else:
        annotation_size_list = annotation_bbox_arr[:, 2:4]
    img_size_list  = ibs.get_image_sizes(valid_gids)
    img_size_stats  = wh_print_stats(img_size_list)
    chip_size_stats = wh_print_stats(annotation_size_list)
    multiton_stats  = utool.stats_str(multiton_nid2_nannots)

    # Time stats
    unixtime_list_ = ibs.get_image_unixtime(valid_gids)
    utvalid_list   = [time != -1 for time in unixtime_list_]
    unixtime_list  = utool.filter_items(unixtime_list_, utvalid_list)
    unixtime_statstr = utool.get_timestats_str(unixtime_list)

    # GPS stats
    gps_list_ = ibs.get_image_gps(valid_gids)
    gpsvalid_list = [gps != (-1, -1) for gps in gps_list_]
    gps_list  = utool.filter_items(gps_list_, gpsvalid_list)

    # print
    info_str = '\n'.join([
        ('+--------'),
        ('+ singleton = single sighting'),
        ('+ multiton = multiple sightings'),
        (' DB Info: ' + ibs.get_dbname()),
        (' * #Img   = %d' % len(valid_gids)),
        (' * #Annots = %d' % len(valid_aids)),
        (' * #Names = %d' % len(valid_nids)),
        (' * #Names  (singleton)  = %d' % len(singleton_nids)),
        (' * #Names  (multiton)   = %d' % len(multiton_nids)),
        (' * #Unknown Annots      = %d' % len(unknown_aids)),
        (' * #Annots (multiton)   = %d' % len(multiton_aids)),
        (' * #Annots per Name (multiton) = %s' % (multiton_stats,)),
        (' * #Img with gps        = %d/%d' % (len(gps_list), len(valid_gids))),
        (' * #Img with timestamp  = %d/%d' % (len(unixtime_list), len(valid_gids))),
        (' * #Img time stats      = %s' % (unixtime_statstr,)),
        (' * #Img in dir = %d' % len(gpath_list)),
        (' * Image Size Stats = %s' % (img_size_stats,)),
        #(' * Chip Size Stats = %s' % (chip_size_stats,)),
        ('L--------'),
    ])
    print(utool.indent(info_str, '[dbinfo]'))
    return locals()


def get_keypoint_stats(ibs):
    from utool import util_latex
    #from hsdev import dev_consistency
    #dev_consistency.check_keypoint_consistency(ibs)
    # Keypoint stats
    #ibs.refresh_features()
    from ibeis.control.IBEISControl import IBEISController
    assert(isinstance(ibs, IBEISController))
    valid_aids = np.array(ibs.get_valid_aids())
    cx2_kpts = ibs.get_annot_kpts(valid_aids)
    #cx2_kpts = ibs.feats.cx2_kpts
    # Check cx2_kpts
    cx2_nFeats = list(map(len, cx2_kpts))
    kpts = np.vstack(cx2_kpts)
    print('[dbinfo] --- LaTeX --- ')
    _printopts = np.get_printoptions()
    np.set_printoptions(precision=3)
    scales = ktool.get_scales(kpts)
    scales = np.array(sorted(scales))
    tex_scale_stats = util_latex.latex_mystats(r'kpt scale', scales)
    tex_nKpts       = util_latex.latex_scalar(r'\# kpts', len(kpts))
    tex_kpts_stats  = util_latex.latex_mystats(r'\# kpts/chip', cx2_nFeats)
    print(tex_nKpts)
    print(tex_kpts_stats)
    print(tex_scale_stats)
    np.set_printoptions(**_printopts)
    print('[dbinfo] ---/LaTeX --- ')
    return (tex_nKpts, tex_kpts_stats, tex_scale_stats)


def dbstats(ibs):
    # Chip / Name / Image stats
    dbinfo_locals = get_dbinfo(ibs)
    db_name = ibs.get_dbname()
    #num_images = dbinfo_locals['num_images']
    num_annots = dbinfo_locals['num_annots']
    num_names = len(dbinfo_locals['valid_nids'])
    num_singlenames = len(dbinfo_locals['singleton_nids'])
    num_multinames = len(dbinfo_locals['multiton_nids'])
    num_multiannots = len(dbinfo_locals['multiton_aids'])
    multiton_nid2_nannots = dbinfo_locals['multiton_nid2_nannots']

    #tex_nImage = util_latex.latex_scalar(r'\# images', num_images)
    tex_nChip = util_latex.latex_scalar(r'\# annots', num_annots)
    tex_nName = util_latex.latex_scalar(r'\# names', num_names)
    tex_nSingleName = util_latex.latex_scalar(r'\# singlenames', num_singlenames)
    tex_nMultiName  = util_latex.latex_scalar(r'\# multinames', num_multinames)
    tex_nMultiChip  = util_latex.latex_scalar(r'\# multiannots', num_multiannots)
    tex_multi_stats = util_latex.latex_mystats(r'\# multistats', multiton_nid2_nannots)

    tex_kpts_scale_thresh = util_latex.latex_multicolumn('Scale Threshold (%d %d)' %
                                                              (ibs.cfg.feat_cfg.scale_min,
                                                               ibs.cfg.feat_cfg.scale_max)) + r'\\' + '\n'

    (tex_nKpts, tex_kpts_stats, tex_scale_stats) = get_keypoint_stats(ibs)
    tex_title = util_latex.latex_multicolumn(db_name + ' database statistics') + r'\\' + '\n'
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
    tabular = util_latex.tabular_join(tabular_body_list)
    print('[dev stats]')
    print(tabular)


def cache_memory_stats(ibs, cid_list, fnum=None):
    from utool import util_latex
    print('[dev stats] cache_memory_stats()')
    #kpts_list = ibs.get_annot_kpts(cid_list)
    #desc_list = ibs.get_annot_desc(cid_list)
    #nFeats_list = map(len, kpts_list)
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
    for key, val in six.iteritems(bytes_map):
        key2 = key.replace('bytes', convert_to)
        if isinstance(val, list):
            val2 = [bytes_ / byte_units[convert_to] for bytes_ in val]
            tex_str = util_latex.latex_mystats(key2, val2)
        else:
            val2 = val / byte_units[convert_to]
            tex_str = util_latex.latex_scalar(key2, val2)
        tabular_body_list.append(tex_str)

    tabular = util_latex.tabular_join(tabular_body_list)

    print(tabular)
    util_latex.render(tabular)

    if fnum is None:
        fnum = 0

    return fnum + 1
