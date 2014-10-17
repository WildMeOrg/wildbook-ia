"""
python -c "import doctest, ibeis.dev.dbinfo; print(doctest.testmod(ibeis.dev.dbinfo))"

"""
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


def test_name_consistency(ibs):
    """
    Example:
        >>> import ibeis
        >>> ibs = ibeis.opendb(db='PZ_Master0')  #doctest: +ELLIPSIS
        >>> #ibs = ibeis.opendb(db='GZ_ALL')  #doctest: +ELLIPSIS

    """
    from ibeis import ibsfuncs
    max_ = -1
    #max_ = 10
    valid_aids = ibs.get_valid_aids()[0:max_]
    valid_nids = ibs.get_valid_nids()[0:max_]
    ax2_nid = ibs.get_annot_nids(valid_aids)
    nx2_aids = ibs.get_name_aids(valid_nids)

    print('len(valid_aids) = %r' % (len(valid_aids),))
    print('len(valid_nids) = %r' % (len(valid_nids),))
    print('len(ax2_nid) = %r' % (len(ax2_nid),))
    print('len(nx2_aids) = %r' % (len(nx2_aids),))

    # annots are grouped by names, so mapping aid back to nid should
    # result in each list having the same value
    _nids_list = ibsfuncs.unflat_map(ibs.get_annot_nids, nx2_aids)
    assert all(map(utool.list_allsame, _nids_list))


def get_dbinfo(ibs, verbose=True):
    """ Returns dictionary of digestable database information
    Infostr is a string summary of all the stats. Prints infostr in addition to
    returning locals

    >>> from ibeis.dev.dbinfo import *  # NOQA
    >>> from ibeis.dev import dbinfo
    >>> import ibeis
    >>> verbose = True
    >>> #ibs = ibeis.opendb(db='GZ_ALL')  #doctest: +ELLIPSIS
    >>> ibs = ibeis.opendb(db='PZ_Mothers')  #doctest: +ELLIPSIS
    >>> output = dbinfo.get_dbinfo(ibs, verbose=False)
    >>> print(utool.dict_str(output))
    >>> print(output['info_str'])
    66w+mdzw!n%3+i6h

    #>>> print(utool.hashstr(repr(output)))

    """
    # TODO Database size in bytes

    # Basic variables
    valid_aids = ibs.get_valid_aids()
    valid_nids = ibs.get_valid_nids()
    valid_gids = ibs.get_valid_gids()

    # Image info
    gname_list = ibs.get_image_gnames(valid_gids)
    gx2_aids = ibs.get_image_aids(valid_gids)
    gx2_nAnnots = np.array(map(len, gx2_aids))
    image_without_annots = len(np.where(gx2_nAnnots == 0)[0])
    gx2_nAnnots_stats  = utool.get_stats_str(gx2_nAnnots)
    image_reviewed_list = ibs.get_image_reviewed(valid_gids)

    # Name stats
    nx2_aids = np.array(ibs.get_name_aids(valid_nids))

    # Annot Stats
    # TODO: number of images where chips cover entire image
    # TODO: total image coverage of annotation
    # TODO: total annotation overlap
    """
    ax2_unknown = ibs.is_aid_unknown(valid_aids)
    ax2_nid = ibs.get_annot_nids(valid_aids)
    assert all([nid < 0 if unknown else nid > 0 for nid, unknown in
                zip(ax2_nid, ax2_unknown)]), 'bad annot nid'
    """
    #
    unknown_aids = utool.filter_items(valid_aids, ibs.is_aid_unknown(valid_aids))
    species_list = ibs.get_annot_species(valid_aids)
    species2_aids = utool.group_items(valid_aids, species_list)
    species2_nAids = {key: len(val) for key, val in species2_aids.items()}

    nx2_nAnnots = np.array(list(map(len, nx2_aids)))
    # Seperate singleton / multitons
    multiton_nxs  = np.where(nx2_nAnnots > 1)[0]
    singleton_nxs = np.where(nx2_nAnnots == 1)[0]
    assert len(np.intersect1d(singleton_nxs, multiton_nxs)) == 0, 'intersecting names'
    valid_nxs      = np.hstack([multiton_nxs, singleton_nxs])
    num_names_with_gt = len(multiton_nxs)

    # DEBUGGING CODE
    try:
        from ibeis import ibsfuncs
        _nids_list = ibsfuncs.unflat_map(ibs.get_annot_nids, nx2_aids)
        assert all(map(utool.list_allsame, _nids_list))
    except Exception as ex:
        # THESE SHOULD BE CONSISTENT BUT THEY ARE NOT!!?
        #name_annots = [ibs.get_annot_nids(aids) for aids in nx2_aids]
        bad = 0
        good = 0
        huh = 0
        for nx, aids in enumerate(nx2_aids):
            nids = ibs.get_annot_nids(aids)
            if np.all(np.array(nids) > 0):
                print(nids)
                if utool.list_allsame(nids):
                    good += 1
                else:
                    huh += 1
            else:
                bad += 1
        utool.printex(ex, keys=['good', 'bad', 'huh'])

    # Annot Info
    multiton_aids_list = nx2_aids[multiton_nxs]
    assert len(set(multiton_nxs)) == len(multiton_nxs)
    if len(multiton_aids_list) == 0:
        multiton_aids = np.array([], dtype=np.int)
    else:
        multiton_aids = np.hstack(multiton_aids_list)
        assert len(set(multiton_aids)) == len(multiton_aids), 'duplicate annot'
    singleton_aids = nx2_aids[singleton_nxs]
    multiton_nid2_nannots = list(map(len, multiton_aids_list))

    # Image size stats
    gpath_list = ibs.get_image_paths(valid_gids)
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
    multiton_stats  = utool.get_stats_str(multiton_nid2_nannots)

    # Time stats
    unixtime_list_ = ibs.get_image_unixtime(valid_gids)
    utvalid_list   = [time != -1 for time in unixtime_list_]
    unixtime_list  = utool.filter_items(unixtime_list_, utvalid_list)
    unixtime_statstr = utool.get_timestats_str(unixtime_list)

    # GPS stats
    gps_list_ = ibs.get_image_gps(valid_gids)
    gpsvalid_list = [gps != (-1, -1) for gps in gps_list_]
    gps_list  = utool.filter_items(gps_list_, gpsvalid_list)

    ibsdir_space = utool.byte_str2(utool.get_disk_space(ibs.get_ibsdir()))
    dbdir_space  = utool.byte_str2(utool.get_disk_space(ibs.get_dbdir()))
    imgdir_space  = utool.byte_str2(utool.get_disk_space(ibs.get_imgdir()))
    cachedir_space  = utool.byte_str2(utool.get_disk_space(ibs.get_cachedir()))

    # Summarize stats
    num_names = len(valid_nids)
    num_names_singleton = len(singleton_nxs)
    num_names_multiton =  len(multiton_nxs)

    num_multiton_annots = len(multiton_aids)
    num_unknown_annots = len(unknown_aids)
    num_annots = len(valid_aids)

    try:
        bad_aids = np.intersect1d(multiton_aids, unknown_aids)
        assert len(bad_aids) == 0, 'intersecting multiton aids and unknown aids'
        assert num_names_singleton + num_names_multiton == num_names, 'inconsistent num names'
        assert num_unknown_annots + num_multiton_annots == num_annots, 'inconsistent num annots'
    except Exception as ex:
        utool.printex(ex)

    # print
    info_str = '\n'.join([
        ('+============================'),
        ('+ singleton = single sighting'),
        ('+ multiton = multiple sightings'),
        (' DB Info: ' + ibs.get_dbname()),
        (' DB Notes: ' + ibs.get_dbnotes()),
        (' DB Bytes: '),
        (' +- dbdir nBytes:         ' + dbdir_space),
        (' |  +- _ibsdb nBytes:     ' + ibsdir_space),
        (' |  |  +-imgdir nBytes:   ' + imgdir_space),
        (' |  |  +-cachedir nBytes: ' + cachedir_space),
        (' * #Img   = %d' % len(valid_gids)),
        (' * #Annots = %d' % num_annots),
        (' *  -- -- --'),
        (' * #Names = %d' % num_names),
        (' * #Names  (singleton)  = %d' % num_names_singleton),
        (' * #Names  (multiton)   = %d' % num_names_multiton),
        (' *  -- -- --'),
        (' * #Unknown Annots      = %d' % num_unknown_annots),
        (' * #Annots (multiton)   = %d' % num_multiton_annots),
        (' * #Annots per Name (multiton) = %s' % (multiton_stats,)),
        (' * #Annots per Image    = %s' % (gx2_nAnnots_stats,)),
        (' *  -- -- --'),
        (' * #Img reviewed        = %d/%d' % (sum(image_reviewed_list), len(valid_gids))),
        (' * #Img with gps        = %d/%d' % (len(gps_list), len(valid_gids))),
        (' * #Img with timestamp  = %d/%d' % (len(unixtime_list), len(valid_gids))),
        (' * #Img time stats      = %s' % (unixtime_statstr,)),
        (' * #Img in dir = %d' % len(gpath_list)),
        (' * Image Size Stats = %s' % (img_size_stats,)),
        (' * # Annots per Species = %s' % (utool.dict_str(species2_nAids),)),
        #(' * Chip Size Stats = %s' % (chip_size_stats,)),
        ('L============================'),
    ])
    if verbose:
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
    tex_scale_stats = util_latex.latex_get_stats(r'kpt scale', scales)
    tex_nKpts       = util_latex.latex_scalar(r'\# kpts', len(kpts))
    tex_kpts_stats  = util_latex.latex_get_stats(r'\# kpts/chip', cx2_nFeats)
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
    # num_images = dbinfo_locals['num_images']
    # num_annots = dbinfo_locals['num_annots']
    num_names = len(dbinfo_locals['valid_nids'])
    num_singlenames = len(dbinfo_locals['singleton_nxs'])
    num_multinames = len(dbinfo_locals['multiton_nxs'])
    num_multiannots = len(dbinfo_locals['multiton_aids'])
    multiton_nid2_nannots = dbinfo_locals['multiton_nid2_nannots']

    # tex_nImage = util_latex.latex_scalar(r'\# images', num_images)
    # tex_nChip = util_latex.latex_scalar(r'\# annots', num_annots)
    tex_nName = util_latex.latex_scalar(r'\# names', num_names)
    tex_nSingleName = util_latex.latex_scalar(r'\# singlenames', num_singlenames)
    tex_nMultiName  = util_latex.latex_scalar(r'\# multinames', num_multinames)
    tex_nMultiChip  = util_latex.latex_scalar(r'\# multiannots', num_multiannots)
    tex_multi_stats = util_latex.latex_get_stats(r'\# multistats', multiton_nid2_nannots)

    tex_kpts_scale_thresh = util_latex.latex_multicolumn('Scale Threshold (%d %d)' %
                                                              (ibs.cfg.feat_cfg.scale_min,
                                                               ibs.cfg.feat_cfg.scale_max)) + r'\\' + '\n'

    (tex_nKpts, tex_kpts_stats, tex_scale_stats) = get_keypoint_stats(ibs)
    tex_title = util_latex.latex_multicolumn(db_name + ' database statistics') + r'\\' + '\n'
    tabular_body_list = [
        tex_title,
        # tex_nChip,
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
            tex_str = util_latex.latex_get_stats(key2, val2)
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
