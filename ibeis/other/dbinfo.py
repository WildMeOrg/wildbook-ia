# -*- coding: utf-8 -*-
"""
get_dbinfo is probably the only usefull funciton in here
# This is not the cleanest module
"""
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import utool as ut
import six
import numpy as np
import ibeis.constants as const
from collections import OrderedDict
from utool import util_latex as util_latex
import functools
from vtool import keypoint as ktool
print, print_, printDBG, rrr, profile = ut.inject(__name__, '[dbinfo]')


def print_qd_info(ibs, qaid_list, daid_list, verbose=False):
    """
    information for a query/database aid configuration
    """
    bigstr = functools.partial(ut.truncate_str, maxlen=64, truncmsg=' ~TRUNC~ ')
    print('[qd_info] * dbname = %s' % ibs.get_dbname())
    print('[qd_info] * qaid_list = %s' % bigstr(str(qaid_list)))
    print('[qd_info] * daid_list = %s' % bigstr(str(daid_list)))
    print('[qd_info] * len(qaid_list) = %d' % len(qaid_list))
    print('[qd_info] * len(daid_list) = %d' % len(daid_list))
    print('[qd_info] * intersection = %r' % len(list(set(daid_list).intersection(set(qaid_list)))))
    if verbose:
        infokw = dict(with_contrib=False, with_agesex=False, with_header=False, verbose=False)
        d_info_str = get_dbinfo(ibs, aid_list=daid_list, tag='DataInfo', **infokw)['info_str2']
        q_info_str = get_dbinfo(ibs, aid_list=qaid_list, tag='QueryInfo', **infokw)['info_str2']
        print(q_info_str)
        print('\n')
        print(d_info_str)


def get_dbinfo(ibs, verbose=True,
               with_imgsize=False,
               with_bytes=False,
               with_contrib=True,
               with_agesex=False,
               with_header=True,
               short=False,
               tag='dbinfo',
               aid_list=None):
    """

    Returns dictionary of digestable database information
    Infostr is a string summary of all the stats. Prints infostr in addition to
    returning locals

    Args:
        ibs (IBEISController):
        verbose (bool):
        with_imgsize (bool):
        with_bytes (bool):

    Returns:
        dict:

    CommandLine:
        python -m ibeis.other.dbinfo --exec-get_dbinfo:0
        python -m ibeis.other.dbinfo --test-get_dbinfo:1
        python -m ibeis.other.dbinfo --test-get_dbinfo:0 --db NNP_Master3
        python -m ibeis.other.dbinfo --test-get_dbinfo:0 --db PZ_Master1
        python -m ibeis.other.dbinfo --test-get_dbinfo:0 --db GZ_ALL
        python -m ibeis.other.dbinfo --exec-get_dbinfo:0 --db PZ_ViewPoints
        python -m ibeis.other.dbinfo --exec-get_dbinfo:0 --db GZ_Master1

        python -m ibeis.other.dbinfo --exec-get_dbinfo:0 -a ctrl

    Example1:
        >>> # SCRIPT
        >>> from ibeis.other.dbinfo import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> # <HACK FOR FILTERING>
        >>> #from ibeis.experiments import cfghelpers
        >>> #from ibeis.experiments import annotation_configs
        >>> #from ibeis.init import filter_annots
        >>> #named_defaults_dict = ut.dict_take(annotation_configs.__dict__,
        >>> #                                   annotation_configs.TEST_NAMES)
        >>> #named_qcfg_defaults = dict(zip(annotation_configs.TEST_NAMES,
        >>> #                               ut.get_list_column(named_defaults_dict, 'qcfg')))
        >>> #acfg = cfghelpers.parse_argv_cfg(('--annot-filter', '-a'), named_defaults_dict=named_qcfg_defaults, default=None)[0]
        >>> aid_list = ibs.get_valid_aids()
        >>> #aid_list = filter_annots.filter_annots_independent(ibs, aid_list, acfg)
        >>> # </HACK FOR FILTERING>
        >>> kwargs = ut.get_kwdefaults(get_dbinfo)
        >>> kwargs['verbose'] = False
        >>> kwargs['aid_list'] = aid_list
        >>> kwargs = ut.parse_dict_from_argv(kwargs)
        >>> output = get_dbinfo(ibs, **kwargs)
        >>> result = (output['info_str'])
        >>> print(result)

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.other.dbinfo import *  # NOQA
        >>> import ibeis
        >>> verbose = True
        >>> short = True
        >>> #ibs = ibeis.opendb(db='GZ_ALL')
        >>> #ibs = ibeis.opendb(db='PZ_Master0')
        >>> ibs = ibeis.opendb('testdb1')
        >>> assert ibs.get_dbname() == 'testdb1', 'DO NOT DELETE CONTRIBUTORS OF OTHER DBS'
        >>> ibs.delete_contributors(ibs.get_valid_contrib_rowids())
        >>> ibs.delete_empty_nids()
        >>> #ibs = ibeis.opendb(db='PZ_MTEST')
        >>> output = get_dbinfo(ibs, with_contrib=False, verbose=False, short=True)
        >>> result = (output['info_str'])
        >>> print(result)
        +============================
        DB Info:  testdb1
        DB Notes: None
        DB NumContrib: 0
        ----------
        # Names                      = 7
        # Names (unassociated)       = 0
        # Names (singleton)          = 5
        # Names (multiton)           = 2
        ----------
        # Annots                     = 13
        # Annots (unknown)           = 4
        # Annots (singleton)         = 5
        # Annots (multiton)          = 4
        ----------
        # Img                        = 13
        L============================
    """
    # TODO Database size in bytes
    # TODO: encounters, contributors, etc...

    # Basic variables
    request_annot_subset = False
    if aid_list is None:
        valid_aids = ibs.get_valid_aids()
        valid_nids = ibs.get_valid_nids()
        valid_gids = ibs.get_valid_gids()
    else:
        if isinstance(aid_list, str):
            # Hack to get experiment stats on aids
            acfg_name_list = [aid_list]
            print('Specified custom aids via acfgname %s' % (acfg_name_list,))
            from ibeis.experiments import experiment_helpers
            acfg_list, expanded_aids_list = experiment_helpers.get_annotcfg_list(ibs, acfg_name_list)
            aid_list = sorted(list(set(ut.flatten(ut.flatten(expanded_aids_list)))))
            #aid_list =
        if verbose:
            print('Specified %d custom aids' % (len(aid_list,)))
        request_annot_subset = True
        valid_aids = aid_list
        valid_nids = list(set(ibs.get_annot_nids(aid_list, distinguish_unknowns=False)) - {ibs.UNKNOWN_NAME_ROWID})
        valid_gids = list(set(ibs.get_annot_gids(aid_list)))
    #associated_nids = ibs.get_valid_nids(filter_empty=True)  # nids with at least one annotation
    FILTER_HACK = True
    if FILTER_HACK:
        # HUGE HACK - get only images and names with filtered aids
        valid_aids_ = ibs.filter_aids_custom(valid_aids)
        valid_nids_ = ibs.filter_nids_custom(valid_nids)
        valid_gids_ = ibs.filter_gids_custom(valid_gids)
        if verbose:
            print('Filtered %d names' % (len(valid_nids) - len(valid_nids_)))
            print('Filtered %d images' % (len(valid_gids) - len(valid_gids_)))
            print('Filtered %d annots' % (len(valid_aids) - len(valid_aids_)))
        valid_gids = valid_gids_
        valid_nids = valid_nids_
        valid_aids = valid_aids_
        #associated_nids = ut.list_compress(associated_nids, map(any, ibs.unflat_map(ibs.get_annot_custom_filterflags, ibs.get_name_aids(associated_nids))))

    # Image info
    if verbose:
        print('Checking Image Info')
    gx2_aids = ibs.get_image_aids(valid_gids)
    if FILTER_HACK:
        gx2_aids = [ibs.filter_aids_custom(aids) for aids in gx2_aids]  # HACK FOR FILTER
    if request_annot_subset:
        # remove annots not in this subset
        valid_aids_set = set(valid_aids)
        gx2_aids = [list(set(aids).intersection(valid_aids_set)) for aids in gx2_aids]

    gx2_nAnnots = np.array(map(len, gx2_aids))
    image_without_annots = len(np.where(gx2_nAnnots == 0)[0])
    gx2_nAnnots_stats  = ut.get_stats_str(gx2_nAnnots, newlines=True, use_median=True)
    image_reviewed_list = ibs.get_image_reviewed(valid_gids)

    # Name stats
    if verbose:
        print('Checking Name Info')
    nx2_aids = ibs.get_name_aids(valid_nids)
    if FILTER_HACK:
        nx2_aids =  [ibs.filter_aids_custom(aids) for aids in nx2_aids]    # HACK FOR FILTER
    if request_annot_subset:
        # remove annots not in this subset
        valid_aids_set = set(valid_aids)
        nx2_aids = [list(set(aids).intersection(valid_aids_set)) for aids in nx2_aids]
    associated_nids = ut.list_compress(valid_nids, map(len, nx2_aids))

    ibs.check_name_mapping_consistency(nx2_aids)

    # Annot Stats
    # TODO: number of images where chips cover entire image
    # TODO: total image coverage of annotation
    # TODO: total annotation overlap
    """
    ax2_unknown = ibs.is_aid_unknown(valid_aids)
    ax2_nid = ibs.get_annot_name_rowids(valid_aids)
    assert all([nid < 0 if unknown else nid > 0 for nid, unknown in
                zip(ax2_nid, ax2_unknown)]), 'bad annot nid'
    """
    #
    if verbose:
        print('Checking Annot Species')
    unknown_aids = ut.list_compress(valid_aids, ibs.is_aid_unknown(valid_aids))
    species_list = ibs.get_annot_species_texts(valid_aids)
    species2_aids = ut.group_items(valid_aids, species_list)
    species2_nAids = {key: len(val) for key, val in species2_aids.items()}

    if verbose:
        print('Checking Multiton/Singleton Species')
    nx2_nAnnots = np.array(list(map(len, nx2_aids)))
    # Seperate singleton / multitons
    multiton_nxs  = np.where(nx2_nAnnots > 1)[0]
    singleton_nxs = np.where(nx2_nAnnots == 1)[0]
    unassociated_nxs = np.where(nx2_nAnnots == 0)[0]
    assert len(np.intersect1d(singleton_nxs, multiton_nxs)) == 0, 'intersecting names'
    valid_nxs      = np.hstack([multiton_nxs, singleton_nxs])
    num_names_with_gt = len(multiton_nxs)

    # Annot Info
    if verbose:
        print('Checking Annot Info')
    multiton_aids_list = ut.list_take(nx2_aids, multiton_nxs)
    assert len(set(multiton_nxs)) == len(multiton_nxs)
    if len(multiton_aids_list) == 0:
        multiton_aids = np.array([], dtype=np.int)
    else:
        multiton_aids = np.hstack(multiton_aids_list)
        assert len(set(multiton_aids)) == len(multiton_aids), 'duplicate annot'
    singleton_aids = ut.list_take(nx2_aids, singleton_nxs)
    multiton_nid2_nannots = list(map(len, multiton_aids_list))

    # Image size stats
    if with_imgsize:
        if verbose:
            print('Checking ImageSize Info')
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
            ret = (',\n    '.join(['%s:%s' % (key, arr2str(val)) for key, val in stat_dict.items()]))
            return '{\n    ' + ret + '\n}'

        print('reading image sizes')
        # Image size stats
        img_size_list  = ibs.get_image_sizes(valid_gids)
        img_size_stats  = wh_print_stats(img_size_list)

        # Chip size stats
        annotation_bbox_list = ibs.get_annot_bboxes(valid_aids)
        annotation_bbox_arr = np.array(annotation_bbox_list)
        if len(annotation_bbox_arr) == 0:
            annotation_size_list = []
        else:
            annotation_size_list = annotation_bbox_arr[:, 2:4]
        chip_size_stats = wh_print_stats(annotation_size_list)
        imgsize_stat_lines = [
            (' # Img in dir                 = %d' % len(gpath_list)),
            (' Image Size Stats  = %s' % (img_size_stats,)),
            (' * Chip Size Stats = %s' % (chip_size_stats,)),
        ]
    else:
        imgsize_stat_lines = []

    if verbose:
        print('Building Stats String')

    multiton_stats = ut.get_stats_str(multiton_nid2_nannots, newlines=True, use_median=True)

    # Time stats
    unixtime_list = ibs.get_image_unixtime(valid_gids)
    unixtime_list = ut.list_replace(unixtime_list, -1, float('nan'))
    #valid_unixtime_list = [time for time in unixtime_list if time != -1]
    #unixtime_statstr = ibs.get_image_time_statstr(valid_gids)
    if ut.get_argflag('--hackshow-unixtime'):
        show_time_distributions(ibs, unixtime_list)
        ut.show_if_requested()
    unixtime_statstr = ut.get_timestats_str(unixtime_list, newlines=True, full=True)

    # GPS stats
    gps_list_ = ibs.get_image_gps(valid_gids)
    gpsvalid_list = [gps != (-1, -1) for gps in gps_list_]
    gps_list  = ut.list_compress(gps_list_, gpsvalid_list)

    def get_annot_age_stats(aid_list):
        annot_age_months_est_min = ibs.get_annot_age_months_est_min(aid_list)
        annot_age_months_est_max = ibs.get_annot_age_months_est_max(aid_list)
        age_dict = ut.ddict((lambda : 0))
        for min_age, max_age in zip(annot_age_months_est_min, annot_age_months_est_max):
            if (min_age is None or min_age < 12) and max_age < 12:
                age_dict['Infant'] += 1
            elif 12 <= min_age and min_age < 36 and 12 <= max_age and max_age < 36:
                age_dict['Juvenile'] += 1
            elif 36 <= min_age and (36 <= max_age or max_age is None):
                age_dict['Adult'] += 1
            else:
                print('Found UNKNOWN Age: %r, %r' % (min_age, max_age, ))
                age_dict['UNKNOWN'] += 1
        return age_dict

    def get_annot_sex_stats(aid_list):
        annot_sextext_list = ibs.get_annot_sex_texts(aid_list)
        sextext2_aids = ut.group_items(aid_list, annot_sextext_list)
        sex_keys = list(const.SEX_TEXT_TO_INT.keys())
        assert set(sex_keys) >= set(annot_sextext_list), 'bad keys: ' + str(set(annot_sextext_list) - set(sex_keys))
        sextext2_nAnnots = ut.odict([(key, len(sextext2_aids.get(key, []))) for key in sex_keys])
        # Filter 0's
        sextext2_nAnnots = {key: val for key, val in six.iteritems(sextext2_nAnnots) if val != 0}
        return sextext2_nAnnots

    if verbose:
        print('Checking Other Annot Stats')

    qualtext2_nAnnots = ibs.get_annot_qual_stats(valid_aids)
    yawtext2_nAnnots = ibs.get_annot_yaw_stats(valid_aids)
    agetext2_nAnnots = get_annot_age_stats(valid_aids)
    sextext2_nAnnots = get_annot_sex_stats(valid_aids)

    if verbose:
        print('Checking Contrib Stats')

    # Contributor Statistics
    # hack remove colon for image alignment
    def fix_tag_list(tag_list):
        return [None if tag is None else tag.replace(':', ';') for tag in tag_list]
    image_contrib_tags = fix_tag_list(ibs.get_image_contributor_tag(valid_gids))
    annot_contrib_tags = fix_tag_list(ibs.get_annot_image_contributor_tag(valid_aids))
    contrib_tag_to_gids = ut.group_items(valid_gids, image_contrib_tags)
    contrib_tag_to_aids = ut.group_items(valid_aids, annot_contrib_tags)

    contrib_tag_to_qualstats = {key: ibs.get_annot_qual_stats(aids) for key, aids in six.iteritems(contrib_tag_to_aids)}
    contrib_tag_to_viewstats = {key: ibs.get_annot_yaw_stats(aids) for key, aids in six.iteritems(contrib_tag_to_aids)}

    contrib_tag_to_nImages = {key: len(val) for key, val in six.iteritems(contrib_tag_to_gids)}
    contrib_tag_to_nAnnots = {key: len(val) for key, val in six.iteritems(contrib_tag_to_aids)}

    if verbose:
        print('Summarizing')

    # Summarize stats
    num_names = len(valid_nids)
    num_names_unassociated = len(valid_nids) - len(associated_nids)
    num_names_singleton = len(singleton_nxs)
    num_names_multiton =  len(multiton_nxs)

    num_singleton_annots = len(singleton_aids)
    num_multiton_annots = len(multiton_aids)
    num_unknown_annots = len(unknown_aids)
    num_annots = len(valid_aids)

    if with_bytes:
        if verbose:
            print('Checking Disk Space')
        ibsdir_space   = ut.byte_str2(ut.get_disk_space(ibs.get_ibsdir()))
        dbdir_space    = ut.byte_str2(ut.get_disk_space(ibs.get_dbdir()))
        imgdir_space   = ut.byte_str2(ut.get_disk_space(ibs.get_imgdir()))
        cachedir_space = ut.byte_str2(ut.get_disk_space(ibs.get_cachedir()))

    if True:
        if verbose:
            print('Check asserts')
        try:
            bad_aids = np.intersect1d(multiton_aids, unknown_aids)
            _num_names_total_check = num_names_singleton + num_names_unassociated + num_names_multiton
            _num_annots_total_check = num_unknown_annots + num_singleton_annots + num_multiton_annots
            assert len(bad_aids) == 0, 'intersecting multiton aids and unknown aids'
            assert _num_names_total_check == num_names, 'inconsistent num names'
            #if not request_annot_subset:
            # dont check this if you have an annot subset
            assert _num_annots_total_check == num_annots, 'inconsistent num annots'
        except Exception as ex:
            ut.printex(ex, keys=[
                '_num_names_total_check',
                'num_names',
                '_num_annots_total_check',
                'num_annots',
                'num_names_singleton',
                'num_names_multiton',
                'num_unknown_annots',
                'num_multiton_annots',
                'num_singleton_annots',
            ])
            raise

    # Get contributor statistics
    contrib_rowids = ibs.get_valid_contrib_rowids()
    num_contributors = len(contrib_rowids)

    # print
    num_tabs = 5

    def align2(str_):
        return ut.align(str_, ':', ' :')

    def align_dict2(dict_):
        str_ = ut.dict_str(dict_)
        return align2(str_)

    header_block_lines = (
        [('+============================'), ] + (
            [
                ('+ singleton := single sighting'),
                ('+ multiton  := multiple sightings'),
                ('--' * num_tabs),
            ] if not short and with_header else []
        )
    )

    source_block_lines = [
        ('DB Info:  ' + ibs.get_dbname()),
        ('DB Notes: ' + ibs.get_dbnotes()),
        ('DB NumContrib: %d' % num_contributors),
    ]

    bytes_block_lines = [
        ('--' * num_tabs),
        ('DB Bytes: '),
        ('     +- dbdir nBytes:         ' + dbdir_space),
        ('     |  +- _ibsdb nBytes:     ' + ibsdir_space),
        ('     |  |  +-imgdir nBytes:   ' + imgdir_space),
        ('     |  |  +-cachedir nBytes: ' + cachedir_space),
    ] if with_bytes else []

    name_block_lines = [
        ('--' * num_tabs),
        ('# Names                      = %d' % num_names),
        ('# Names (unassociated)       = %d' % num_names_unassociated),
        ('# Names (singleton)          = %d' % num_names_singleton),
        ('# Names (multiton)           = %d' % num_names_multiton),
    ]

    subset_str = '        ' if not request_annot_subset else '(SUBSET)'

    annot_block_lines = [
        ('--' * num_tabs),
        ('# Annots %s            = %d' % (subset_str, num_annots,)),
        ('# Annots (unknown)           = %d' % num_unknown_annots),
        ('# Annots (singleton)         = %d' % num_singleton_annots),
        ('# Annots (multiton)          = %d' % num_multiton_annots),
    ]

    annot_per_basic_block_lines = [
        ('--' * num_tabs),
        ('# Annots per Name (multiton) = %s' % (align2(multiton_stats),)),
        ('# Annots per Image           = %s' % (align2(gx2_nAnnots_stats),)),
        ('# Annots per Species         = %s' % (align_dict2(species2_nAids),)),
    ] if not short else []

    annot_per_qualview_block_lines = [
        None if short else '# Annots per Viewpoint = %s' % align_dict2(yawtext2_nAnnots),
        None if short else '# Annots per Quality = %s' % align_dict2(qualtext2_nAnnots),
    ]

    annot_per_agesex_block_lines = [
        '# Annots per Age = %s' % align_dict2(agetext2_nAnnots),
        '# Annots per Sex = %s' % align_dict2(sextext2_nAnnots),
    ] if not short  and with_agesex else []

    contrib_block_lines = [
        '# Images per contributor       = ' + align_dict2(contrib_tag_to_nImages),
        '# Annots per contributor       = ' + align_dict2(contrib_tag_to_nAnnots),
        '# Quality per contributor      = ' + ut.dict_str(contrib_tag_to_qualstats, sorted_=True),
        '# Viewpoint per contributor    = ' + ut.dict_str(contrib_tag_to_viewstats, sorted_=True),
    ] if with_contrib else []

    img_block_lines = [
        ('--' * num_tabs),
        ('# Img                        = %d' % len(valid_gids)),
        None if short else ('# Img reviewed               = %d' % sum(image_reviewed_list)),
        None if short else ('# Img with gps               = %d' % len(gps_list)),
        #('# Img with timestamp         = %d' % len(valid_unixtime_list)),
        None if short else ('Img Time Stats               = %s' % (align2(unixtime_statstr),)),
    ]

    info_str_lines = (
        header_block_lines +
        bytes_block_lines +
        source_block_lines +
        name_block_lines +
        annot_block_lines +
        annot_per_basic_block_lines +
        annot_per_qualview_block_lines +
        annot_per_agesex_block_lines +
        img_block_lines +
        contrib_block_lines +
        imgsize_stat_lines +
        [('L============================'), ]
    )
    info_str = '\n'.join(ut.filter_Nones(info_str_lines))
    info_str2 = ut.indent(info_str, '[{tag}]'.format(tag=tag))
    if verbose:
        print(info_str2)
    locals_ = locals()
    return locals_


def hackshow_names(ibs, aid_list, fnum=None):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        aid_list (list):

    CommandLine:
        python -m ibeis.other.dbinfo --exec-hackshow_names --show
        python -m ibeis.other.dbinfo --exec-hackshow_names --show --db PZ_Master1

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.other.dbinfo import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> aid_list = ibs.get_valid_aids()
        >>> result = hackshow_names(ibs, aid_list)
        >>> print(result)
        >>> ut.show_if_requested()
    """
    grouped_aids, nid_list = ibs.group_annots_by_name(aid_list)
    grouped_aids = [aids for aids in grouped_aids if len(aids) > 1]
    unixtimes_list = ibs.unflat_map(ibs.get_annot_image_unixtimes_asfloat, grouped_aids)
    yaws_list = ibs.unflat_map(ibs.get_annot_yaws, grouped_aids)
    #markers_list = [[(1, 2, yaw * 360 / (np.pi * 2)) for yaw in yaws] for yaws in yaws_list]

    unixtime_list = ut.flatten(unixtimes_list)
    timemax = np.nanmax(unixtime_list)
    timemin = np.nanmin(unixtime_list)
    timerange = timemax - timemin
    unixtimes_list = [((unixtimes[:] - timemin) / timerange) for unixtimes in unixtimes_list]
    for unixtimes in unixtimes_list:
        num_nan = sum(np.isnan(unixtimes))
        unixtimes[np.isnan(unixtimes)] = np.linspace(-1, -.5, num_nan)
    import plottool as pt
    #ydata_list = [np.arange(len(aids)) for aids in grouped_aids]
    import vtool as vt
    sortx_list = vt.argsort_groups(unixtimes_list, reverse=False)
    #markers_list = ut.list_ziptake(markers_list, sortx_list)
    yaws_list = ut.list_ziptake(yaws_list, sortx_list)
    ydatas_list = vt.ziptake(unixtimes_list, sortx_list)
    #ydatas_list = sortx_list
    #ydatas_list = vt.argsort_groups(unixtimes_list, reverse=False)

    # Sort by num members
    #ydatas_list = ut.list_take(ydatas_list, np.argsort(list(map(len, ydatas_list))))
    xdatas_list = [np.zeros(len(ydatas)) + count for count, ydatas in enumerate(ydatas_list)]
    #markers = ut.flatten(markers_list)
    #yaws = np.array(ut.flatten(yaws_list))
    y_data = np.array(ut.flatten(ydatas_list))
    x_data = np.array(ut.flatten(xdatas_list))
    fnum = pt.ensure_fnum(fnum)
    pt.figure(fnum=fnum)
    ax = pt.gca()

    #unique_yaws, groupxs = vt.group_indices(yaws)

    ax.scatter(x_data, y_data, color=[1, 0, 0], s=1, marker='.')
    #pt.draw_stems(x_data, y_data, marker=markers, setlims=True, linestyle='')
    pt.dark_background()
    ax = pt.gca()
    ax.set_xlim(min(x_data) - .1, max(x_data) + .1)
    ax.set_ylim(min(y_data) - .1, max(y_data) + .1)


def show_image_time_distributions(ibs, gid_list):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object
        gid_list (list):

    CommandLine:
        python -m ibeis.other.dbinfo --exec-show_image_time_distributions --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.other.dbinfo import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> gid_list = ibs.get_valid_gids()
        >>> result = show_image_time_distributions(ibs, gid_list)
        >>> print(result)
        >>> ut.show_if_requested()
    """
    unixtime_list = ibs.get_image_unixtime(gid_list)
    unixtime_list = np.array(unixtime_list, dtype=np.float)
    unixtime_list = ut.list_replace(unixtime_list, -1, float('nan'))
    show_time_distributions(ibs, unixtime_list)


def show_time_distributions(ibs, unixtime_list):
    r"""
    """
    #import vtool as vt
    import plottool as pt
    unixtime_list = np.array(unixtime_list)
    num_nan = np.isnan(unixtime_list).sum()
    num_total = len(unixtime_list)
    unixtime_list = unixtime_list[~np.isnan(unixtime_list)]
    if False:
        from matplotlib import dates as mpldates
        #data_list = list(map(ut.unixtime_to_datetimeobj, unixtime_list))
        n, bins, patches = pt.plt.hist(unixtime_list, 365)
        #n_ = list(map(ut.unixtime_to_datetimeobj, n))
        #bins_ = list(map(ut.unixtime_to_datetimeobj, bins))
        pt.plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
        ax = pt.gca()
        #ax.xaxis.set_major_locator(mpldates.YearLocator())
        #hfmt = mpldates.DateFormatter('%y/%m/%d')
        #ax.xaxis.set_major_formatter(hfmt)
        mpldates.num2date(unixtime_list)
        #pt.gcf().autofmt_xdate()
        #y = pt.plt.normpdf( bins, unixtime_list.mean(), unixtime_list.std())
        #ax.set_xticks(bins_)
        #l = pt.plt.plot(bins_, y, 'k--', linewidth=1.5)
    else:
        pt.draw_time_distribution(unixtime_list)
        #pt.draw_histogram()
        ax = pt.gca()
        ax.set_xlabel('Date')
        ax.set_title('Timestamp distribution of %s. #nan=%d/%d' % (
            ibs.get_dbname_alias(),
            num_nan, num_total))
        pt.gcf().autofmt_xdate()
    if ut.get_argflag('--contextadjust'):
        #pt.adjust_subplots2(left=.08, bottom=.1, top=.9, wspace=.3, hspace=.1)
        pt.adjust_subplots2(use_argv=True)


def latex_dbstats(ibs_list, **kwargs):
    r"""
    Args:
        ibs (IBEISController):  ibeis controller object

    CommandLine:
        python -m ibeis.other.dbinfo --exec-latex_dbstats --dblist testdb1
        python -m ibeis.other.dbinfo --exec-latex_dbstats --dblist testdb1 --show
        python -m ibeis.other.dbinfo --exec-latex_dbstats --dblist PZ_Master0 testdb1 --show
        python -m ibeis.other.dbinfo --exec-latex_dbstats --dblist PZ_Master0 PZ_MTEST GZ_ALL --show
        python -m ibeis.other.dbinfo --test-latex_dbstats --dblist GZ_ALL NNP_MasterGIRM_core --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.other.dbinfo import *  # NOQA
        >>> import ibeis
        >>> db_list = ut.get_argval('--dblist', type_=list, default=['testdb1'])
        >>> ibs_list = [ibeis.opendb(db=db) for db in db_list]
        >>> tabular_str = latex_dbstats(ibs_list)
        >>> tabular_cmd = ut.latex_newcommand(ut.latex_sanatize_command_name('DatabaseInfo'), tabular_str)
        >>> ut.copy_text_to_clipboard(tabular_cmd)
        >>> write_fpath = ut.get_argval('--write', type_=str, default=None)
        >>> if write_fpath is not None:
        >>>     fpath = ut.truepath(write_fpath)
        >>>     text = ut.readfrom(fpath)
        >>>     new_text = ut.replace_between_tags(text, tabular_cmd, '% <DBINFO>', '% </DBINFO>')
        >>>     ut.writeto(fpath, new_text)
        >>> ut.print_code(tabular_cmd, 'latex')
        >>> ut.quit_if_noshow()
        >>> ut.render_latex_text('\\noindent \n' + tabular_str)
    """
    #title = db_name + ' database statistics'
    title = 'Database statistics'
    stat_title = '# Annot per name (multiton)'

    #col_lbls = [
    #    'multiton',
    #    #'singleton',
    #    'total',
    #    'multiton',
    #    'singleton',
    #    'total',
    #]
    key_to_col_lbls = {
        'num_names_multiton':   'multiton',
        'num_names_singleton':  'singleton',
        'num_names':            'total',

        'num_multiton_annots':  'multiton',
        'num_singleton_annots': 'singleton',
        'num_unknown_annots':   'unknown',
        'num_annots':           'total',
    }
    # Structure of columns / multicolumns
    multi_col_keys = [
        ('# Names', (
            'num_names_multiton',
            #'num_names_singleton',
            'num_names',
        )),

        ('# Annots', (
            'num_multiton_annots',
            'num_singleton_annots',
            'num_unknown_annots',
            'num_annots')),
    ]
    #multicol_lbls = [('# Names', 3), ('# Annots', 3)]
    multicol_lbls = [(mcolname, len(mcols)) for mcolname, mcols in multi_col_keys]

    # Flatten column labels
    col_keys = ut.flatten(ut.get_list_column(multi_col_keys, 1))
    col_lbls = ut.dict_take(key_to_col_lbls, col_keys)

    row_lbls   = []
    row_values = []

    #stat_col_lbls = ['max', 'min', 'mean', 'std', 'nMin', 'nMax']
    stat_col_lbls = ['max', 'min', 'mean', 'std', 'median']
    #stat_row_lbls = ['# Annot per Name (multiton)']
    stat_row_lbls = []
    stat_row_values = []

    dbinfo_list = [get_dbinfo(ibs, with_contrib=False, verbose=False) for ibs in ibs_list]

    SINGLE_TABLE = False
    EXTRA = True

    for ibs, dbinfo_locals in zip(ibs_list, dbinfo_list):
        row_ = ut.dict_take(dbinfo_locals, col_keys)
        dbname = ibs.get_dbname_alias()
        row_lbls.append(dbname)
        multiton_annot_stats = ut.get_stats(dbinfo_locals['multiton_nid2_nannots'], use_median=True)
        stat_rows = ut.dict_take(multiton_annot_stats, stat_col_lbls)
        if SINGLE_TABLE:
            row_.extend(stat_rows)
        else:
            stat_row_lbls.append(dbname)
            stat_row_values.append(stat_rows)

        row_values.append(row_)

    CENTERLINE = False
    AS_TABLE = True
    tablekw = dict(astable=AS_TABLE, centerline=CENTERLINE, FORCE_INT=False,
                   precision=2, col_sep='', multicol_sep='|', **kwargs)

    if EXTRA:
        extra_keys = [
            #'species2_nAids',
            'qualtext2_nAnnots',
            'yawtext2_nAnnots',
        ]
        extra_titles = {
            'species2_nAids': 'Annots per Species',
            'qualtext2_nAnnots': 'Annots per Quality',
            'yawtext2_nAnnots': 'Annots per Viewpoint',
        }
        extra_collbls = ut.ddict(list)
        extra_rowvalues = ut.ddict(list)
        extra_tables = ut.ddict(list)

        for ibs, dbinfo_locals in zip(ibs_list, dbinfo_list):
            for key in extra_keys:
                extra_collbls[key] = ut.unique_keep_order2(extra_collbls[key] + list(dbinfo_locals[key].keys()))

        extra_collbls['qualtext2_nAnnots'] = ['excellent', 'good', 'ok', 'poor', 'junk', 'UNKNOWN']
        #extra_collbls['yawtext2_nAnnots'] = ['backleft', 'left', 'frontleft', 'front', 'frontright', 'right', 'backright', 'back', None]
        extra_collbls['yawtext2_nAnnots'] = ['BL', 'L', 'FL', 'F', 'FR', 'R', 'BR', 'B', None]

        for ibs, dbinfo_locals in zip(ibs_list, dbinfo_list):
            for key in extra_keys:
                extra_rowvalues[key].append(ut.dict_take(dbinfo_locals[key], extra_collbls[key], 0))

        qualalias = {'UNKNOWN': None}

        extra_collbls['yawtext2_nAnnots'] = [const.YAWALIAS.get(val, val) for val in extra_collbls['yawtext2_nAnnots']]
        extra_collbls['qualtext2_nAnnots'] = [qualalias.get(val, val) for val in extra_collbls['qualtext2_nAnnots']]

        for key in extra_keys:
            extra_tables[key] = ut.util_latex.make_score_tabular(
                row_lbls, extra_collbls[key], extra_rowvalues[key],
                title=extra_titles[key], col_align='r', **tablekw)

    #tabular_str = util_latex.tabular_join(tabular_body_list)
    if SINGLE_TABLE:
        col_lbls += stat_col_lbls
        multicol_lbls += [(stat_title, len(stat_col_lbls))]

    count_tabular_str = ut.util_latex.make_score_tabular(
        row_lbls, col_lbls, row_values, title=title, multicol_lbls=multicol_lbls, **tablekw)

    #print(row_lbls)

    if SINGLE_TABLE:
        tabular_str = count_tabular_str
    else:
        stat_tabular_str = ut.util_latex.make_score_tabular(
            stat_row_lbls, stat_col_lbls, stat_row_values, title=stat_title,
            col_align='r', **tablekw)

        # Make a table of statistics
        if tablekw['astable']:
            tablesep = '\n%--\n'
        else:
            tablesep = '\\\\\n%--\n'
        if EXTRA:
            tabular_str = tablesep.join([count_tabular_str, stat_tabular_str] + ut.dict_take(extra_tables, extra_keys))
        else:
            tabular_str = tablesep.join([count_tabular_str, stat_tabular_str])

    return tabular_str

    #print('[dev stats]')
    #print(tabular_str)
    # Chip / Name / Image stats
    # num_images = dbinfo_locals['num_images']
    # num_annots = dbinfo_locals['num_annots']
    #num_names = len(dbinfo_locals['valid_nids'])
    #num_singlenames = len(dbinfo_locals['singleton_nxs'])
    #num_multinames = len(dbinfo_locals['multiton_nxs'])
    #num_multiannots = len(dbinfo_locals['multiton_aids'])
    #multiton_nid2_nannots = dbinfo_locals['multiton_nid2_nannots']

    #(num_names, num_names_singleton, num_names_multiton, num_annots, num_singleton_annots, num_multiton_annots) = ut.dict_take(
    #    dbinfo_locals, 'num_names, num_names_singleton, num_names_multiton, num_annots, num_singleton_annots, num_multiton_annots')

    # tex_nImage = util_latex.latex_scalar(r'\# images', num_images)
    # tex_nChip = util_latex.latex_scalar(r'\# annots', num_annots)
    #tex_multi_stats = util_latex.latex_get_stats(r'\# Annots per Name (multiton)', multiton_nid2_nannots)

    #tex_kpts_scale_thresh = util_latex.latex_multicolumn('Scale Threshold (%d %d)' %
    #                                                          (ibs.cfg.feat_cfg.scale_min,
    #                                                           ibs.cfg.feat_cfg.scale_max)) + r'\\' + '\n'

    #(tex_nKpts, tex_kpts_stats, tex_scale_stats) = get_keypoint_stats(ibs)
    #tex_title = util_latex.latex_multicolumn(db_name + ' database statistics') + r'\\' + '\n'
    #tabular_body_list = [
    #    tex_title,
    #    '',
    #    util_latex.latex_scalar(r'\# Names (multiton)',   dbinfo_locals['num_names_multiton']),
    #    util_latex.latex_scalar(r'\# Names (singleton)',  dbinfo_locals['num_names_singleton']),
    #    util_latex.latex_scalar(r'\# Names',              dbinfo_locals['num_names']),
    #    '',
    #    util_latex.latex_scalar(r'\# Annots (multiton)',  dbinfo_locals['num_multiton_annots']),
    #    util_latex.latex_scalar(r'\# Annots (singleton)', dbinfo_locals['num_singleton_annots']),
    #    util_latex.latex_scalar(r'\# Annots',             dbinfo_locals['num_names_singleton']),
    #    #tex_multi_stats,
    #    #'',
    #    #tex_kpts_scale_thresh,
    #    #tex_nKpts,
    #    #tex_kpts_stats,
    #    #tex_scale_stats,
    #]


def get_short_infostr(ibs):
    """ Returns printable database information

    Args:
        ibs (IBEISController):  ibeis controller object

    Returns:
        str: infostr

    CommandLine:
        python -m ibeis.other.dbinfo --test-get_short_infostr

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.other.dbinfo import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> infostr = get_short_infostr(ibs)
        >>> result = str(infostr)
        >>> print(result)
        dbname = 'testdb1'
        num_images = 13
        num_annotations = 13
        num_names = 7
    """
    dbname = ibs.get_dbname()
    #workdir = ut.unixpath(ibs.get_workdir())
    num_images = ibs.get_num_images()
    num_annotations = ibs.get_num_annotations()
    num_names = ibs.get_num_names()
    #workdir = %r
    infostr = ut.codeblock('''
    dbname = %r
    num_images = %r
    num_annotations = %r
    num_names = %r
    ''' % (dbname, num_images, num_annotations, num_names))
    return infostr


def test_name_consistency(ibs):
    """
    Example:
        >>> import ibeis
        >>> ibs = ibeis.opendb(db='PZ_Master0')
        >>> #ibs = ibeis.opendb(db='GZ_ALL')

    """
    from ibeis import ibsfuncs
    import utool as ut
    max_ = -1
    #max_ = 10
    valid_aids = ibs.get_valid_aids()[0:max_]
    valid_nids = ibs.get_valid_nids()[0:max_]
    ax2_nid = ibs.get_annot_name_rowids(valid_aids)
    nx2_aids = ibs.get_name_aids(valid_nids)

    print('len(valid_aids) = %r' % (len(valid_aids),))
    print('len(valid_nids) = %r' % (len(valid_nids),))
    print('len(ax2_nid) = %r' % (len(ax2_nid),))
    print('len(nx2_aids) = %r' % (len(nx2_aids),))

    # annots are grouped by names, so mapping aid back to nid should
    # result in each list having the same value
    _nids_list = ibsfuncs.unflat_map(ibs.get_annot_name_rowids, nx2_aids)
    print(_nids_list[-20:])
    print(nx2_aids[-20:])
    assert all(map(ut.list_allsame, _nids_list))


def get_keypoint_stats(ibs):
    """
    kp info
    """
    # from ut import util_latex
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
    #_printopts = np.get_printoptions()
    #np.set_printoptions(precision=3)
    scales = ktool.get_scales(kpts)
    scales = np.array(sorted(scales))
    tex_scale_stats = util_latex.latex_get_stats(r'kpt scale', scales)
    tex_nKpts       = util_latex.latex_scalar(r'\# kpts', len(kpts))
    tex_kpts_stats  = util_latex.latex_get_stats(r'\# kpts/chip', cx2_nFeats)
    print(tex_nKpts)
    print(tex_kpts_stats)
    print(tex_scale_stats)
    #np.set_printoptions(**_printopts)
    print('[dbinfo] ---/LaTeX --- ')
    return (tex_nKpts, tex_kpts_stats, tex_scale_stats)


def cache_memory_stats(ibs, cid_list, fnum=None):
    from ut import util_latex
    print('[dev stats] cache_memory_stats()')
    #kpts_list = ibs.get_annot_kpts(cid_list)
    #desc_list = ibs.get_annot_vecs(cid_list)
    #nFeats_list = map(len, kpts_list)
    gx_list = np.unique(ibs.cx2_gx(cid_list))

    bytes_map = {
        'chip dbytes': [ut.file_bytes(fpath) for fpath in ibs.get_rchip_path(cid_list)],
        'img dbytes':  [ut.file_bytes(gpath) for gpath in ibs.gx2_gname(gx_list, full=True)],
        #'flann dbytes':  ut.file_bytes(flann_fpath),
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


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.other.dbinfo
        python -m ibeis.other.dbinfo --allexamples
        python -m ibeis.other.dbinfo --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
