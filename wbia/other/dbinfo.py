# -*- coding: utf-8 -*-
"""
get_dbinfo is probably the only usefull funciton in here
# This is not the cleanest module
"""
import collections
import functools

# TODO: ADD COPYRIGHT TAG
import logging
from os.path import abspath, join, split

import matplotlib.pyplot as plt
import numpy as np
import utool as ut

from wbia import constants as const

print, rrr, profile = ut.inject2(__name__)
logger = logging.getLogger('wbia')


def print_qd_info(ibs, qaid_list, daid_list, verbose=False):
    """
    SeeAlso:
        ibs.print_annotconfig_stats(qaid_list, daid_list)

    information for a query/database aid configuration
    """
    bigstr = functools.partial(ut.truncate_str, maxlen=64, truncmsg=' ~TRUNC~ ')
    logger.info('[qd_info] * dbname = %s' % ibs.get_dbname())
    logger.info('[qd_info] * qaid_list = %s' % bigstr(str(qaid_list)))
    logger.info('[qd_info] * daid_list = %s' % bigstr(str(daid_list)))
    logger.info('[qd_info] * len(qaid_list) = %d' % len(qaid_list))
    logger.info('[qd_info] * len(daid_list) = %d' % len(daid_list))
    logger.info(
        '[qd_info] * intersection = %r'
        % len(list(set(daid_list).intersection(set(qaid_list))))
    )
    if verbose:
        infokw = dict(
            with_contrib=False, with_agesex=False, with_header=False, verbose=False
        )
        d_info_str = get_dbinfo(ibs, aid_list=daid_list, tag='DataInfo', **infokw)[
            'info_str2'
        ]
        q_info_str = get_dbinfo(ibs, aid_list=qaid_list, tag='QueryInfo', **infokw)[
            'info_str2'
        ]
        logger.info(q_info_str)
        logger.info('\n')
        logger.info(d_info_str)


def get_dbinfo(
    ibs,
    verbose=True,
    with_imgsize=True,
    with_bytes=True,
    with_contrib=True,
    with_agesex=True,
    with_header=True,
    with_reviews=True,
    with_ggr=False,
    with_ca=False,
    with_map=False,
    short=False,
    tag='dbinfo',
    aid_list=None,
    aids=None,
    gmt_offset=3.0,
):
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

    SeeAlso:
        python -m wbia.other.ibsfuncs --exec-get_annot_stats_dict --db PZ_PB_RF_TRAIN --use-hist=True --old=False --per_name_vpedge=False
        python -m wbia.other.ibsfuncs --exec-get_annot_stats_dict --db PZ_PB_RF_TRAIN --all

    CommandLine:
        python -m wbia.other.dbinfo --exec-get_dbinfo:0
        python -m wbia.other.dbinfo --test-get_dbinfo:1
        python -m wbia.other.dbinfo --test-get_dbinfo:0 --db NNP_Master3
        python -m wbia.other.dbinfo --test-get_dbinfo:0 --db PZ_Master1
        python -m wbia.other.dbinfo --test-get_dbinfo:0 --db GZ_ALL
        python -m wbia.other.dbinfo --exec-get_dbinfo:0 --db PZ_ViewPoints
        python -m wbia.other.dbinfo --exec-get_dbinfo:0 --db GZ_Master1

        python -m wbia.other.dbinfo --exec-get_dbinfo:0 --db LF_Bajo_bonito -a default
        python -m wbia.other.dbinfo --exec-get_dbinfo:0 --db DETECT_SEATURTLES -a default --readonly

        python -m wbia.other.dbinfo --exec-get_dbinfo:0 -a ctrl
        python -m wbia.other.dbinfo --exec-get_dbinfo:0 -a default:minqual=ok,require_timestamp=True --dbdir ~/lev/media/danger/LEWA
        python -m wbia.other.dbinfo --exec-get_dbinfo:0 -a default:minqual=ok,require_timestamp=True --dbdir ~/lev/media/danger/LEWA --loadbackup=0

        python -m wbia.other.dbinfo --exec-get_dbinfo:0 -a default: --dbdir ~/lev/media/danger/LEWA
        python -m wbia.other.dbinfo --exec-get_dbinfo:0 -a default: --dbdir ~/lev/media/danger/LEWA --loadbackup=0

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.other.dbinfo import *  # NOQA
        >>> import wbia
        >>> defaultdb = 'testdb1'
        >>> ibs, aid_list = wbia.testdata_aids(defaultdb, a='default:minqual=ok,view=primary,view_ext1=1')
        >>> kwargs = ut.get_kwdefaults(get_dbinfo)
        >>> kwargs['verbose'] = False
        >>> kwargs['aid_list'] = aid_list
        >>> kwargs = ut.parse_dict_from_argv(kwargs)
        >>> output = get_dbinfo(ibs, **kwargs)
        >>> result = (output['info_str'])
        >>> print(result)
        >>> #ibs = wbia.opendb(defaultdb='testdb1')
        >>> # <HACK FOR FILTERING>
        >>> #from wbia.expt import cfghelpers
        >>> #from wbia.expt import annotation_configs
        >>> #from wbia.init import filter_annots
        >>> #named_defaults_dict = ut.dict_take(annotation_configs.__dict__,
        >>> #                                   annotation_configs.TEST_NAMES)
        >>> #named_qcfg_defaults = dict(zip(annotation_configs.TEST_NAMES,
        >>> #                               ut.get_list_column(named_defaults_dict, 'qcfg')))
        >>> #acfg = cfghelpers.parse_argv_cfg(('--annot-filter', '-a'), named_defaults_dict=named_qcfg_defaults, default=None)[0]
        >>> #aid_list = ibs.get_valid_aids()
        >>> # </HACK FOR FILTERING>

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.other.dbinfo import *  # NOQA
        >>> import wbia
        >>> verbose = True
        >>> short = True
        >>> #ibs = wbia.opendb(db='GZ_ALL')
        >>> #ibs = wbia.opendb(db='PZ_Master0')
        >>> ibs = wbia.opendb('testdb1')
        >>> assert ibs.get_dbname() == 'testdb1', 'DO NOT DELETE CONTRIBUTORS OF OTHER DBS'
        >>> ibs.delete_contributors(ibs.get_valid_contributor_rowids())
        >>> ibs.delete_empty_nids()
        >>> #ibs = wbia.opendb(db='PZ_MTEST')
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
    # TODO: occurrence, contributors, etc...
    if aids is not None:
        aid_list = aids

    # Basic variables
    request_annot_subset = False
    _input_aid_list = aid_list  # NOQA

    if aid_list is None:
        valid_aids = ibs.get_valid_aids()
    else:
        if isinstance(aid_list, str):
            # Hack to get experiment stats on aids
            acfg_name_list = [aid_list]
            logger.info('Specified custom aids via acfgname {}'.format(acfg_name_list))
            from wbia.expt import experiment_helpers

            acfg_list, expanded_aids_list = experiment_helpers.get_annotcfg_list(
                ibs, acfg_name_list
            )
            aid_list = sorted(list(set(ut.flatten(ut.flatten(expanded_aids_list)))))
        if verbose:
            logger.info('Specified %d custom aids' % (len(aid_list)))
        request_annot_subset = True
        valid_aids = aid_list

    def get_dates(ibs, gid_list):
        unixtime_list = ibs.get_image_unixtime2(gid_list)
        unixtime_list = [unixtime + (gmt_offset * 60 * 60) for unixtime in unixtime_list]
        datetime_list = [
            ut.unixtime_to_datetimestr(unixtime) if unixtime is not None else 'UNKNOWN'
            for unixtime in unixtime_list
        ]
        date_str_list = [value[:10] for value in datetime_list]
        return date_str_list

    if with_ggr:
        request_annot_subset = True
        valid_gids = list(set(ibs.get_annot_gids(valid_aids)))
        date_str_list = get_dates(ibs, valid_gids)
        flag_list = [
            value in ['2016/01/30', '2016/01/31', '2018/01/27', '2018/01/28']
            for value in date_str_list
        ]
        valid_gids = ut.compress(valid_gids, flag_list)
        ggr_aids = set(ut.flatten(ibs.get_image_aids(valid_gids)))
        valid_aids = sorted(list(set(valid_aids) & ggr_aids))

    valid_nids = list(
        set(ibs.get_annot_nids(valid_aids, distinguish_unknowns=False))
        - {const.UNKNOWN_NAME_ROWID}
    )
    valid_gids = list(set(ibs.get_annot_gids(valid_aids)))
    # valid_rids = ibs._get_all_review_rowids()
    valid_rids = []
    valid_rids += ibs.get_review_rowids_from_aid1(valid_aids)
    valid_rids += ibs.get_review_rowids_from_aid2(valid_aids)
    valid_rids = ut.flatten(valid_rids)
    valid_rids = list(set(valid_rids))

    num_all_total_reviews = len(valid_rids)

    aids_tuple = ibs.get_review_aid_tuple(valid_rids)
    flag_list = []
    for aid_tuple in aids_tuple:
        aid1, aid2 = aid_tuple
        flag = aid1 in valid_aids and aid2 in valid_aids
        flag_list.append(flag)
    valid_rids = ut.compress(valid_rids, flag_list)

    # associated_nids = ibs.get_valid_nids(filter_empty=True)  # nids with at least one annotation
    valid_images = ibs.images(valid_gids)
    valid_annots = ibs.annots(valid_aids)

    # Image info
    if verbose:
        logger.info('Checking Image Info')
    gx2_aids = valid_images.aids
    if request_annot_subset:
        # remove annots not in this subset
        valid_aids_set = set(valid_aids)
        gx2_aids = [list(set(aids_).intersection(valid_aids_set)) for aids_ in gx2_aids]

    gx2_nAnnots = np.array(list(map(len, gx2_aids)))
    image_without_annots = len(np.where(gx2_nAnnots == 0)[0])
    gx2_nAnnots_stats = ut.repr4(
        ut.get_stats(gx2_nAnnots, use_median=True), nl=0, precision=2, si=True
    )
    image_reviewed_list = ibs.get_image_reviewed(valid_gids)

    # Name stats
    if verbose:
        logger.info('Checking Name Info')
    nx2_aids = ibs.get_name_aids(valid_nids)
    if request_annot_subset:
        # remove annots not in this subset
        valid_aids_set = set(valid_aids)
        nx2_aids = [list(set(aids_).intersection(valid_aids_set)) for aids_ in nx2_aids]
    associated_nids = ut.compress(valid_nids, list(map(len, nx2_aids)))

    ibs.check_name_mapping_consistency(nx2_aids)

    # Occurrence Info
    def compute_annot_occurrence_ids(ibs, aid_list, config):
        import utool as ut

        from wbia.algo.preproc import preproc_occurrence

        gid_list = ibs.get_annot_gids(aid_list)
        gid2_aids = ut.group_items(aid_list, gid_list)
        flat_imgsetids, flat_gids = preproc_occurrence.wbia_compute_occurrences(
            ibs, gid_list, config=config, verbose=False
        )
        occurid2_gids = ut.group_items(flat_gids, flat_imgsetids)
        occurid2_aids = {
            oid: ut.flatten(ut.take(gid2_aids, gids))
            for oid, gids in occurid2_gids.items()
        }
        return occurid2_aids

    nids = ibs.get_annot_nids(valid_aids)
    nid2_annotxs = ut.ddict(set)
    for aid, nid in zip(valid_aids, nids):
        if nid >= 0:
            nid2_annotxs[nid].add(aid)

    occurence_config = {'use_gps': True, 'seconds_thresh': 10 * 60}
    occurid2_aids = compute_annot_occurrence_ids(ibs, valid_aids, config=occurence_config)

    aid2_occurxs = ut.ddict(set)
    occurid2_aids_named = ut.ddict(set)
    occurid2_nids = ut.ddict(set)
    for occurx, aids in occurid2_aids.items():
        nids = ibs.get_annot_nids(aids)
        for aid, nid in zip(aids, nids):
            if nid >= 0:
                aid2_occurxs[aid].add(occurx)
                occurid2_aids_named[occurx].add(aid)
                occurid2_nids[occurx].add(nid)

    # assert sorted(set(list(map(len, aid2_occurxs.values())))) == [1]

    occur_nids = ibs.unflat_map(ibs.get_annot_nids, occurid2_aids.values())
    occur_unique_nids = [ut.unique(nids) for nids in occur_nids]
    nid2_occurxs = ut.ddict(set)
    for occurx, nids in enumerate(occur_unique_nids):
        for nid in nids:
            if nid >= 0:
                nid2_occurxs[nid].add(occurx)

    name_annot_stats = ut.get_stats(
        list(map(len, nid2_annotxs.values())), use_median=True, use_sum=True
    )
    occurence_annot_stats = ut.get_stats(
        list(map(len, occurid2_aids_named.values())), use_median=True, use_sum=True
    )
    occurence_encounter_stats = ut.get_stats(
        list(map(len, occurid2_nids.values())), use_median=True, use_sum=True
    )
    annot_encounter_stats = ut.get_stats(
        list(map(len, nid2_occurxs.values())), use_median=True, use_sum=True
    )

    if verbose:
        logger.info('Checking Annot Species')
    unknown_annots = valid_annots.compress(ibs.is_aid_unknown(valid_annots))
    species_list = valid_annots.species_texts
    species2_annots = valid_annots.group_items(valid_annots.species_texts)
    species2_nAids = {key: len(val) for key, val in species2_annots.items()}

    if verbose:
        logger.info('Checking Multiton/Singleton Species')
    nx2_nAnnots = np.array(list(map(len, nx2_aids)))
    # Seperate singleton / multitons
    multiton_nxs = np.where(nx2_nAnnots > 1)[0]
    singleton_nxs = np.where(nx2_nAnnots == 1)[0]
    unassociated_nxs = np.where(nx2_nAnnots == 0)[0]
    assert len(np.intersect1d(singleton_nxs, multiton_nxs)) == 0, 'intersecting names'
    valid_nxs = np.hstack([multiton_nxs, singleton_nxs])
    num_names_with_gt = len(multiton_nxs)

    # Annot Info
    if verbose:
        logger.info('Checking Annot Info')
    multiton_aids_list = ut.take(nx2_aids, multiton_nxs)
    assert len(set(multiton_nxs)) == len(multiton_nxs)
    if len(multiton_aids_list) == 0:
        multiton_aids = np.array([], dtype=np.int)
    else:
        multiton_aids = np.hstack(multiton_aids_list)
        assert len(set(multiton_aids)) == len(multiton_aids), 'duplicate annot'
    singleton_aids = ut.take(nx2_aids, singleton_nxs)
    multiton_nid2_nannots = list(map(len, multiton_aids_list))

    # Image size stats
    if with_imgsize:
        if verbose:
            logger.info('Checking ImageSize Info')
        gpath_list = ibs.get_image_paths(valid_gids)

        def wh_print_stats(wh_list):
            if len(wh_list) == 0:
                return '{empty}'
            wh_list = np.asarray(wh_list)
            stat_dict = collections.OrderedDict(
                [
                    ('max', wh_list.max(0)),
                    ('min', wh_list.min(0)),
                    ('mean', wh_list.mean(0)),
                    ('std', wh_list.std(0)),
                ]
            )

            def arr2str(var):
                return '[' + (', '.join(list(map(lambda x: '%.1f' % x, var)))) + ']'

            ret = ',\n    '.join(
                ['{}:{}'.format(key, arr2str(val)) for key, val in stat_dict.items()]
            )
            return '{\n    ' + ret + '\n}'

        logger.info('reading image sizes')
        # Image size stats
        img_size_list = ibs.get_image_sizes(valid_gids)
        img_size_stats = wh_print_stats(img_size_list)

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
            (' Image Size Stats  = {}'.format(img_size_stats)),
            (' * Chip Size Stats = {}'.format(chip_size_stats)),
        ]
    else:
        imgsize_stat_lines = []

    if verbose:
        logger.info('Building Stats String')

    multiton_stats = ut.repr3(
        ut.get_stats(multiton_nid2_nannots, use_median=True), nl=0, precision=2, si=True
    )

    # Time stats
    unixtime_list = valid_images.unixtime2
    unixtime_list = [unixtime + (gmt_offset * 60 * 60) for unixtime in unixtime_list]

    # valid_unixtime_list = [time for time in unixtime_list if time != -1]
    # unixtime_statstr = ibs.get_image_time_statstr(valid_gids)
    if ut.get_argflag('--hackshow-unixtime'):
        show_time_distributions(ibs, unixtime_list)
        ut.show_if_requested()

    unixtime_statstr = ut.repr3(ut.get_timestats_dict(unixtime_list, full=True), si=True)

    date_str_list = get_dates(ibs, valid_gids)
    ggr_dates_stats = ut.dict_hist(date_str_list)

    # GPS stats
    gps_list_ = ibs.get_image_gps(valid_gids)
    gpsvalid_list = [gps != (-1, -1) for gps in gps_list_]
    gps_list = ut.compress(gps_list_, gpsvalid_list)

    if with_map:

        def plot_kenya(ibs, ax, gps_list=[], focus=False, focus2=False, margin=0.1):
            import geopandas
            import pandas as pd
            import shapely
            import utool as ut

            if focus2:
                focus = True

            world = geopandas.read_file(
                geopandas.datasets.get_path('naturalearth_lowres')
            )
            africa = world[world.continent == 'Africa']
            kenya = africa[africa.name == 'Kenya']

            cities = geopandas.read_file(
                geopandas.datasets.get_path('naturalearth_cities')
            )
            nairobi = cities[cities.name == 'Nairobi']

            kenya.plot(ax=ax, color='white', edgecolor='black')

            path_dict = ibs.compute_ggr_path_dict()
            meru = path_dict['County Meru']

            for key in path_dict:
                path = path_dict[key]

                polygon = shapely.geometry.Polygon(path.vertices[:, ::-1])
                gdf = geopandas.GeoDataFrame([1], geometry=[polygon], crs=world.crs)

                if key.startswith('County'):
                    if 'Meru' in key:
                        gdf.plot(ax=ax, color=(1, 0, 0, 0.2), edgecolor='red')
                    else:
                        gdf.plot(ax=ax, color='grey', edgecolor='black')
                if focus:
                    if key.startswith('Land Tenure'):
                        gdf.plot(ax=ax, color=(1, 0, 0, 0.0), edgecolor='blue')

            if focus2:
                flag_list = []
                for gps in gps_list:
                    flag = meru.contains_point(gps)
                    flag_list.append(flag)
                gps_list = ut.compress(gps_list, flag_list)

            df = pd.DataFrame(
                {
                    'Latitude': ut.take_column(gps_list, 0),
                    'Longitude': ut.take_column(gps_list, 1),
                }
            )
            gdf = geopandas.GeoDataFrame(
                df, geometry=geopandas.points_from_xy(df.Longitude, df.Latitude)
            )
            gdf.plot(ax=ax, color='red')

            min_lat, min_lon = gdf.min()
            max_lat, max_lon = gdf.max()
            dom_lat = max_lat - min_lat
            dom_lon = max_lon - min_lon
            margin_lat = dom_lat * margin
            margin_lon = dom_lon * margin
            min_lat -= margin_lat
            min_lon -= margin_lon
            max_lat += margin_lat
            max_lon += margin_lon

            polygon = shapely.geometry.Polygon(
                [
                    [min_lon, min_lat],
                    [min_lon, max_lat],
                    [max_lon, max_lat],
                    [max_lon, min_lat],
                ]
            )
            gdf = geopandas.GeoDataFrame([1], geometry=[polygon], crs=world.crs)
            gdf.plot(ax=ax, color=(1, 0, 0, 0.0), edgecolor='blue')

            nairobi.plot(ax=ax, marker='*', color='black', markersize=500)

            ax.grid(False, which='major')
            ax.grid(False, which='minor')
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])

            if focus:
                ax.set_autoscalex_on(False)
                ax.set_autoscaley_on(False)
                ax.set_xlim([min_lon, max_lon])
                ax.set_ylim([min_lat, max_lat])

        fig = plt.figure(figsize=(30, 30), dpi=400)

        ax = plt.subplot(131)
        plot_kenya(ibs, ax, gps_list)
        ax = plt.subplot(132)
        plot_kenya(ibs, ax, gps_list, focus=True)
        ax = plt.subplot(133)
        plot_kenya(ibs, ax, gps_list, focus2=True)

        plt.savefig('map.png', bbox_inches='tight')

    def get_annot_age_stats(aid_list):
        annot_age_months_est_min = ibs.get_annot_age_months_est_min(aid_list)
        annot_age_months_est_max = ibs.get_annot_age_months_est_max(aid_list)
        age_dict = ut.ddict(lambda: 0)
        for min_age, max_age in zip(annot_age_months_est_min, annot_age_months_est_max):
            if max_age is None:
                max_age = min_age
            if min_age is None:
                min_age = max_age
            if max_age is None and min_age is None:
                logger.info('Found UNKNOWN Age: {!r}, {!r}'.format(min_age, max_age))
                age_dict['UNKNOWN'] += 1
            elif (min_age is None or min_age < 12) and max_age < 12:
                age_dict['Infant'] += 1
            elif 12 <= min_age and min_age < 36 and 12 <= max_age and max_age < 36:
                age_dict['Juvenile'] += 1
            elif 36 <= min_age and (max_age is None or 36 <= max_age):
                age_dict['Adult'] += 1
        return age_dict

    def get_annot_sex_stats(aid_list):
        annot_sextext_list = ibs.get_annot_sex_texts(aid_list)
        sextext2_aids = ut.group_items(aid_list, annot_sextext_list)
        sex_keys = list(ibs.const.SEX_TEXT_TO_INT.keys())
        assert set(sex_keys) >= set(annot_sextext_list), 'bad keys: ' + str(
            set(annot_sextext_list) - set(sex_keys)
        )
        sextext2_nAnnots = ut.odict(
            [(key, len(sextext2_aids.get(key, []))) for key in sex_keys]
        )
        # Filter 0's
        sextext2_nAnnots = {key: val for key, val in sextext2_nAnnots.items() if val != 0}
        return sextext2_nAnnots

    def get_annot_qual_stats(ibs, aid_list):
        annots = ibs.annots(aid_list)
        qualtext2_nAnnots = ut.order_dict_by(
            ut.map_vals(len, annots.group_items(annots.quality_texts)),
            list(ibs.const.QUALITY_TEXT_TO_INT.keys()),
        )
        return qualtext2_nAnnots

    def get_annot_viewpoint_stats(ibs, aid_list):
        annots = ibs.annots(aid_list)
        viewcode2_nAnnots = ut.order_dict_by(
            ut.map_vals(len, annots.group_items(annots.viewpoint_code)),
            list(ibs.const.VIEW.CODE_TO_INT.keys()) + [None],
        )
        return viewcode2_nAnnots

    if verbose:
        logger.info('Checking Other Annot Stats')

    qualtext2_nAnnots = get_annot_qual_stats(ibs, valid_aids)
    viewcode2_nAnnots = get_annot_viewpoint_stats(ibs, valid_aids)
    agetext2_nAnnots = get_annot_age_stats(valid_aids)
    sextext2_nAnnots = get_annot_sex_stats(valid_aids)

    if verbose:
        logger.info('Checking Contrib Stats')

    # Contributor Statistics
    # hack remove colon for image alignment
    def fix_tag_list(tag_list):
        return [None if tag is None else tag.replace(':', ';') for tag in tag_list]

    image_contributor_tags = fix_tag_list(ibs.get_image_contributor_tag(valid_gids))
    annot_contributor_tags = fix_tag_list(ibs.get_annot_image_contributor_tag(valid_aids))
    contributor_tag_to_gids = ut.group_items(valid_gids, image_contributor_tags)
    contributor_tag_to_aids = ut.group_items(valid_aids, annot_contributor_tags)

    contributor_tag_to_qualstats = {
        key: get_annot_qual_stats(ibs, aids)
        for key, aids in contributor_tag_to_aids.items()
    }
    contributor_tag_to_viewstats = {
        key: get_annot_viewpoint_stats(ibs, aids)
        for key, aids in contributor_tag_to_aids.items()
    }

    contributor_tag_to_nImages = {
        key: len(val) for key, val in contributor_tag_to_gids.items()
    }
    contributor_tag_to_nAnnots = {
        key: len(val) for key, val in contributor_tag_to_aids.items()
    }

    if verbose:
        logger.info('Summarizing')

    # Summarize stats
    num_names = len(valid_nids)
    num_names_unassociated = len(valid_nids) - len(associated_nids)
    num_names_singleton = len(singleton_nxs)
    num_names_multiton = len(multiton_nxs)

    num_singleton_annots = len(singleton_aids)
    num_multiton_annots = len(multiton_aids)
    num_unknown_annots = len(unknown_annots)
    num_annots = len(valid_aids)

    if with_bytes:
        if verbose:
            logger.info('Checking Disk Space')
        ibsdir_space = ut.byte_str2(ut.get_disk_space(ibs.get_ibsdir()))
        dbdir_space = ut.byte_str2(ut.get_disk_space(ibs.get_dbdir()))
        imgdir_space = ut.byte_str2(ut.get_disk_space(ibs.get_imgdir()))
        cachedir_space = ut.byte_str2(ut.get_disk_space(ibs.get_cachedir()))

    if True:
        if verbose:
            logger.info('Check asserts')
        try:
            bad_aids = np.intersect1d(multiton_aids, unknown_annots)
            _num_names_total_check = (
                num_names_singleton + num_names_unassociated + num_names_multiton
            )
            _num_annots_total_check = (
                num_unknown_annots + num_singleton_annots + num_multiton_annots
            )
            assert len(bad_aids) == 0, 'intersecting multiton aids and unknown aids'
            assert _num_names_total_check == num_names, 'inconsistent num names'
            # if not request_annot_subset:
            # dont check this if you have an annot subset
            # assert _num_annots_total_check == num_annots, 'inconsistent num annots'
        except Exception as ex:
            ut.printex(
                ex,
                keys=[
                    '_num_names_total_check',
                    'num_names',
                    '_num_annots_total_check',
                    'num_annots',
                    'num_names_singleton',
                    'num_names_multiton',
                    'num_unknown_annots',
                    'num_multiton_annots',
                    'num_singleton_annots',
                ],
            )
            raise

    # Get contributor statistics
    contributor_rowids = ibs.get_valid_contributor_rowids()
    num_contributors = len(contributor_rowids)

    if verbose:
        logger.info('Checking Review Info')

    # Get reviewer statistics
    def get_review_decision_stats(ibs, rid_list):
        review_decision_list = ibs.get_review_decision_str(rid_list)
        review_decision_to_rids = ut.group_items(rid_list, review_decision_list)
        review_decision_stats = {
            key: len(val) for key, val in review_decision_to_rids.items()
        }
        return review_decision_stats

    def get_review_identity(rid_list):
        review_identity_list = ibs.get_review_identity(rid_list)
        review_identity_list = [
            value.replace('user:web', 'human:web')
            .replace('web:None', 'web')
            .replace('auto_clf', 'vamp')
            .replace(':', '[')
            + ']'
            for value in review_identity_list
        ]
        return review_identity_list

    def get_review_identity_stats(ibs, rid_list):
        review_identity_list = get_review_identity(rid_list)
        review_identity_to_rids = ut.group_items(rid_list, review_identity_list)
        review_identity_stats = {
            key: len(val) for key, val in review_identity_to_rids.items()
        }
        return review_identity_to_rids, review_identity_stats

    def get_review_participation(
        review_aids_list, value_list, aid2_occurxs, nid2_occurxs
    ):
        annot_review_participation_dict = {}
        encounter_review_participation_dict = {}

        review_aid_list = ut.flatten(review_aids_list)
        review_nid_list = ibs.get_annot_nids(review_aid_list)
        review_aid_nid_dict = dict(zip(review_aid_list, review_nid_list))

        known_aids = set(aid2_occurxs.keys())
        known_encounters = set()
        for nid, occurxs in nid2_occurxs.items():
            for occurx in occurxs:
                encounter = '{},{}'.format(
                    occurx,
                    nid,
                )
                known_encounters.add(encounter)

        for review_aids, value in list(zip(review_aids_list, value_list)):
            for value_ in [value, 'Any']:
                enc_values_ = [
                    (None, value_),
                    (True, '%s (INTRA)' % (value_)),
                    (False, '%s (INTER)' % (value_)),
                ]

                review_nids = ut.take(review_aid_nid_dict, review_aids)
                review_occurxs = ut.flatten(ut.take(aid2_occurxs, review_aids))

                is_intra = len(set(review_occurxs)) == 1

                if value_ not in annot_review_participation_dict:
                    annot_review_participation_dict[value_] = {
                        '__KNOWN__': known_aids,
                        '__HIT__': set(),
                    }
                for env_flag_, enc_value_ in enc_values_:
                    if enc_value_ not in encounter_review_participation_dict:
                        encounter_review_participation_dict[enc_value_] = {
                            '__KNOWN__': known_encounters,
                            '__HIT__': set(),
                        }

                for aid, nid, occurx in zip(review_aids, review_nids, review_occurxs):
                    encounter = '{},{}'.format(
                        occurx,
                        nid,
                    )
                    annot_review_participation_dict[value_]['__HIT__'].add(aid)
                    if aid not in annot_review_participation_dict[value_]:
                        annot_review_participation_dict[value_][aid] = 0
                    annot_review_participation_dict[value_][aid] += 1
                    for env_flag_, enc_value_ in enc_values_:
                        if env_flag_ in [None, is_intra]:
                            encounter_review_participation_dict[enc_value_][
                                '__HIT__'
                            ].add(encounter)
                            if (
                                encounter
                                not in encounter_review_participation_dict[enc_value_]
                            ):
                                encounter_review_participation_dict[enc_value_][
                                    encounter
                                ] = 0
                            encounter_review_participation_dict[enc_value_][
                                encounter
                            ] += 1

        for review_participation_dict in [
            annot_review_participation_dict,
            encounter_review_participation_dict,
        ]:
            for value in review_participation_dict:
                known_values = review_participation_dict[value].pop('__KNOWN__')
                hit_values = review_participation_dict[value].pop('__HIT__')
                missed_values = known_values - hit_values
                values = list(review_participation_dict[value].values())
                stats = ut.get_stats(values, use_median=True, use_sum=True)
                stats['known'] = len(known_values)
                stats['hit'] = len(hit_values)
                stats['miss'] = len(missed_values)
                review_participation_dict[value] = stats

        return annot_review_participation_dict, encounter_review_participation_dict

    review_decision_stats = get_review_decision_stats(ibs, valid_rids)
    review_identity_to_rids, review_identity_stats = get_review_identity_stats(
        ibs, valid_rids
    )

    review_identity_to_decision_stats = {
        key: get_review_decision_stats(ibs, aids)
        for key, aids in review_identity_to_rids.items()
    }

    review_aids_list = ibs.get_review_aid_tuple(valid_rids)
    review_decision_list = ibs.get_review_decision_str(valid_rids)
    review_identity_list = get_review_identity(valid_rids)
    (
        review_decision_annot_participation_dict,
        review_decision_encounter_participation_dict,
    ) = get_review_participation(
        review_aids_list, review_decision_list, aid2_occurxs, nid2_occurxs
    )
    (
        review_identity_annot_participation_dict,
        review_identity_encounter_participation_dict,
    ) = get_review_participation(
        review_aids_list, review_identity_list, aid2_occurxs, nid2_occurxs
    )

    review_tags_list = ibs.get_review_tags(valid_rids)
    review_tag_list = [
        review_tag if review_tag is None else '+'.join(sorted(review_tag))
        for review_tag in review_tags_list
    ]

    review_tag_to_rids = ut.group_items(valid_rids, review_tag_list)
    review_tag_stats = {key: len(val) for key, val in review_tag_to_rids.items()}

    if with_ca:
        species_list = ibs.get_annot_species_texts(valid_aids)
        viewpoint_list = ibs.get_annot_viewpoints(valid_aids)
        quality_list = ibs.get_annot_qualities(valid_aids)
        interest_list = ibs.get_annot_interest(valid_aids)
        canonical_list = ibs.get_annot_canonical(valid_aids)

        # ggr_num_relevant = 0
        ggr_num_species = 0
        ggr_num_viewpoints = 0
        ggr_num_qualities = 0
        ggr_num_filter = 0
        ggr_num_aois = 0
        ggr_num_cas = 0
        ggr_num_filter_overlap = 0
        ggr_num_filter_remove = 0
        ggr_num_filter_add = 0
        ggr_num_aoi_overlap = 0
        ggr_num_aoi_remove = 0
        ggr_num_aoi_add = 0

        zipped = list(
            zip(
                valid_aids,
                species_list,
                viewpoint_list,
                quality_list,
                interest_list,
                canonical_list,
            )
        )
        ca_removed_aids = []
        ca_added_aids = []
        for aid, species_, viewpoint_, quality_, interest_, canonical_ in zipped:
            if species_ == 'zebra_grevys+_canonical_':
                continue
            assert None not in [species_, viewpoint_, quality_]
            species_ = species_.lower()
            viewpoint_ = viewpoint_.lower()
            quality_ = int(quality_)
            # if species_ in ['zebra_grevys']:
            #     ggr_num_relevant += 1
            if species_ in ['zebra_grevys']:
                ggr_num_species += 1
                filter_viewpoint_ = 'right' in viewpoint_
                filter_quality_ = quality_ >= 3
                filter_ = filter_viewpoint_ and filter_quality_

                if canonical_:
                    ggr_num_cas += 1

                if filter_viewpoint_:
                    ggr_num_viewpoints += 1

                if filter_quality_:
                    ggr_num_qualities += 1

                if filter_:
                    ggr_num_filter += 1
                    if canonical_:
                        ggr_num_filter_overlap += 1
                    else:
                        ggr_num_filter_remove += 1
                        ca_removed_aids.append(aid)
                else:
                    if canonical_:
                        ggr_num_filter_add += 1
                        ca_added_aids.append(aid)

                if interest_:
                    ggr_num_aois += 1
                    if canonical_:
                        ggr_num_aoi_overlap += 1
                    else:
                        ggr_num_aoi_remove += 1
                else:
                    if canonical_:
                        ggr_num_aoi_add += 1

        print('CA REMOVED: {}'.format(ca_removed_aids))
        print('CA ADDED: {}'.format(ca_added_aids))

        removed_chip_paths = ibs.get_annot_chip_fpath(ca_removed_aids)
        added_chip_paths = ibs.get_annot_chip_fpath(ca_added_aids)

        removed_output_path = abspath(join('.', 'ca_removed'))
        added_output_path = abspath(join('.', 'ca_added'))

        ut.delete(removed_output_path)
        ut.delete(added_output_path)

        ut.ensuredir(removed_output_path)
        ut.ensuredir(added_output_path)

        for removed_chip_path in removed_chip_paths:
            removed_chip_filename = split(removed_chip_path)[1]
            removed_output_filepath = join(removed_output_path, removed_chip_filename)
            ut.copy(removed_chip_path, removed_output_filepath, verbose=False)

        for added_chip_path in added_chip_paths:
            added_chip_filename = split(added_chip_path)[1]
            added_output_filepath = join(added_output_path, added_chip_filename)
            ut.copy(added_chip_path, added_output_filepath, verbose=False)

    #########

    num_tabs = 30

    def align2(str_):
        return ut.align(str_, ':', ' :')

    def align_dict2(dict_):
        # str_ = ut.repr2(dict_, si=True)
        str_ = ut.repr3(dict_, si=True)
        return align2(str_)

    header_block_lines = [('+============================')] + (
        [
            ('+ singleton := names with a single annotation'),
            ('+ multiton  := names with multiple annotations'),
            ('--' * num_tabs),
        ]
        if not short and with_header
        else []
    )

    source_block_lines = [
        ('DB Info:  ' + ibs.get_dbname()),
        # ('DB Notes: ' + ibs.get_dbnotes()),
        ('DB NumContrib: %d' % num_contributors),
    ]

    bytes_block_lines = (
        [
            ('--' * num_tabs),
            ('DB Bytes: '),
            ('     +- dbdir nBytes:         ' + dbdir_space),
            ('     |  +- _ibsdb nBytes:     ' + ibsdir_space),
            ('     |  |  +-imgdir nBytes:   ' + imgdir_space),
            ('     |  |  +-cachedir nBytes: ' + cachedir_space),
        ]
        if with_bytes
        else []
    )

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
        ('# Annots %s            = %d' % (subset_str, num_annots)),
        ('# Annots (unknown)           = %d' % num_unknown_annots),
        (
            '# Annots (named)             = %d'
            % (num_singleton_annots + num_multiton_annots)
        ),
        ('# Annots (singleton)         = %d' % num_singleton_annots),
        ('# Annots (multiton)          = %d' % num_multiton_annots),
    ]

    annot_per_basic_block_lines = (
        [
            ('--' * num_tabs),
            # ('# Annots per Name (multiton) = %s' % (align2(multiton_stats),)),
            ('# Annots per Image           = {}'.format(align2(gx2_nAnnots_stats))),
            ('# Annots per Species         = {}'.format(align_dict2(species2_nAids))),
        ]
        if not short
        else []
    )

    annot_per_qualview_block_lines = [
        None if short else '# Annots per Viewpoint = %s' % align_dict2(viewcode2_nAnnots),
        None if short else '# Annots per Quality = %s' % align_dict2(qualtext2_nAnnots),
    ]

    annot_per_agesex_block_lines = (
        [
            ('# Annots per Age = %s' % align_dict2(agetext2_nAnnots)),
            ('# Annots per Sex = %s' % align_dict2(sextext2_nAnnots)),
        ]
        if not short and with_agesex
        else []
    )

    annot_ggr_census = (
        [
            ('GGR Annots: '),
            # ('     +-Relevant:            %s' % (ggr_num_relevant,)),
            ("     +- Grevy's Species:    {}".format(ggr_num_species)),
            ('     |  +-AoIs:             {}'.format(ggr_num_aois)),
            ('     |  |  +-Right Side:    {}'.format(ggr_num_viewpoints)),
            ('     |  |  +-Good Quality:  {}'.format(ggr_num_qualities)),
            ('     |  |  +-Filter:        {}'.format(ggr_num_filter)),
            ('     |  +-CAs:              {}'.format(ggr_num_cas)),
            (
                '     +-CA & Filter Overlap: %s (CA removed %d, added %d)'
                % (ggr_num_filter_overlap, ggr_num_filter_remove, ggr_num_filter_add)
            ),
            (
                '     +-CA & AOI    Overlap: %s (CA removed %d, added %d)'
                % (ggr_num_aoi_overlap, ggr_num_aoi_remove, ggr_num_aoi_add)
            ),
        ]
        if with_ggr
        else []
    )

    from wbia.algo.preproc import occurrence_blackbox

    valid_nids_ = ibs.get_annot_nids(valid_aids)
    valid_gids_ = ibs.get_annot_gids(valid_aids)
    date_str_list_ = get_dates(ibs, valid_gids_)
    name_dates_stats = {}
    for valid_aid, valid_nid, date_str in zip(valid_aids, valid_nids_, date_str_list_):
        if valid_nid < 0:
            continue
        if valid_nid not in name_dates_stats:
            name_dates_stats[valid_nid] = set()
        name_dates_stats[valid_nid].add(date_str)

    if with_ggr:
        ggr_name_dates_stats = {
            'GGR-16 D1 OR D2': 0,
            'GGR-16 D1 AND D2': 0,
            'GGR-18 D1 OR D2': 0,
            'GGR-18 D1 AND D2': 0,
            'GGR-16 AND GGR-18': 0,
            '1+ Days': 0,
            '2+ Days': 0,
            '3+ Days': 0,
            '4+ Days': 0,
        }
        for date_str in sorted(set(date_str_list_)):
            ggr_name_dates_stats[date_str] = 0
        for nid in name_dates_stats:
            date_strs = name_dates_stats[nid]
            total_days = len(date_strs)
            assert 0 < total_days and total_days <= 4
            for val in range(1, total_days + 1):
                key = '%d+ Days' % (val,)
                ggr_name_dates_stats[key] += 1
            for date_str in date_strs:
                ggr_name_dates_stats[date_str] += 1
            if '2016/01/30' in date_strs or '2016/01/31' in date_strs:
                ggr_name_dates_stats['GGR-16 D1 OR D2'] += 1
                if '2018/01/27' in date_strs or '2018/01/28' in date_strs:
                    ggr_name_dates_stats['GGR-16 AND GGR-18'] += 1
            if '2018/01/27' in date_strs or '2018/01/28' in date_strs:
                ggr_name_dates_stats['GGR-18 D1 OR D2'] += 1
            if '2016/01/30' in date_strs and '2016/01/31' in date_strs:
                ggr_name_dates_stats['GGR-16 D1 AND D2'] += 1
            if '2018/01/27' in date_strs and '2018/01/28' in date_strs:
                ggr_name_dates_stats['GGR-18 D1 AND D2'] += 1

        ggr16_pl_index, ggr16_pl_error = sight_resight_count(
            ggr_name_dates_stats['2016/01/30'],
            ggr_name_dates_stats['2016/01/31'],
            ggr_name_dates_stats['GGR-16 D1 AND D2'],
        )
        ggr_name_dates_stats['GGR-16 PL INDEX'] = '{:0.01f} +/- {:0.01f}'.format(
            ggr16_pl_index,
            ggr16_pl_error,
        )
        total = ggr_name_dates_stats['GGR-16 D1 OR D2']
        ggr_name_dates_stats['GGR-16 COVERAGE'] = '{:0.01f} ({:0.01f} - {:0.01f})'.format(
            100.0 * total / ggr16_pl_index,
            100.0 * total / (ggr16_pl_index + ggr16_pl_error),
            100.0 * min(1.0, total / (ggr16_pl_index - ggr16_pl_error)),
        )

        ggr18_pl_index, ggr18_pl_error = sight_resight_count(
            ggr_name_dates_stats['2018/01/27'],
            ggr_name_dates_stats['2018/01/28'],
            ggr_name_dates_stats['GGR-18 D1 AND D2'],
        )
        ggr_name_dates_stats['GGR-18 PL INDEX'] = '{:0.01f} +/- {:0.01f}'.format(
            ggr18_pl_index,
            ggr18_pl_error,
        )
        total = ggr_name_dates_stats['GGR-18 D1 OR D2']
        ggr_name_dates_stats['GGR-18 COVERAGE'] = '{:0.01f} ({:0.01f} - {:0.01f})'.format(
            100.0 * total / ggr18_pl_index,
            100.0 * total / (ggr18_pl_index + ggr18_pl_error),
            100.0 * min(1.0, total / (ggr18_pl_index - ggr18_pl_error)),
        )
    else:
        ggr_name_dates_stats = {}

    occurrence_block_lines = (
        [
            ('--' * num_tabs),
            '# Occurrences                    = {}'.format(len(occurid2_aids)),
            '# Occurrences with Named         = %s'
            % (len(set(ut.flatten(aid2_occurxs.values()))),),
            '#      +- GPS Filter             = %s'
            % (occurence_config.get('use_gps', False),),
            '#      +- GPS Threshold KM/Sec.  = %0.04f'
            % (occurrence_blackbox.KM_PER_SEC,),
            '#      +- Time Filter            = {}'.format(True),
            '#      +- Time Threshold Sec.    = %0.1f'
            % (occurence_config.get('seconds_thresh', None),),
            (
                '# Named Annots per Occurrence    = %s'
                % (align_dict2(occurence_annot_stats),)
            ),
            (
                '# Encounters per Occurrence    = %s'
                % (align_dict2(occurence_encounter_stats),)
            ),
            '# Encounters                     = %s'
            % (len(ut.flatten(nid2_occurxs.values())),),
            (
                '# Encounters per Name            = %s'
                % (align_dict2(annot_encounter_stats),)
            ),
            '# Annotations with Names           = %s'
            % (len(set(ut.flatten(nid2_annotxs.values()))),),
            (
                '# Annotations per Name             = %s'
                % (align_dict2(name_annot_stats),)
            ),
            # ('# Pair Tag Info (annots) = %s' % (align_dict2(pair_tag_info),)),
        ]
        if not short
        else []
    )

    reviews_block_lines = (
        [
            ('--' * num_tabs),
            ('# All Reviews                = %d' % num_all_total_reviews),
            ('# Relevant Reviews           = %d' % len(valid_rids)),
            ('# Reviews per Decision       = %s' % align_dict2(review_decision_stats)),
            ('# Reviews per Reviewer       = %s' % align_dict2(review_identity_stats)),
            (
                '# Review Breakdown           = %s'
                % align_dict2(review_identity_to_decision_stats)
            ),
            ('# Reviews with Tag           = %s' % align_dict2(review_tag_stats)),
            (
                '# Annot Review Participation by Decision = %s'
                % align_dict2(review_decision_annot_participation_dict)
            ),
            (
                '# Encounter Review Participation by Decision = %s'
                % align_dict2(review_decision_encounter_participation_dict)
            ),
            (
                '# Annot Review Participation by Reviewer = %s'
                % align_dict2(review_identity_annot_participation_dict)
            ),
            (
                '# Encounter Review Participation by Reviewer = %s'
                % align_dict2(review_identity_encounter_participation_dict)
            ),
        ]
        if with_reviews
        else []
    )

    img_block_lines = [
        ('--' * num_tabs),
        ('# Img                        = %d' % len(valid_gids)),
        None
        if short
        else ('# Img reviewed               = %d' % sum(image_reviewed_list)),
        None if short else ('# Img with gps               = %d' % len(gps_list)),
        # ('# Img with timestamp         = %d' % len(valid_unixtime_list)),
        None
        if short
        else ('Img Time Stats               = {}'.format(align2(unixtime_statstr))),
        ('GGR Days                     = {}'.format(align_dict2(ggr_dates_stats)))
        if with_ggr
        else None,
        ('GGR Name Stats               = {}'.format(align_dict2(ggr_name_dates_stats)))
        if with_ggr
        else None,
    ]

    contributor_block_lines = (
        [
            ('--' * num_tabs),
            (
                '# Images per contributor       = '
                + align_dict2(contributor_tag_to_nImages)
            ),
            (
                '# Annots per contributor       = '
                + align_dict2(contributor_tag_to_nAnnots)
            ),
            (
                '# Quality per contributor      = '
                + align_dict2(contributor_tag_to_qualstats)
            ),
            (
                '# Viewpoint per contributor    = '
                + align_dict2(contributor_tag_to_viewstats)
            ),
        ]
        if with_contrib
        else []
    )

    info_str_lines = (
        header_block_lines
        + bytes_block_lines
        + source_block_lines
        + name_block_lines
        + annot_block_lines
        + annot_per_basic_block_lines
        + annot_per_qualview_block_lines
        + annot_per_agesex_block_lines
        + annot_ggr_census
        + occurrence_block_lines
        + reviews_block_lines
        + img_block_lines
        + imgsize_stat_lines
        + contributor_block_lines
        + [('L============================')]
    )
    info_str = '\n'.join(ut.filter_Nones(info_str_lines))
    info_str2 = ut.indent(info_str, '[{tag}] '.format(tag=tag))
    if verbose:
        logger.info(info_str2)
    locals_ = locals()
    return locals_


def hackshow_names(ibs, aid_list, fnum=None):
    r"""
    Args:
        ibs (IBEISController):  wbia controller object
        aid_list (list):

    CommandLine:
        python -m wbia.other.dbinfo --exec-hackshow_names --show
        python -m wbia.other.dbinfo --exec-hackshow_names --show --db PZ_Master1

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.other.dbinfo import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='PZ_MTEST')
        >>> aid_list = ibs.get_valid_aids()
        >>> result = hackshow_names(ibs, aid_list)
        >>> print(result)
        >>> ut.show_if_requested()
    """
    import vtool as vt

    import wbia.plottool as pt

    grouped_aids, nid_list = ibs.group_annots_by_name(aid_list)
    grouped_aids = [aids for aids in grouped_aids if len(aids) > 1]
    unixtimes_list = ibs.unflat_map(ibs.get_annot_image_unixtimes_asfloat, grouped_aids)
    yaws_list = ibs.unflat_map(ibs.get_annot_yaws, grouped_aids)
    # markers_list = [[(1, 2, yaw * 360 / (np.pi * 2)) for yaw in yaws] for yaws in yaws_list]

    unixtime_list = ut.flatten(unixtimes_list)
    timemax = np.nanmax(unixtime_list)
    timemin = np.nanmin(unixtime_list)
    timerange = timemax - timemin
    unixtimes_list = [
        ((unixtimes[:] - timemin) / timerange) for unixtimes in unixtimes_list
    ]
    for unixtimes in unixtimes_list:
        num_nan = sum(np.isnan(unixtimes))
        unixtimes[np.isnan(unixtimes)] = np.linspace(-1, -0.5, num_nan)
    # ydata_list = [np.arange(len(aids)) for aids in grouped_aids]
    sortx_list = vt.argsort_groups(unixtimes_list, reverse=False)
    # markers_list = ut.list_ziptake(markers_list, sortx_list)
    yaws_list = ut.list_ziptake(yaws_list, sortx_list)
    ydatas_list = vt.ziptake(unixtimes_list, sortx_list)
    # ydatas_list = sortx_list
    # ydatas_list = vt.argsort_groups(unixtimes_list, reverse=False)

    # Sort by num members
    # ydatas_list = ut.take(ydatas_list, np.argsort(list(map(len, ydatas_list))))
    xdatas_list = [
        np.zeros(len(ydatas)) + count for count, ydatas in enumerate(ydatas_list)
    ]
    # markers = ut.flatten(markers_list)
    # yaws = np.array(ut.flatten(yaws_list))
    y_data = np.array(ut.flatten(ydatas_list))
    x_data = np.array(ut.flatten(xdatas_list))
    fnum = pt.ensure_fnum(fnum)
    pt.figure(fnum=fnum)
    ax = pt.gca()

    # unique_yaws, groupxs = vt.group_indices(yaws)

    ax.scatter(x_data, y_data, color=[1, 0, 0], s=1, marker='.')
    # pt.draw_stems(x_data, y_data, marker=markers, setlims=True, linestyle='')
    pt.dark_background()
    ax = pt.gca()
    ax.set_xlim(min(x_data) - 0.1, max(x_data) + 0.1)
    ax.set_ylim(min(y_data) - 0.1, max(y_data) + 0.1)


def show_image_time_distributions(ibs, gid_list):
    r"""
    Args:
        ibs (IBEISController):  wbia controller object
        gid_list (list):

    CommandLine:
        python -m wbia.other.dbinfo show_image_time_distributions --show
        python -m wbia.other.dbinfo show_image_time_distributions --show --db lynx

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.other.dbinfo import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> aids = wbia.testdata_aids(ibs=ibs)
        >>> gid_list = ut.unique_unordered(ibs.get_annot_gids(aids))
        >>> result = show_image_time_distributions(ibs, gid_list)
        >>> print(result)
        >>> ut.show_if_requested()
    """
    unixtime_list = ibs.get_image_unixtime(gid_list)
    unixtime_list = np.array(unixtime_list, dtype=np.float)
    unixtime_list = ut.list_replace(unixtime_list, -1, float('nan'))
    show_time_distributions(ibs, unixtime_list)


def show_time_distributions(ibs, unixtime_list):
    r""""""
    # import vtool as vt
    import wbia.plottool as pt

    unixtime_list = np.array(unixtime_list)
    num_nan = np.isnan(unixtime_list).sum()
    num_total = len(unixtime_list)
    unixtime_list = unixtime_list[~np.isnan(unixtime_list)]

    import matplotlib as mpl

    from wbia.scripts.thesis import TMP_RC

    mpl.rcParams.update(TMP_RC)

    if False:
        from matplotlib import dates as mpldates

        # data_list = list(map(ut.unixtime_to_datetimeobj, unixtime_list))
        n, bins, patches = pt.plt.hist(unixtime_list, 365)
        # n_ = list(map(ut.unixtime_to_datetimeobj, n))
        # bins_ = list(map(ut.unixtime_to_datetimeobj, bins))
        pt.plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
        ax = pt.gca()
        # ax.xaxis.set_major_locator(mpldates.YearLocator())
        # hfmt = mpldates.DateFormatter('%y/%m/%d')
        # ax.xaxis.set_major_formatter(hfmt)
        mpldates.num2date(unixtime_list)
        # pt.gcf().autofmt_xdate()
        # y = pt.plt.normpdf( bins, unixtime_list.mean(), unixtime_list.std())
        # ax.set_xticks(bins_)
        # l = pt.plt.plot(bins_, y, 'k--', linewidth=1.5)
    else:
        pt.draw_time_distribution(unixtime_list)
        # pt.draw_histogram()
        ax = pt.gca()
        ax.set_xlabel('Date')
        ax.set_title(
            'Timestamp distribution of %s. #nan=%d/%d'
            % (ibs.get_dbname_alias(), num_nan, num_total)
        )
        pt.gcf().autofmt_xdate()

        icon = ibs.get_database_icon()
        if False and icon is not None:
            # import matplotlib as mpl
            # import vtool as vt
            ax = pt.gca()
            # Overlay a species icon
            # http://matplotlib.org/examples/pylab_examples/demo_annotation_box.html
            # icon = vt.convert_image_list_colorspace([icon], 'RGB', 'BGR')[0]
            # pt.overlay_icon(icon, coords=(0, 1), bbox_alignment=(0, 1))
            pt.overlay_icon(
                icon,
                coords=(0, 1),
                bbox_alignment=(0, 1),
                as_artist=1,
                max_asize=(100, 200),
            )
            # imagebox = mpl.offsetbox.OffsetImage(icon, zoom=1.0)
            # # xy = [ax.get_xlim()[0] + 5, ax.get_ylim()[1]]
            # # ax.set_xlim(1, 100)
            # # ax.set_ylim(0, 100)
            # # x = np.array(ax.get_xlim()).sum() / 2
            # # y = np.array(ax.get_ylim()).sum() / 2
            # # xy = [x, y]
            # # logger.info('xy = %r' % (xy,))
            # # x = np.nanmin(unixtime_list)
            # # xy = [x, y]
            # # logger.info('xy = %r' % (xy,))
            # # ax.get_ylim()[0]]
            # xy = [ax.get_xlim()[0], ax.get_ylim()[1]]
            # ab = mpl.offsetbox.AnnotationBbox(
            #    imagebox, xy, xycoords='data',
            #    xybox=(-0., 0.),
            #    boxcoords="offset points",
            #    box_alignment=(0, 1), pad=0.0)
            # ax.add_artist(ab)

    if ut.get_argflag('--contextadjust'):
        # pt.adjust_subplots(left=.08, bottom=.1, top=.9, wspace=.3, hspace=.1)
        pt.adjust_subplots(use_argv=True)


def latex_dbstats(ibs_list, **kwargs):
    r"""
    Args:
        ibs (IBEISController):  wbia controller object

    CommandLine:
        python -m wbia.other.dbinfo --exec-latex_dbstats --dblist testdb1
        python -m wbia.other.dbinfo --exec-latex_dbstats --dblist testdb1 --show
        python -m wbia.other.dbinfo --exec-latex_dbstats --dblist PZ_Master0 testdb1 --show
        python -m wbia.other.dbinfo --exec-latex_dbstats --dblist PZ_Master0 PZ_MTEST GZ_ALL --show
        python -m wbia.other.dbinfo --test-latex_dbstats --dblist GZ_ALL NNP_MasterGIRM_core --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.other.dbinfo import *  # NOQA
        >>> import wbia
        >>> db_list = ut.get_argval('--dblist', type_=list, default=['testdb1'])
        >>> ibs_list = [wbia.opendb(db=db) for db in db_list]
        >>> tabular_str = latex_dbstats(ibs_list)
        >>> tabular_cmd = ut.latex_newcommand(ut.latex_sanitize_command_name('DatabaseInfo'), tabular_str)
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
    import wbia

    # Parse for aids test data
    aids_list = [wbia.testdata_aids(ibs=ibs) for ibs in ibs_list]

    # dbinfo_list = [get_dbinfo(ibs, with_contrib=False, verbose=False) for ibs in ibs_list]
    dbinfo_list = [
        get_dbinfo(ibs, with_contrib=False, verbose=False, aid_list=aids)
        for ibs, aids in zip(ibs_list, aids_list)
    ]

    # title = db_name + ' database statistics'
    title = 'Database statistics'
    stat_title = '# Annotations per name (multiton)'

    # col_lbls = [
    #    'multiton',
    #    #'singleton',
    #    'total',
    #    'multiton',
    #    'singleton',
    #    'total',
    # ]
    key_to_col_lbls = {
        'num_names_multiton': 'multiton',
        'num_names_singleton': 'singleton',
        'num_names': 'total',
        'num_multiton_annots': 'multiton',
        'num_singleton_annots': 'singleton',
        'num_unknown_annots': 'unknown',
        'num_annots': 'total',
    }
    # Structure of columns / multicolumns
    multi_col_keys = [
        (
            '# Names',
            (
                'num_names_multiton',
                # 'num_names_singleton',
                'num_names',
            ),
        ),
        (
            '# Annots',
            (
                'num_multiton_annots',
                'num_singleton_annots',
                # 'num_unknown_annots',
                'num_annots',
            ),
        ),
    ]
    # multicol_lbls = [('# Names', 3), ('# Annots', 3)]
    multicol_lbls = [(mcolname, len(mcols)) for mcolname, mcols in multi_col_keys]

    # Flatten column labels
    col_keys = ut.flatten(ut.get_list_column(multi_col_keys, 1))
    col_lbls = ut.dict_take(key_to_col_lbls, col_keys)

    row_lbls = []
    row_values = []

    # stat_col_lbls = ['max', 'min', 'mean', 'std', 'nMin', 'nMax']
    stat_col_lbls = ['max', 'min', 'mean', 'std', 'med']
    # stat_row_lbls = ['# Annot per Name (multiton)']
    stat_row_lbls = []
    stat_row_values = []

    SINGLE_TABLE = False
    EXTRA = True

    for ibs, dbinfo_locals in zip(ibs_list, dbinfo_list):
        row_ = ut.dict_take(dbinfo_locals, col_keys)
        dbname = ibs.get_dbname_alias()
        row_lbls.append(dbname)
        multiton_annot_stats = ut.get_stats(
            dbinfo_locals['multiton_nid2_nannots'], use_median=True, nl=1
        )
        stat_rows = ut.dict_take(multiton_annot_stats, stat_col_lbls)
        if SINGLE_TABLE:
            row_.extend(stat_rows)
        else:
            stat_row_lbls.append(dbname)
            stat_row_values.append(stat_rows)

        row_values.append(row_)

    CENTERLINE = False
    AS_TABLE = True
    tablekw = dict(
        astable=AS_TABLE,
        centerline=CENTERLINE,
        FORCE_INT=False,
        precision=2,
        col_sep='',
        multicol_sep='|',
        **kwargs
    )

    if EXTRA:
        extra_keys = [
            # 'species2_nAids',
            'qualtext2_nAnnots',
            'viewcode2_nAnnots',
        ]
        extra_titles = {
            'species2_nAids': 'Annotations per species.',
            'qualtext2_nAnnots': 'Annotations per quality.',
            'viewcode2_nAnnots': 'Annotations per viewpoint.',
        }
        extra_collbls = ut.ddict(list)
        extra_rowvalues = ut.ddict(list)
        extra_tables = ut.ddict(list)

        for ibs, dbinfo_locals in zip(ibs_list, dbinfo_list):
            for key in extra_keys:
                extra_collbls[key] = ut.unique_ordered(
                    extra_collbls[key] + list(dbinfo_locals[key].keys())
                )

        extra_collbls['qualtext2_nAnnots'] = [
            'excellent',
            'good',
            'ok',
            'poor',
            'junk',
            'UNKNOWN',
        ]
        # extra_collbls['viewcode2_nAnnots'] = ['backleft', 'left', 'frontleft', 'front', 'frontright', 'right', 'backright', 'back', None]
        extra_collbls['viewcode2_nAnnots'] = [
            'BL',
            'L',
            'FL',
            'F',
            'FR',
            'R',
            'BR',
            'B',
            None,
        ]

        for ibs, dbinfo_locals in zip(ibs_list, dbinfo_list):
            for key in extra_keys:
                extra_rowvalues[key].append(
                    ut.dict_take(dbinfo_locals[key], extra_collbls[key], 0)
                )

        qualalias = {'UNKNOWN': None}

        extra_collbls['viewcode2_nAnnots'] = [
            ibs.const.YAWALIAS.get(val, val) for val in extra_collbls['viewcode2_nAnnots']
        ]
        extra_collbls['qualtext2_nAnnots'] = [
            qualalias.get(val, val) for val in extra_collbls['qualtext2_nAnnots']
        ]

        for key in extra_keys:
            extra_tables[key] = ut.util_latex.make_score_tabular(
                row_lbls,
                extra_collbls[key],
                extra_rowvalues[key],
                title=extra_titles[key],
                col_align='r',
                table_position='[h!]',
                **tablekw
            )

    # tabular_str = util_latex.tabular_join(tabular_body_list)
    if SINGLE_TABLE:
        col_lbls += stat_col_lbls
        multicol_lbls += [(stat_title, len(stat_col_lbls))]

    count_tabular_str = ut.util_latex.make_score_tabular(
        row_lbls,
        col_lbls,
        row_values,
        title=title,
        multicol_lbls=multicol_lbls,
        table_position='[ht!]',
        **tablekw
    )

    # logger.info(row_lbls)

    if SINGLE_TABLE:
        tabular_str = count_tabular_str
    else:
        stat_tabular_str = ut.util_latex.make_score_tabular(
            stat_row_lbls,
            stat_col_lbls,
            stat_row_values,
            title=stat_title,
            col_align='r',
            table_position='[h!]',
            **tablekw
        )

        # Make a table of statistics
        if tablekw['astable']:
            tablesep = '\n%--\n'
        else:
            tablesep = '\\\\\n%--\n'
        if EXTRA:
            tabular_str = tablesep.join(
                [count_tabular_str, stat_tabular_str]
                + ut.dict_take(extra_tables, extra_keys)
            )
        else:
            tabular_str = tablesep.join([count_tabular_str, stat_tabular_str])

    return tabular_str


def get_short_infostr(ibs):
    """Returns printable database information

    Args:
        ibs (IBEISController):  wbia controller object

    Returns:
        str: infostr

    CommandLine:
        python -m wbia.other.dbinfo --test-get_short_infostr

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.other.dbinfo import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb('testdb1')
        >>> infostr = get_short_infostr(ibs)
        >>> result = str(infostr)
        >>> print(result)
        dbname = 'testdb1'
        num_images = 13
        num_annotations = 13
        num_names = 7
    """
    dbname = ibs.get_dbname()
    # workdir = ut.unixpath(ibs.get_workdir())
    num_images = ibs.get_num_images()
    num_annotations = ibs.get_num_annotations()
    num_names = ibs.get_num_names()
    # workdir = %r
    infostr = ut.codeblock(
        """
    dbname = %s
    num_images = %r
    num_annotations = %r
    num_names = %r
    """
        % (ut.repr2(dbname), num_images, num_annotations, num_names)
    )
    return infostr


def cache_memory_stats(ibs, cid_list, fnum=None):
    logger.info('[dev stats] cache_memory_stats()')
    # kpts_list = ibs.get_annot_kpts(cid_list)
    # desc_list = ibs.get_annot_vecs(cid_list)
    # nFeats_list = map(len, kpts_list)
    gx_list = np.unique(ibs.cx2_gx(cid_list))

    bytes_map = {
        'chip dbytes': [ut.file_bytes(fpath) for fpath in ibs.get_rchip_path(cid_list)],
        'img dbytes': [
            ut.file_bytes(gpath) for gpath in ibs.gx2_gname(gx_list, full=True)
        ],
        # 'flann dbytes':  ut.file_bytes(flann_fpath),
    }

    byte_units = {
        'GB': 2 ** 30,
        'MB': 2 ** 20,
        'KB': 2 ** 10,
    }

    tabular_body_list = []

    convert_to = 'KB'
    for key, val in bytes_map.items():
        key2 = key.replace('bytes', convert_to)
        if isinstance(val, list):
            val2 = [bytes_ / byte_units[convert_to] for bytes_ in val]
            tex_str = ut.util_latex.latex_get_stats(key2, val2)
        else:
            val2 = val / byte_units[convert_to]
            tex_str = ut.util_latex.latex_scalar(key2, val2)
        tabular_body_list.append(tex_str)

    tabular = ut.util_latex.tabular_join(tabular_body_list)

    logger.info(tabular)
    ut.util_latex.render(tabular)

    if fnum is None:
        fnum = 0

    return fnum + 1


def sight_resight_count(nvisit1, nvisit2, resight):
    r"""
    Lincoln Petersen Index

    The Lincoln-Peterson index is a method used to estimate the total number of
    individuals in a population given two independent sets observations.  The
    likelihood of a population size is a hypergeometric distribution given by
    assuming a uniform sampling distribution.

    Args:
        nvisit1 (int): the number of individuals seen on visit 1.
        nvisit2 (int): be the number of individuals seen on visit 2.
        resight (int): the number of (matched) individuals seen on both visits.

    Returns:
        tuple: (pl_index, pl_error)

    LaTeX:
        \begin{equation}\label{eqn:lpifull}
            L(\poptotal \given \nvisit_1, \nvisit_2, \resight) =
            \frac{
                \binom{\nvisit_1}{\resight}
                \binom{\poptotal - \nvisit_1}{\nvisit_2 - \resight}
            }{
                \binom{\poptotal}{\nvisit_2}
            }
        \end{equation}
        Assuming that $T$ has a uniform prior distribution, the maximum
          likelihood estimation of population size given two visits to a
          location is:
        \begin{equation}\label{eqn:lpi}
            \poptotal \approx
            \frac{\nvisit_1 \nvisit_2}{\resight} \pm 1.96 \sqrt{\frac{{(\nvisit_1)}^2 (\nvisit_2) (\nvisit_2 - \resight)}{\resight^3}}
        \end{equation}

    References:
        https://en.wikipedia.org/wiki/Mark_and_recapture
        https://en.wikipedia.org/wiki/Talk:Mark_and_recapture#Statistical_treatment
        https://mail.google.com/mail/u/0/#search/lincoln+peterse+n/14c6b50227f5209f
        https://probabilityandstats.wordpress.com/tag/maximum-likelihood-estimate/
        http://math.arizona.edu/~jwatkins/o-mle.pdf

    CommandLine:
        python -m wbia.other.dbinfo sight_resight_count --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.other.dbinfo import *  # NOQA
        >>> nvisit1 = 100
        >>> nvisit2 = 20
        >>> resight = 10
        >>> (pl_index, pl_error) = sight_resight_count(nvisit1, nvisit2, resight)
        >>> result = '(pl_index, pl_error) = %s' % ut.repr2((pl_index, pl_error))
        >>> pl_low = max(pl_index - pl_error, 1)
        >>> pl_high = pl_index + pl_error
        >>> print('pl_low = %r' % (pl_low,))
        >>> print('pl_high = %r' % (pl_high,))
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> import scipy, scipy.stats
        >>> x = pl_index  # np.array([10, 11, 12])
        >>> k, N, K, n = resight, x, nvisit1, nvisit2
        >>> #k, M, n, N = k, N, k, n  # Wiki to SciPy notation
        >>> #prob = scipy.stats.hypergeom.cdf(k, N, K, n)
        >>> fig = pt.figure(1)
        >>> fig.clf()
        >>> N_range = np.arange(1, pl_high * 2)
        >>> # Something seems to be off
        >>> probs = sight_resight_prob(N_range, nvisit1, nvisit2, resight)
        >>> pl_prob = sight_resight_prob([pl_index], nvisit1, nvisit2, resight)[0]
        >>> pt.plot(N_range, probs, 'b-', label='probability of population size')
        >>> pt.plt.title('nvisit1=%r, nvisit2=%r, resight=%r' % (
        >>>     nvisit1, nvisit2, resight))
        >>> pt.plot(pl_index, pl_prob, 'rx', label='Lincoln Peterson Estimate')
        >>> pt.plot([pl_low, pl_high], [pl_prob, pl_prob], 'gx-',
        >>>         label='Lincoln Peterson Error Bar')
        >>> pt.legend()
        >>> ut.show_if_requested()
    """
    import math

    try:
        nvisit1 = float(nvisit1)
        nvisit2 = float(nvisit2)
        resight = float(resight)
        pl_index = int(math.ceil((nvisit1 * nvisit2) / resight))
        pl_error_num = float((nvisit1 ** 2) * nvisit2 * (nvisit2 - resight))
        pl_error_dom = float(resight ** 3)
        pl_error = int(math.ceil(1.96 * math.sqrt(pl_error_num / pl_error_dom)))
    except ZeroDivisionError:
        # pl_index = 'Undefined - Zero recaptured (k = 0)'
        pl_index = 0
        pl_error = 0
    return pl_index, pl_error
