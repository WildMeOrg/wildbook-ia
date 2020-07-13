# -*- coding: utf-8 -*-
"""
sysres.py == system_resources
Module for dealing with system resoureces in the context of IBEIS
but without the need for an actual IBEIS Controller
"""
from __future__ import absolute_import, division, print_function  # , unicode_literals
import os
from os.path import exists, join, realpath
import utool as ut
import ubelt as ub
from six.moves import input, zip, map
from wbia import constants as const
from wbia import params

(print, rrr, profile) = ut.inject2(__name__)

WORKDIR_CACHEID = 'work_directory_cache_id'
DEFAULTDB_CAHCEID = 'cached_dbdir'
LOGDIR_CACHEID = ut.logdir_cacheid
__APPNAME__ = 'wbia'

ALLOW_GUI = ut.WIN32 or os.environ.get('DISPLAY', None) is not None


def get_wbia_resource_dir():
    return ub.ensure_app_cache_dir('wbia')


def _wbia_cache_dump():
    ut.global_cache_dump(appname=__APPNAME__)


def _wbia_cache_write(key, val):
    """ Writes to global IBEIS cache
    TODO: Use text based config file
    """
    print('[sysres] set %s=%r' % (key, val))
    ut.global_cache_write(key, val, appname=__APPNAME__)


def _wbia_cache_read(key, **kwargs):
    """ Reads from global IBEIS cache """
    return ut.global_cache_read(key, appname=__APPNAME__, **kwargs)


# Specific cache getters / setters


def set_default_dbdir(dbdir):
    if ut.DEBUG2:
        print('[sysres] SETTING DEFAULT DBDIR: %r' % dbdir)
    _wbia_cache_write(DEFAULTDB_CAHCEID, dbdir)


def get_default_dbdir():
    dbdir = _wbia_cache_read(DEFAULTDB_CAHCEID, default=None)
    if ut.DEBUG2:
        print('[sysres] READING DEFAULT DBDIR: %r' % dbdir)
    return dbdir


def get_workdir(allow_gui=True):
    """
    Returns the work directory set for this computer.  If allow_gui is true,
    a dialog will ask a user to specify the workdir if it does not exist.

    python -c "import wbia; print(wbia.get_workdir())"

    Args:
        allow_gui (bool): (default = True)

    Returns:
        str: work_dir

    CommandLine:
        python -m wbia.init.sysres get_workdir

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.init.sysres import *  # NOQA
        >>> allow_gui = True
        >>> work_dir = get_workdir(allow_gui)
        >>> result = ('work_dir = %s' % (str(work_dir),))
        >>> print(result)
    """
    work_dir = _wbia_cache_read(WORKDIR_CACHEID, default='.')
    print('[wbia.sysres.get_workdir] work_dir = {!r}'.format(work_dir))
    if work_dir != '.' and exists(work_dir):
        return work_dir
    if allow_gui:
        work_dir = set_workdir()
        return get_workdir(allow_gui=False)
    return None


def set_workdir(work_dir=None, allow_gui=ALLOW_GUI):
    """ Sets the workdirectory for this computer

    Args:
        work_dir (None): (default = None)
        allow_gui (bool): (default = True)

    CommandLine:
        python -c "import wbia; wbia.sysres.set_workdir('/raid/work2')"
        python -c "import wbia; wbia.sysres.set_workdir('/raid/work')"

        python -m wbia.init.sysres set_workdir

    Example:
        >>> # SCRIPT
        >>> from wbia.init.sysres import *  # NOQA
        >>> print('current_work_dir = %s' % (str(get_workdir(False)),))
        >>> work_dir = ut.get_argval('--workdir', type_=str, default=None)
        >>> allow_gui = True
        >>> result = set_workdir(work_dir, allow_gui)
    """
    if work_dir is None:
        if allow_gui:
            try:
                work_dir = guiselect_workdir()
            except ImportError:
                allow_gui = False
        if not allow_gui:
            work_dir = ut.truepath(input('specify a workdir: '))
    if work_dir is None or not exists(work_dir):
        raise AssertionError('invalid workdir=%r' % work_dir)
    _wbia_cache_write(WORKDIR_CACHEID, work_dir)


def set_logdir(log_dir):
    from os.path import realpath, expanduser

    log_dir = realpath(expanduser(log_dir))
    ut.ensuredir(log_dir, verbose=True)
    ut.stop_logging()
    _wbia_cache_write(LOGDIR_CACHEID, log_dir)
    ut.start_logging(appname=__APPNAME__)


def get_logdir_global():
    return _wbia_cache_read(LOGDIR_CACHEID, default=ut.get_logging_dir(appname='wbia'))


def get_rawdir():
    """ Returns the standard raw data directory """
    workdir = get_workdir()
    rawdir = ut.truepath(join(workdir, '../raw'))
    return rawdir


def guiselect_workdir():
    """ Prompts the user to specify a work directory """
    from wbia import guitool

    guitool.ensure_qtapp()
    # Gui selection
    work_dir = guitool.select_directory('Select a work directory')

    # Make sure selection is ok
    if not exists(work_dir):
        try_again = guitool.user_option(
            paremt=None,
            msg='Directory %r does not exist.' % work_dir,
            title='get work dir failed',
            options=['Try Again'],
            use_cache=False,
        )
        if try_again == 'Try Again':
            return guiselect_workdir()
    return work_dir


def get_dbalias_dict():
    # HACK: DEPRICATE
    dbalias_dict = {}
    if ut.is_developer():
        # For jon's convinience
        dbalias_dict.update(
            {
                'NAUTS': 'NAUT_Dan',
                'WD': 'WD_Siva',
                'LF': 'LF_all',
                'GZ': 'GZ_ALL',
                'MOTHERS': 'PZ_MOTHERS',
                'FROGS': 'Frogs',
                'TOADS': 'WY_Toads',
                'SEALS_SPOTTED': 'Seals',
                'OXFORD': 'Oxford_Buildings',
                'PARIS': 'Paris_Buildings',
                'JAG_KELLY': 'JAG_Kelly',
                'JAG_KIERYN': 'JAG_Kieryn',
                'WILDEBEAST': 'Wildebeast',
                'WDOGS': 'WD_Siva',
                'PZ': 'PZ_FlankHack',
                'PZ2': 'PZ-Sweatwater',
                'PZ_MARIANNE': 'PZ_Marianne',
                'PZ_DANEXT_TEST': 'PZ_DanExt_Test',
                'PZ_DANEXT_ALL': 'PZ_DanExt_All',
                'LF_ALL': 'LF_all',
                'WS_HARD': 'WS_hard',
                'SONOGRAMS': 'sonograms',
            }
        )
        dbalias_dict['JAG'] = dbalias_dict['JAG_KELLY']
    return dbalias_dict


def db_to_dbdir(db, allow_newdir=False, extra_workdirs=[]):
    """
    Implicitly gets dbdir. Searches for db inside of workdir
    """
    if ut.VERBOSE:
        print('[sysres] db_to_dbdir: db=%r, allow_newdir=%r' % (db, allow_newdir))

    if db is None:
        raise ValueError('db is None')

    work_dir = get_workdir()
    dbalias_dict = get_dbalias_dict()

    workdir_list = []
    for extra_dir in extra_workdirs:
        if exists(extra_dir):
            workdir_list.append(extra_dir)
    workdir_list.append(work_dir)  # TODO: Allow multiple workdirs?

    # Check all of your work directories for the database
    for _dir in workdir_list:
        dbdir = realpath(join(_dir, db))
        # Use db aliases
        if not exists(dbdir) and db.upper() in dbalias_dict:
            dbdir = join(_dir, dbalias_dict[db.upper()])
        if exists(dbdir):
            break

    # Create the database if newdbs are allowed in the workdir
    # print('allow_newdir=%r' % allow_newdir)
    if allow_newdir:
        ut.ensuredir(dbdir, verbose=True)

    # Complain if the implicit dbdir does not exist
    if not exists(dbdir):
        print('!!!')
        print('[sysres] WARNING: db=%r not found in work_dir=%r' % (db, work_dir))
        fname_list = os.listdir(work_dir)
        lower_list = [fname.lower() for fname in fname_list]
        index = ut.listfind(lower_list, db.lower())
        if index is not None:
            print('[sysres] WARNING: db capitalization seems to be off')
            if not ut.STRICT:
                print('[sysres] attempting to fix it')
                db = fname_list[index]
                dbdir = join(work_dir, db)
                print('[sysres] dbdir=%r' % dbdir)
                print('[sysres] db=%r' % db)
        if not exists(dbdir):
            msg = '[sysres!] ERROR: Database does not exist and allow_newdir=False'
            print('<!!!>')
            print(msg)
            print(
                '[sysres!] Here is a list of valid dbs: '
                + ut.indentjoin(sorted(fname_list), '\n  * ')
            )
            print('[sysres!] dbdir=%r' % dbdir)
            print('[sysres!] db=%r' % db)
            print('[sysres!] work_dir=%r' % work_dir)
            print('</!!!>')
            raise AssertionError(msg)
        print('!!!')
    return dbdir


def get_args_dbdir(defaultdb=None, allow_newdir=False, db=None, dbdir=None):
    r"""
    Machinery for finding a database directory using the following priorities.
    The function first defaults to the specified function arguments.  If those
    are not specified, then command line arguments are used.  In all other
    circumstances the defaultdb is used. If defaultdb='cache' then the most
    recently used database directory is returned.

    Args:
        defaultdb (None): database return if none other is specified
        allow_newdir (bool): raises error if True and directory not found
        db (None): specification using workdir priority
        dbdir (None): specification using normal directory priority
        cache_priority (bool): (default = False)

    Returns:
        str: dbdir

    CommandLine:
        python -m wbia.init.sysres get_args_dbdir

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.init.sysres import *  # NOQA
        >>> dir1 = get_args_dbdir(None, False, 'testdb1', None)
        >>> print('dir1 = %r' % (dir1,))
        >>> dir2 = get_args_dbdir(None, False, dir1, None)
        >>> print('dir2 = %r' % (dir2,))
        >>> ut.assert_raises(ValueError, get_args_dbdir)
        >>> print('dir3 = %r' % (dir2,))
    """
    if not ut.QUIET and ut.VERBOSE:
        print('[sysres] get_args_dbdir: parsing commandline for dbdir')
        print('[sysres] defaultdb=%r, allow_newdir=%r' % (defaultdb, allow_newdir))
        print('[sysres] db=%r, dbdir=%r' % (db, dbdir))

    def _db_arg_priorty(dbdir_, db_):
        invalid = ['', ' ', '.', 'None']
        # Invalidate bad db's
        if dbdir_ in invalid:
            dbdir_ = None
        if db_ in invalid:
            db_ = None
        # Return values with a priority
        if dbdir_ is not None:
            return realpath(dbdir_)
        if db_ is not None:
            return db_to_dbdir(db_, allow_newdir=allow_newdir)
        return None

    # Check function arguments
    dbdir1 = _db_arg_priorty(dbdir, db)
    if dbdir1 is not None:
        return dbdir1

    # Check command line arguments
    dbdir_arg = params.args.dbdir
    db_arg = params.args.db
    # TODO: use these instead of params
    # ut.get_argval('--dbdir', return_was_specified=True))
    # ut.get_argval('--db', return_was_specified=True)
    # Check command line passed args
    dbdir2 = _db_arg_priorty(dbdir_arg, db_arg)
    if dbdir2 is not None:
        return dbdir2

    # Return cached database directory
    if defaultdb is None:
        raise ValueError('Must specify at least db, dbdir, or defaultdb')
    elif defaultdb == 'cache':
        return get_default_dbdir()
    else:
        return db_to_dbdir(defaultdb, allow_newdir=allow_newdir)


lookup_dbdir = db_to_dbdir


def is_wbiadb(path):
    """ Checks to see if path contains the IBEIS internal dir """
    return exists(join(path, const.PATH_NAMES._ibsdb))


def get_ibsdb_list(workdir=None):
    r"""
    Lists the available valid wbia databases inside of a work directory

    Args:
        workdir (None):

    Returns:
        IBEISController: ibsdb_list -  wbia controller object

    CommandLine:
        python -m wbia.init.sysres --test-get_ibsdb_list

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.init.sysres import *  # NOQA
        >>> workdir = None
        >>> ibsdb_list = get_ibsdb_list(workdir)
        >>> result = str('\n'.join(ibsdb_list))
        >>> print(result)
    """
    import numpy as np

    if workdir is None:
        workdir = get_workdir()
    dbname_list = os.listdir(workdir)
    dbpath_list = np.array([join(workdir, name) for name in dbname_list])
    is_ibs_list = np.array(list(map(is_wbiadb, dbpath_list)))
    ibsdb_list = dbpath_list[is_ibs_list].tolist()
    return ibsdb_list


list_dbs = get_ibsdb_list
get_available_databases = get_ibsdb_list


def ensure_wd_peter2():
    zipped_db_url = 'https://wildbookiarepository.azureedge.net/databases/wd_peter2.zip'
    return ensure_db_from_url(zipped_db_url)


def ensure_pz_mtest():
    """
    Ensures that you have the PZ_MTEST dataset

    CommandLine:
        python -m wbia.init.sysres --exec-ensure_pz_mtest
        python -m wbia --tf ensure_pz_mtest

    Ignore:
        from wbia.tests.reset_testdbs import delete_dbdir
        delete_dbdir('PZ_MTEST')

    Example:
        >>> # SCRIPT
        >>> from wbia.init.sysres import *  # NOQA
        >>> ensure_pz_mtest()
    """
    print('ensure_pz_mtest')
    from wbia import sysres

    workdir = sysres.get_workdir()
    mtest_zipped_url = const.ZIPPED_URLS.PZ_MTEST
    mtest_dir = ut.grab_zipped_url(mtest_zipped_url, ensure=True, download_dir=workdir)
    print('have mtest_dir=%r' % (mtest_dir,))
    # update the the newest database version
    import wbia

    ibs = wbia.opendb('PZ_MTEST')
    print('cleaning up old database and ensureing everything is properly computed')
    ibs.db.vacuum()
    valid_aids = ibs.get_valid_aids()
    assert len(valid_aids) == 119
    ibs.update_annot_semantic_uuids(valid_aids)
    if ut.VERYVERBOSE:
        ibs.print_annotation_table()
    nid = ibs.get_name_rowids_from_text('', ensure=False)
    if nid is not None:
        ibs.set_name_texts([nid], ['lostname'])

    # Remove old imagesets and update to new special ones
    all_imgset_ids = ibs.get_valid_imgsetids()
    special_imgset_ids = ibs.get_special_imgsetids()
    other_imgset_ids = ut.setdiff(all_imgset_ids, special_imgset_ids)
    ibs.delete_imagesets(other_imgset_ids)
    ibs.set_exemplars_from_quality_and_viewpoint()
    ibs.update_all_image_special_imageset()

    occurrence_gids = [
        2,
        9,
        12,
        16,
        25,
        26,
        29,
        30,
        32,
        33,
        35,
        46,
        47,
        52,
        57,
        61,
        66,
        70,
        71,
        73,
        74,
        76,
        77,
        78,
        79,
        87,
        88,
        90,
        96,
        97,
        103,
        106,
        108,
        110,
        112,
        113,
    ]

    other_gids = ut.setdiff(ibs.get_valid_gids(), occurrence_gids)
    other_gids1 = other_gids[0::2]
    other_gids2 = other_gids[1::2]
    ibs.set_image_imagesettext(occurrence_gids, ['Occurrence 1'] * len(occurrence_gids))
    ibs.set_image_imagesettext(other_gids1, ['Occurrence 2'] * len(other_gids1))
    ibs.set_image_imagesettext(other_gids2, ['Occurrence 3'] * len(other_gids2))

    # hack in some tags
    print('Hacking in some tags')
    foal_aids = [
        4,
        8,
        15,
        21,
        28,
        34,
        38,
        41,
        45,
        49,
        51,
        56,
        60,
        66,
        69,
        74,
        80,
        83,
        91,
        97,
        103,
        107,
        109,
        119,
    ]
    mother_aids = [9, 16, 35, 42, 52, 57, 61, 67, 75, 84, 98, 104, 108, 114]
    ibs.append_annot_case_tags(foal_aids, ['foal'] * len(foal_aids))
    ibs.append_annot_case_tags(mother_aids, ['mother'] * len(mother_aids))

    # make part of the database complete and the other part semi-complete
    # make staging ahead of annotmatch.
    reset_mtest_graph()


def reset_mtest_graph():
    """
    Resets the annotmatch and stating table

    CommandLine:
        python -m wbia reset_mtest_graph

    Example:
        >>> # SCRIPT
        >>> from wbia.init.sysres import *  # NOQA
        >>> reset_mtest_graph()
    """
    if True:
        # Delete the graph databases to and set them up for tests
        import wbia

        ibs = wbia.opendb('PZ_MTEST')
        annotmatch = ibs.db['annotmatch']
        staging = ibs.staging['reviews']
        annotmatch.clear()
        staging.clear()

    # Make this CC connected using positive edges
    from wbia.algo.graph.state import POSTV, NEGTV, INCMP, DIFF, NULL, SAME  # NOQA
    from wbia.algo.graph import nx_utils as nxu
    import itertools as it

    # Add some graph properties to MTEST
    infr = wbia.AnnotInference(ibs, 'all', autoinit=True)
    # Connect the names with meta decisions
    infr.ensure_mst(meta_decision=SAME)

    # big_ccs = [cc for cc in infr.positive_components() if len(cc) > 3]
    small_ccs = [cc for cc in infr.positive_components() if len(cc) <= 3 and len(cc) > 1]
    # single_ccs = [cc for cc in infr.positive_components() if len(cc) == 1]

    cc = infr.pos_graph.connected_to(1)
    for edge in nxu.edges_between(infr.graph, cc):
        infr.add_feedback(edge, POSTV, user_id='user:setup1')

    # Make all small PCCs k-negative-redundant
    count = 0
    for cc1, cc2 in it.combinations(small_ccs, 2):
        count += 1
        for edge in infr.find_neg_augment_edges(cc1, cc2, k=1):
            if count > 10:
                # So some with meta
                infr.add_feedback(edge, meta_decision=DIFF, user_id='user:setup2')
            else:
                # So some with evidence
                infr.add_feedback(edge, NEGTV, user_id='user:setup3')

    # Make some small PCCs k-positive-redundant
    from wbia.algo.graph.state import POSTV, NEGTV, INCMP, UNREV, UNKWN  # NOQA

    cand = list(infr.find_pos_redun_candidate_edges())
    for edge in cand[0:2]:
        infr.add_feedback(edge, evidence_decision=POSTV, user_id='user:setup4')

    assert infr.status()['nInconsistentCCs'] == 0

    # Write consistent state to both annotmatch and staging
    infr.write_wbia_staging_feedback()
    infr.write_wbia_annotmatch_feedback()

    # Add an 2 inconsistencies to the staging database ONLY
    cand = list(infr.find_pos_redun_candidate_edges())
    for edge in cand[0:2]:
        infr.add_feedback(edge, evidence_decision=NEGTV, user_id='user:voldemort')

    assert infr.status()['nInconsistentCCs'] == 2
    infr.write_wbia_staging_feedback()

    infr.reset_feedback('annotmatch', apply=True)
    assert infr.status()['nInconsistentCCs'] == 0

    # print(ibs.staging['reviews'].as_pandas())


def copy_wbiadb(source_dbdir, dest_dbdir):
    # TODO: rectify with rsync, script, and merge script.
    from os.path import normpath
    import wbia

    exclude_dirs_ = wbia.const.EXCLUDE_COPY_REL_DIRS + ['_hsdb', '.hs_internals']
    exclude_dirs = [ut.ensure_unixslash(normpath(rel)) for rel in exclude_dirs_]

    rel_tocopy = ut.glob(
        source_dbdir,
        '*',
        exclude_dirs=exclude_dirs,
        recursive=True,
        with_files=True,
        with_dirs=False,
        fullpath=False,
    )
    rel_tocopy_dirs = ut.glob(
        source_dbdir,
        '*',
        exclude_dirs=exclude_dirs,
        recursive=True,
        with_files=False,
        with_dirs=True,
        fullpath=False,
    )

    src_list = [join(source_dbdir, relpath) for relpath in rel_tocopy]
    dst_list = [join(dest_dbdir, relpath) for relpath in rel_tocopy]

    # ensure directories exist
    rel_tocopy_dirs = [dest_dbdir] + [
        join(dest_dbdir, dpath_) for dpath_ in rel_tocopy_dirs
    ]
    for dpath in rel_tocopy_dirs:
        ut.ensuredir(dpath)
    # copy files
    ut.copy(src_list, dst_list)


def ensure_pz_mtest_batchworkflow_test():
    r"""
    CommandLine:
        python -m wbia.init.sysres --test-ensure_pz_mtest_batchworkflow_test
        python -m wbia.init.sysres --test-ensure_pz_mtest_batchworkflow_test --reset
        python -m wbia.init.sysres --test-ensure_pz_mtest_batchworkflow_test --reset

    Example:
        >>> # SCRIPT
        >>> from wbia.init.sysres import *  # NOQA
        >>> ensure_pz_mtest_batchworkflow_test()
    """
    import wbia

    wbia.ensure_pz_mtest()
    workdir = wbia.sysres.get_workdir()
    mtest_dbpath = join(workdir, 'PZ_MTEST')

    source_dbdir = mtest_dbpath
    dest_dbdir = join(workdir, 'PZ_BATCH_WORKFLOW_MTEST')

    if ut.get_argflag('--reset'):
        ut.delete(dest_dbdir)

    if ut.checkpath(dest_dbdir):
        return
    else:
        copy_wbiadb(source_dbdir, dest_dbdir)

    ibs = wbia.opendb('PZ_BATCH_WORKFLOW_MTEST')
    assert len(ibs.get_valid_aids()) == 119
    assert len(ibs.get_valid_nids()) == 41

    ibs.delete_all_imagesets()

    aid_list = ibs.get_valid_aids()

    unixtime_list = ibs.get_annot_image_unixtimes(aid_list)
    untimed_aids = ut.compress(aid_list, [t == -1 for t in unixtime_list])

    ibs.get_annot_groundtruth(untimed_aids, aid_list)

    aids_list, nid_list = ibs.group_annots_by_name(aid_list)

    hourdiffs_list = ibs.get_name_hourdiffs(nid_list)

    imageset_aids_list = [[] for _ in range(4)]

    imageset_idx = 0

    for hourdiffs, aids in zip(hourdiffs_list, aids_list):
        # import scipy.spatial.distance as spdist
        if len(aids) == 1:
            imageset_aids_list[imageset_idx].extend(aids)
            imageset_idx = (imageset_idx + 1) % len(imageset_aids_list)
        else:
            for chunk in list(ut.ichunks(aids, 2)):
                imageset_aids_list[imageset_idx].extend(chunk)
                imageset_idx = (imageset_idx + 1) % len(imageset_aids_list)

            # import vtool as vt
            # import networkx as netx
            # nodes = list(range(len(aids)))
            # edges_pairs = vt.pdist_argsort(hourdiffs)
            # edge_weights = -hourdiffs[hourdiffs.argsort()]
            # netx_graph = make_netx_graph(edges_pairs, nodes, edge_weights)
            # cut_edges = netx.minimum_edge_cut(netx_graph)
            # netx_graph.remove_edges_from(cut_edges)
            # components = list(netx.connected_components(netx_graph))
            # components = ut.sortedby(components, list(map(len, components)), reverse=True)
            # print(components)
            # imageset_aids_list[0].extend(components[0])
            # for component in components:

            # TODO do max-nway cut
        # day_diffs = spdist.squareform(hourdiffs) / 24.0
        # print(ut.repr2(day_diffs, precision=2, suppress_small=True))
        # import itertools
        # compare_idxs = [(r, c) for r, c in itertools.product(range(len(aids)), range(len(aids))) if (c > r)]
        # print(len(aids))
    # def make_netx_graph(edges_pairs, nodes=None, edge_weights=None):
    #    import networkx as netx
    #    node_lbls = [('id_', 'int')]

    #    edge_lbls = [('weight', 'float')]
    #    edges = [(pair[0], pair[1], weight) for pair, weight in zip(edges_pairs, edge_weights)]

    #    print('make_netx_graph')
    #    # Make a graph between the chips
    #    netx_nodes = [(ntup[0], {key[0]: val for (key, val) in zip(node_lbls, ntup[1:])})
    #                  for ntup in iter(zip(nodes))]

    #    netx_edges = [(etup[0], etup[1], {key[0]: val for (key, val) in zip(edge_lbls, etup[2:])})
    #                  for etup in iter(edges)]
    #    netx_graph = netx.Graph()
    #    netx_graph.add_nodes_from(netx_nodes)
    #    netx_graph.add_edges_from(netx_edges)
    #    return netx_graph

    # Group into imagesets based on old names
    gids_list = ibs.unflat_map(ibs.get_annot_image_rowids, imageset_aids_list)
    imgsetid_list = ibs.new_imagesets_from_images(gids_list)  # NOQA

    # Remove all names
    ibs.delete_annot_nids(aid_list)


def ensure_pz_mtest_mergesplit_test():
    r"""
    Make a test database for MERGE and SPLIT cases

    CommandLine:
        python -m wbia.init.sysres --test-ensure_pz_mtest_mergesplit_test

    Example:
        >>> # SCRIPT
        >>> from wbia.init.sysres import *  # NOQA
        >>> ensure_pz_mtest_mergesplit_test()
    """
    import wbia

    wbia.ensure_pz_mtest()
    workdir = wbia.sysres.get_workdir()
    mtest_dbpath = join(workdir, 'PZ_MTEST')

    source_dbdir = mtest_dbpath
    dest_dbdir = join(workdir, 'PZ_MERGESPLIT_MTEST')

    if ut.get_argflag('--reset'):
        ut.delete(dest_dbdir)
    if ut.checkpath(dest_dbdir):
        return

    copy_wbiadb(source_dbdir, dest_dbdir)

    ibs = wbia.opendb('PZ_MERGESPLIT_MTEST')
    assert len(ibs.get_valid_aids()) == 119
    assert len(ibs.get_valid_nids()) == 41

    aid_list = ibs.get_valid_aids()
    aids_list, nid_list = ibs.group_annots_by_name(aid_list)
    num_aids = list(map(len, aids_list))

    # num cases wanted
    num_merge = 3
    num_split = 1
    num_combo = 1

    # num inputs needed
    num_merge_names = num_merge
    num_split_names = num_split * 2
    num_combo_names = num_combo * 3

    total_names = num_merge_names + num_split_names + num_combo_names

    modify_aids = list(
        ub.take(aids_list, ut.list_argsort(num_aids, reverse=True)[0:total_names])
    )

    merge_nids1 = ibs.make_next_nids(num_merge, location_text='XMERGE')
    merge_nids2 = ibs.make_next_nids(num_merge, location_text='XMERGE')
    split_nid = ibs.make_next_nids(num_split, location_text='XSPLIT')[0]
    combo_nids = ibs.make_next_nids(num_combo * 2, location_text='XCOMBO')

    # the first 3 become merge cases
    # left = 0
    # right = left + num_merge
    for aids, nid1, nid2 in zip(modify_aids[0:3], merge_nids1, merge_nids2):
        # ibs.get_annot_nids(aids)
        aids_ = aids[::2]
        ibs.set_annot_name_rowids(aids_, [nid1] * len(aids_))
        ibs.set_annot_name_rowids(aids_, [nid2] * len(aids_))

    # the next 2 become split cases
    # left = right
    # right = left + num_split_names
    for aids in modify_aids[3:5]:
        ibs.set_annot_name_rowids(aids, [split_nid] * len(aids))

    # left = right
    # right = left + num_combo_names
    # The final 3 are a combination case
    for aids in modify_aids[5:8]:
        aids_even = aids[::2]
        aids_odd = aids[1::2]
        ibs.set_annot_name_rowids(aids_even, [combo_nids[0]] * len(aids_even))
        ibs.set_annot_name_rowids(aids_odd, [combo_nids[1]] * len(aids_odd))

    final_result = ibs.unflat_map(ibs.get_annot_nids, modify_aids)
    print('final_result = %s' % (ub.repr2(final_result),))


def ensure_wilddogs():
    """ Ensures that you have the NAUT_test dataset """
    return ensure_db_from_url(const.ZIPPED_URLS.WDS)


def ensure_nauts():
    """ Ensures that you have the NAUT_test dataset """
    return ensure_db_from_url(const.ZIPPED_URLS.NAUTS)


def ensure_testdb2():
    zipped_db_url = 'https://wildbookiarepository.azureedge.net/databases/testdb2.tar.gz'
    return ensure_db_from_url(zipped_db_url)


def ensure_testdb_curvrank():
    return ensure_db_from_url(const.ZIPPED_URLS.DF_CURVRANK)


def ensure_testdb_orientation():
    return ensure_db_from_url(const.ZIPPED_URLS.ORIENTATION)


def ensure_testdb_identification_example():
    return ensure_db_from_url(const.ZIPPED_URLS.ID_EXAMPLE)


def ensure_testdb_kaggle7():
    return ensure_db_from_url(const.ZIPPED_URLS.K7_EXAMPLE)


def ensure_db_from_url(zipped_db_url):
    """ SeeAlso wbia.init.sysres """
    from wbia import sysres

    workdir = sysres.get_workdir()
    dbdir = ut.grab_zipped_url(
        zipped_url=zipped_db_url, ensure=True, download_dir=workdir
    )
    print('have %s=%r' % (zipped_db_url, dbdir,))
    return dbdir


def get_global_distinctiveness_modeldir(ensure=True):
    # DEPRICATE
    resource_dir = get_wbia_resource_dir()
    global_distinctdir = join(resource_dir, const.PATH_NAMES.distinctdir)
    if ensure:
        ut.ensuredir(global_distinctdir)
    return global_distinctdir


if __name__ == '__main__':
    """
    CommandLine:
        xdoctest -m wbia.init.sysres
    """
    import xdoctest

    xdoctest.doctest_module(__file__)
