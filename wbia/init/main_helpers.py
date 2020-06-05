# -*- coding: utf-8 -*-
"""
This module defines helper functions to access common input needed to test many
functions. These functions give a rich command line interface to specifically
select subsets of annotations, pipeline configurations, and other filters.

TODO: standardize function signatures
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import six

# from wbia.init import old_main_helpers
(print, rrr, profile) = ut.inject2(__name__, '[main_helpers]')


# DEPRICATE
# get_test_daids = old_main_helpers.get_test_daids
# get_test_qaids = old_main_helpers.get_test_qaids

VERB_TESTDATA, VERYVERB_TESTDATA = ut.get_verbflag('testdata', 'td')
VERYVERB_MAIN_HELPERS = VERYVERB_TESTDATA
VERB_MAIN_HELPERS = VERB_TESTDATA


def testdata_filtcfg(default=None):
    from wbia.expt import cfghelpers

    print('[main_helpers] testdata_filtcfg')
    if default is None:
        default = ['']
    filt_cfg = cfghelpers.parse_argv_cfg(('--filt', '-f'), default=default)[0]
    return filt_cfg


def testdata_expts(
    defaultdb='testdb1',
    default_acfgstr_name_list=['default:qindex=0:10:4,dindex=0:20'],
    default_test_cfg_name_list=['default'],
    a=None,
    t=None,
    p=None,
    qaid_override=None,
    daid_override=None,
    initial_aids=None,
    use_cache=None,
    dbdir=None,
    ibs=None,
):
    r"""
    Use this if you want data from an experiment.
    Command line interface to quickly get testdata for test_results.

    Command line flags can be used to specify db, aidcfg, pipecfg, qaid
    override, daid override (and maybe initial aids).


    CommandLine:
        python -m wbia.init.main_helpers testdata_expts

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.other.dbinfo import *  # NOQA
        >>> import wbia
        >>> ibs, testres = wbia.testdata_expts(defaultdb='pz_mtest',
        >>>                                     a='timectrl:qsize=2',
        >>>                                     t='invar:ai=[false],ri=false',
        >>>                                     use_cache=false)
        >>> print('testres = %r' % (testres,))
    """
    if ut.VERBOSE:
        print('[main_helpers] testdata_expts')
    import wbia
    from wbia.expt import harness

    if a is not None:
        default_acfgstr_name_list = a
    if t is not None and p is None:
        p = t
    if p is not None:
        default_test_cfg_name_list = p

    if isinstance(default_acfgstr_name_list, six.string_types):
        default_acfgstr_name_list = [default_acfgstr_name_list]
    if isinstance(default_test_cfg_name_list, six.string_types):
        default_test_cfg_name_list = [default_test_cfg_name_list]

    # from wbia.expt import experiment_helpers
    if dbdir is not None:
        dbdir = ut.truepath(dbdir)
    if ibs is None:
        ibs = wbia.opendb(defaultdb=defaultdb, dbdir=dbdir)
    acfg_name_list = ut.get_argval(
        ('--aidcfg', '--acfg', '-a'), type_=list, default=default_acfgstr_name_list
    )
    test_cfg_name_list = ut.get_argval(
        ('-t', '-p'), type_=list, default=default_test_cfg_name_list
    )
    daid_override = ut.get_argval(
        ('--daid-override', '--daids-override'), type_=list, default=daid_override
    )
    qaid_override = ut.get_argval(
        ('--qaid', '--qaids-override', '--qaid-override'),
        type_=list,
        default=qaid_override,
    )

    # Hack a cache here
    use_bulk_cache = not ut.get_argflag(('--nocache', '--nocache-hs'))
    use_bulk_cache &= ut.is_developer()
    if use_cache is not None:
        use_bulk_cache &= use_cache
    use_bulk_cache &= False
    # use_bulk_cache = True
    if use_bulk_cache:
        from os.path import dirname

        cache_dir = ut.ensuredir((dirname(ut.get_module_dir(wbia)), 'BULK_TESTRES'))
        _cache_wrp = ut.cached_func('testreslist', cache_dir=cache_dir)
        _load_testres = _cache_wrp(harness.run_expt)
    else:
        _load_testres = harness.run_expt

    testres = _load_testres(
        ibs,
        acfg_name_list,
        test_cfg_name_list,
        qaid_override=qaid_override,
        daid_override=daid_override,
        initial_aids=initial_aids,
        use_cache=use_cache,
    )
    # testres = test_result.combine_testres_list(ibs, testres_list)

    if ut.VERBOSE:
        print(testres)
    return ibs, testres


def testdata_aids(
    defaultdb=None,
    a=None,
    adefault='default',
    ibs=None,
    return_acfg=False,
    verbose=None,
    default_aids=None,
    default_set='qcfg',
):
    r"""
    Grabs default testdata for functions, but is command line overrideable

    CommandLine:
        python -m wbia testdata_aids --verbtd --db PZ_ViewPoints
        python -m wbia testdata_aids --verbtd --db NNP_Master3 -a is_known=True,view_pername='#primary>0&#primary1>=1'
        python -m wbia testdata_aids --verbtd --db PZ_Master1 -a default:is_known=True,view_pername='#primary>0&#primary1>=1'
        python -m wbia testdata_aids --verbtd --db PZ_Master1 -a default:species=primary,minqual=ok --verbtd
        python -m wbia.other.dbinfo --test-latex_dbstats --dblist
        python -m wbia testdata_aids --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.init.main_helpers import *  # NOQA
        >>> from wbia.expt import annotation_configs
        >>> import wbia
        >>> #ibs = wbia.opendb(defaultdb='PZ_ViewPoints')
        >>> ibs = wbia.opendb(defaultdb='testdb1')
        >>> a = None
        >>> adefault = 'default:is_known=True'
        >>> aids, aidcfg = testdata_aids(ibs=ibs, a=a, adefault=adefault, return_acfg=True)
        >>> print('\n RESULT:')
        >>> annotation_configs.print_acfg(aidcfg, aids, ibs, per_name_vpedge=None)
    """
    import wbia
    from wbia.init import filter_annots
    from wbia.expt import annotation_configs
    from wbia.expt import cfghelpers

    if verbose is None or verbose >= 1:
        print('[main_helpers] testdata_aids')
    if a is None:
        a = adefault
    a, _specified_a = ut.get_argval(
        ('--aidcfg', '--acfg', '-a'), type_=str, default=a, return_was_specified=True
    )
    return_ibs = False
    if ibs is None:
        return_ibs = True
        if defaultdb is None:
            defaultdb = 'testdb1'
        ibs = wbia.opendb(defaultdb=defaultdb)
    named_defaults_dict = ut.dict_take(
        annotation_configs.__dict__, annotation_configs.TEST_NAMES
    )

    named_acfg_defaults = dict(
        zip(
            annotation_configs.TEST_NAMES,
            ut.get_list_column(named_defaults_dict, default_set),
        )
    )
    # Allow command line override
    aids, _specified_aids = ut.get_argval(
        ('--aid', '--aids'), type_=list, default=default_aids, return_was_specified=True
    )

    aidcfg = None
    have_aids = aids is not None
    need_expand = (not have_aids) or (_specified_a and not _specified_aids)
    # (not aid) or (sa and (not said))
    if need_expand:
        # base_cfg = annotation_configs.single_default
        aidcfg_combo_list = cfghelpers.parse_cfgstr_list2(
            [a],
            named_acfg_defaults,
            'acfg',
            annotation_configs.ALIAS_KEYS,
            expand_nested=False,
            is_nestedcfgtype=False,
        )
        aidcfg_combo = aidcfg_combo_list[0]
        if len(aidcfg_combo_list) != 1:
            raise AssertionError('Error: combinations not handled for single cfg setting')
        if len(aidcfg_combo) != 1:
            raise AssertionError('Error: combinations not handled for single cfg setting')
        aidcfg = aidcfg_combo[0]
        aids = filter_annots.expand_single_acfg(ibs, aidcfg, verbose=verbose)
    if return_ibs:
        return ibs, aids
    if return_acfg:
        return aids, aidcfg
    else:
        return aids


def testdata_pipecfg(p=None, t=None, ibs=None, verbose=None):
    r"""
    Returns:
        dict: pcfgdict

    CommandLine:
        python -m wbia testdata_pipecfg
        python -m wbia testdata_pipecfg -t default:AI=False

    Ignore:
        from jedi.evaluate import docstrings
        script = jedi.Script(ut.readfrom(main_helpers.__file__))
        mod = script._get_module()
        func = mod.names_dict['testdata_pipecfg'][0].parent
        docstrings.find_return_types(script._evaluator, func)

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.init.main_helpers import *  # NOQA
        >>> pcfgdict = testdata_pipecfg()
        >>> result = ('pcfgdict = %s' % (ut.repr2(pcfgdict),))
        >>> print(result)
    """
    if verbose is None or verbose >= 1:
        print('[main_helpers] testdata_pipecfg')
    if t is not None and p is None:
        p = t
        print('WARNING DO NOT USE t. Use p instead')
    if p is None:
        p = ['default']

    from wbia.expt import experiment_helpers

    test_cfg_name_list, _spec = ut.get_argval(
        ('-t', '-p'), type_=list, default=p, return_was_specified=True
    )
    if not _spec and isinstance(p, dict):
        # allow explict default spec
        return p
    pcfgdict_list = experiment_helpers.get_pipecfg_list(test_cfg_name_list, ibs=ibs)[0]
    assert len(pcfgdict_list) == 1, 'can only specify one pipeline config here'
    pcfgdict = pcfgdict_list[0]
    return pcfgdict


def testdata_expanded_aids(
    defaultdb=None,
    a=None,
    ibs=None,
    default_qaids=None,
    default_daids=None,
    qaid_override=None,
    daid_override=None,
    return_annot_info=False,
    verbose=None,
    use_cache=None,
):
    r"""
    Args:
        default_qaids (list): (default = [1])
        default_daids (str): (default = 'all')
        defaultdb (str): (default = 'testdb1')
        ibs (IBEISController):  wbia controller object(default = None)
        verbose (bool):  verbosity flag(default = False)
        return_annot_info (bool): (default = False)

    Returns:
        ibs, qaid_list, daid_list, annot_info:

    CommandLine:
        python -m wbia.init.main_helpers testdata_expanded_aids
        python -m wbia.init.main_helpers testdata_expanded_aids --db PZ_MTEST --acfg default:index=0:25 --verbose-testdata
        python -m wbia.init.main_helpers testdata_expanded_aids --db PZ_MTEST --qaid 3
        python -m wbia.init.main_helpers testdata_expanded_aids --db GZ_ALL --acfg ctrl --verbose-testdata

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.init.main_helpers import *  # NOQA
        >>> import wbia
        >>> from wbia.expt import annotation_configs
        >>> ibs, qaid_list, daid_list, aidcfg = testdata_expanded_aids(return_annot_info=True)
        >>> print('Printing annot config')
        >>> annotation_configs.print_acfg(aidcfg)
        >>> print('Printing annotconfig stats')
        >>> ibs.print_annotconfig_stats(qaid_list, daid_list)
        >>> print('Combined annotconfig stats')
        >>> ibs.print_annot_stats(qaid_list + daid_list, viewcode_isect=True)
        >>> print('qaid_list = %r' % (qaid_list,))
    """
    if verbose is None:
        verbose = 1

    if verbose:
        print('[main_helpers] testdata_expanded_aids')

    default_qaids = ut.get_argval(
        ('--qaid', '--qaid-override'), type_=list, default=default_qaids
    )
    if default_qaids is None:
        default_qaids = [1]

    if defaultdb is None:
        defaultdb = 'testdb1'
    import wbia

    if ibs is None:
        ibs = wbia.opendb(defaultdb=defaultdb)

    # TODO: rectify command line with function arguments
    from wbia.expt import experiment_helpers

    _specified2 = True
    if a is None:
        _specified2 = False
        a = ['default']
    if isinstance(a, six.string_types):
        a = [a]
    aidcfg_name_list, _specified = ut.get_argval(
        ('--aidcfg', '--acfg', '-a'), type_=list, default=a, return_specified=True
    )

    if not _specified:
        # Allow a to be specified an explicit default
        if len(a) == 2:
            qaids, daids = a
            if ut.is_int(qaids[0]) and ut.is_int(daids[0]):
                if return_annot_info:
                    return ibs, qaids, daids, None
                else:
                    return ibs, qaids, daids

    acfg_list, expanded_aids_list = experiment_helpers.get_annotcfg_list(
        ibs,
        aidcfg_name_list,
        qaid_override=qaid_override,
        use_cache=use_cache,
        daid_override=daid_override,
        verbose=max(0, verbose - 1),
    )

    # aidcfg = old_main_helpers.get_commandline_aidcfg()
    assert len(acfg_list) == 1, (
        'multiple acfgs specified, but this function'
        'is built to return only 1. len(acfg_list)=%r'
    ) % (len(acfg_list),)
    aidcfg = acfg_list[0]

    qaid_list, daid_list = expanded_aids_list[0]

    if not (_specified or _specified2):
        # hack
        if default_qaids is not None and qaid_override is None:
            qaid_list = default_qaids
        if default_daids is not None and daid_override is None:
            daid_list = default_daids

    if ut.VERYVERBOSE:
        ibs.print_annotconfig_stats(qaid_list, daid_list)
        # wbia.other.dbinfo.print_qd_info(ibs, qaid_list, daid_list, verbose=True)
    if return_annot_info:
        return ibs, qaid_list, daid_list, aidcfg
    else:
        return ibs, qaid_list, daid_list


def testdata_qreq_(
    p=None,
    a=None,
    t=None,
    default_qaids=None,
    default_daids=None,
    custom_nid_lookup=None,
    verbose=None,
    **kwargs,
):
    r"""
    Args:
        p (None): (default = None)
        a (None): (default = None)
        t (None): (default = None)
        default_qaids (None): (default = None)
        default_daids (None): (default = None)

    Kwargs:
        defaultdb, ibs, qaid_override, daid_override, return_annot_info,
        verbose, use_cache

    Returns:
        wbia.QueryRequest: qreq_ -  query request object with hyper-parameters

    CommandLine:
        python -m wbia testdata_qreq_ --show --qaid 3

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.init.main_helpers import *  # NOQA
        >>> kwargs = {}
        >>> p = None
        >>> a = None
        >>> qreq_ = testdata_qreq_(p)
        >>> result = ('qreq_ = %s' % (str(qreq_),))
    """
    if verbose is None or verbose >= 1:
        print('[main_helpers] testdata_qreq_')
    if t is not None and p is None:
        p = t
    if p is None:
        p = ['default']

    ibs, qaids, daids, acfg = testdata_expanded_aids(
        a=a,
        return_annot_info=True,
        default_qaids=default_qaids,
        default_daids=default_daids,
        verbose=verbose,
        **kwargs,
    )
    pcfgdict = testdata_pipecfg(t=p, ibs=ibs, verbose=verbose)
    qreq_ = ibs.new_query_request(
        qaids,
        daids,
        cfgdict=pcfgdict,
        custom_nid_lookup=custom_nid_lookup,
        verbose=verbose,
    )
    # Maintain regen command info: TODO: generalize and integrate
    if acfg is not None:
        qreq_._regen_info = {
            '_acfgstr': acfg['qcfg']['_cfgstr'],
            '_pcfgstr': pcfgdict['_cfgstr'],
            'dbname': ibs.get_dbname(),
        }
    else:
        qreq_._regen_info = None
    return qreq_


def testdata_cmlist(
    defaultdb=None,
    default_qaids=None,
    default_daids=None,
    t=None,
    p=None,
    a=None,
    verbose=None,
):
    """
    Returns:
        list, wbia.QueryRequest: cm_list, qreq_
    """
    if verbose is None or verbose >= 1:
        print('[main_helpers] testdata_cmlist')
    qreq_ = testdata_qreq_(
        defaultdb=defaultdb,
        default_qaids=default_qaids,
        default_daids=default_daids,
        t=t,
        p=p,
        a=a,
    )
    cm_list = qreq_.execute()
    return cm_list, qreq_


def testdata_cm(
    defaultdb=None, default_qaids=None, default_daids=None, t=None, p=None, a=None
):
    r"""
    CommandLine:
        python -m wbia.init.main_helpers --test-testdata_cm
        python -m wbia.init.main_helpers --test-testdata_cm --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.init.main_helpers import *  # NOQA
        >>> cm, qreq_ = testdata_cm()
        >>> cm.print_csv(ibs=qreq_.ibs)
        >>> ut.quit_if_noshow()
        >>> cm.show_single_annotmatch(qreq_, 2)
        >>> ut.show_if_requested()
    """
    print('[main_helpers] testdata_cm')
    cm_list, qreq_ = testdata_cmlist(
        defaultdb=defaultdb,
        default_daids=default_daids,
        default_qaids=default_qaids,
        t=t,
        p=p,
        a=a,
    )
    qaids = qreq_.qaids
    print('qaids = %r' % (qaids,))
    assert len(qaids) == 1, 'only one qaid for this tests, qaids=%r' % (qaids,)
    cm = cm_list[0]
    return cm, qreq_


def monkeypatch_encounters(ibs, aids, cache=None, **kwargs):
    """
    Hacks in a temporary custom definition of encounters for this controller

    50 days for PZ_MTEST
    kwargs = dict(days=50)

    if False:
        name_mindeltas = []
        for name in annots.group_items(annots.nids).values():
            times = name.image_unixtimes_asfloat
            deltas = [ut.unixtime_to_timedelta(np.abs(t1 - t2))
                      for t1, t2 in ut.combinations(times, 2)]
            if deltas:
                name_mindeltas.append(min(deltas))
        print(ut.repr3(ut.lmap(ut.get_timedelta_str,
                               sorted(name_mindeltas))))
    """
    from wbia.algo.preproc.occurrence_blackbox import cluster_timespace_sec
    import numpy as np
    import datetime

    if len(aids) == 0:
        return
    annots = ibs.annots(sorted(set(aids)))
    thresh_sec = datetime.timedelta(**kwargs).total_seconds()
    # thresh_sec = datetime.timedelta(minutes=30).seconds

    if cache is None:
        cache = True
        # cache = len(aids) > 200
    cfgstr = str(ut.combine_uuids(annots.visual_uuids)) + str(thresh_sec)
    cacher = ut.Cacher('occurrence_labels', cfgstr=cfgstr, enabled=cache)
    data = cacher.tryload()
    if data is None:
        print('Computing occurrences for monkey patch for %d aids' % (len(aids)))
        posixtimes = annots.image_unixtimes_asfloat
        latlons = annots.gps
        data = cluster_timespace_sec(
            posixtimes, latlons, thresh_sec=thresh_sec, km_per_sec=0.002
        )
        cacher.save(data)
    occurrence_ids = data
    if occurrence_ids is None:
        # return
        # each annot is its own occurrence
        occurrence_ids = list(range(len(annots)))

    ndec = int(np.ceil(np.log10(max(occurrence_ids))))
    suffmt = '-monkey-occur%0' + str(ndec) + 'd'
    encounter_labels = [n + suffmt % (o,) for o, n in zip(occurrence_ids, annots.names)]
    occurrence_labels = [suffmt[1:] % (o,) for o in occurrence_ids]
    enc_lookup = ut.dzip(annots.aids, encounter_labels)
    occur_lookup = ut.dzip(annots.aids, occurrence_labels)

    # annots_per_enc = ut.dict_hist(encounter_labels, ordered=True)
    # ut.get_stats(list(annots_per_enc.values()))

    # encounters = ibs._annot_groups(annots.group(encounter_labels)[1])
    # enc_names = ut.take_column(encounters.nids, 0)
    # name_to_encounters = ut.group_items(encounters, enc_names)

    # print('name_to_encounters = %s' % (ut.repr3(name_to_encounters)),)
    # print('Names to num encounters')
    # name_to_num_enc = ut.dict_hist(
    #     ut.map_dict_vals(len, name_to_encounters).values())

    # monkey patch to override encounter info
    def _monkey_get_annot_occurrence_text(ibs, aids):
        return ut.dict_take(occur_lookup, aids)

    def _monkey_get_annot_encounter_text(ibs, aids):
        return ut.dict_take(enc_lookup, aids)

    ut.inject_func_as_method(
        ibs, _monkey_get_annot_encounter_text, 'get_annot_encounter_text', force=True
    )
    ut.inject_func_as_method(
        ibs, _monkey_get_annot_occurrence_text, 'get_annot_occurrence_text', force=True
    )


def unmonkeypatch_encounters(ibs):
    from wbia.other import ibsfuncs

    ut.inject_func_as_method(
        ibs, ibsfuncs.get_annot_encounter_text, 'get_annot_encounter_text', force=True
    )
    ut.inject_func_as_method(
        ibs, ibsfuncs.get_annot_occurrence_text, 'get_annot_occurrence_text', force=True
    )


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.init.main_helpers
        python -m wbia.init.main_helpers --allexamples
        python -m wbia.init.main_helpers --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
