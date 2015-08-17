# -*- coding: utf-8 -*-
"""
TODO: Rename to ibeis/init/commands.py

The AID configuration selection is getting a mjor update right now
"""
from __future__ import absolute_import, division, print_function
import utool as ut
import numpy as np  # NOQA
import six
from ibeis import params
(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[main_helpers]')


VERB_TESTDATA = ut.get_argflag(('--verbose-testdata', '--verbtd'))
VERB_MAIN_HELPERS = ut.get_argflag(('--verbose-main-helpers', '--verbmhelp')) or ut.VERBOSE or VERB_TESTDATA


def define_named_aid_cfgs():
    """
    Definitions for common aid configurations
    TODO: potentially move to experiment configs
    """
    from ibeis.experiments import annotation_configs
    named_defaults_dict = ut.dict_take(annotation_configs.__dict__, annotation_configs.TEST_NAMES)
    named_qcfg_defaults = dict(zip(annotation_configs.TEST_NAMES, ut.get_list_column(named_defaults_dict, 'qcfg')))
    named_dcfg_defaults = dict(zip(annotation_configs.TEST_NAMES, ut.get_list_column(named_defaults_dict, 'dcfg')))
    alias_keys = annotation_configs.alias_keys
    named_cfg_dict = {
        'qcfg': named_qcfg_defaults,
        'dcfg': named_dcfg_defaults,
    }
    return named_cfg_dict, alias_keys


def get_commandline_aidcfg():
    """
    Parse the command line for "THE NEW AND IMPROVED" cannonical annotation
    configuration dictionaries

    CommandLine:
        python -m ibeis.init.main_helpers --exec-get_commandline_aidcfg
        python -m ibeis.init.main_helpers --exec-get_commandline_aidcfg --qcfg default:shuffle=True,index=0:25 --dcfg default
        python -m ibeis.init.main_helpers --exec-get_commandline_aidcfg --qcfg default --dcfg default

        python -m ibeis.init.main_helpers --exec-get_commandline_aidcfg --qcfg controlled --dcfg controlled

        python -m ibeis.init.main_helpers --exec-get_commandline_aidcfg --acfg controlled

        python -m ibeis.init.main_helpers --exec-get_commandline_aidcfg --acfg varydbsize

        python -m ibeis.init.main_helpers --exec-get_commandline_aidcfg --acfg controlled:qindex=0:10


        --aidcfg=controlled=True,species=primary
        --aidcfg=controlled=True,species=primary,annot_per_name=2
        --aidcfg=controlled=True,species=primary,annot_per_name=3

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.init.main_helpers import *  # NOQA
        >>> aidcfg = get_commandline_aidcfg()
        >>> print('aidcfg = ' + ut.dict_str(aidcfg))
    """

    def parse_cfgstr_list2(cfgstr_list, named_dcfgs_dict, cfgtype=None, alias_keys=None):
        """
        Parse a genetic cfgstr --flag name1:custom_args1 name2:custom_args2
        """
        cfg_list = []
        for cfgstr in cfgstr_list:
            cfgstr_split = cfgstr.split(':')
            cfgname = cfgstr_split[0]
            cfg = named_dcfgs_dict[cfgname].copy()
            # Parse dict out of a string
            if len(cfgstr_split) > 1:
                cfgstr_options =  ':'.join(cfgstr_split[1:]).split(',')
                cfg_options = ut.parse_cfgstr_list(cfgstr_options, smartcast=True, oldmode=False)
            else:
                cfg_options = {}
            # Hack for q/d specific configs
            if cfgtype is not None:
                for key in list(cfg_options.keys()):
                    # check if key is nonstandard
                    if not (key in cfg or key in alias_keys):
                        # does removing prefix make it stanard?
                        prefix = cfgtype[0]
                        if key.startswith(prefix):
                            key_ = key[len(prefix):]
                            if key_ in cfg or key_ in alias_keys:
                                # remove prefix
                                cfg_options[key_] = cfg_options[key]
                        try:
                            assert key[1:] in cfg or key[1:] in alias_keys, 'key=%r, key[1:] =%r' % (key, key[1:] )
                        except AssertionError as ex:
                            ut.printex(ex, 'error', keys=['key', 'cfg', 'alias_keys'])
                            raise
                        del cfg_options[key]
            # Remap keynames based on aliases
            if alias_keys is not None:
                for key in alias_keys.keys():
                    if key in cfg_options:
                        # use standard new key
                        cfg_options[alias_keys[key]] = cfg_options[key]
                        # remove old alised key
                        del cfg_options[key]
            # Finalize configuration dict
            cfg = ut.update_existing(cfg, cfg_options, copy=True, assert_exists=True)
            cfg['_cfgtype'] = cfgtype
            cfg['_cfgname'] = cfgname
            cfg['_cfgstr'] = cfgstr
            cfg_list.append((cfgname, cfg))
            break  # FIXME: do more than one eventually
        return cfg

    named_cfg_dict, alias_keys = define_named_aid_cfgs()

    # Parse the cfgstr list from the command line
    qcfgstr_list, has_qcfg = ut.get_argval('--qcfg', type_=list, default=['default'], return_specified=True)
    dcfgstr_list, has_dcfg = ut.get_argval('--dcfg', type_=list, default=['default'], return_specified=True)

    if not has_qcfg and not has_dcfg:
        # TODO: Specify both with one flag
        acfgstr_list = ut.get_argval(('--aidcfg', '--acfg', '-a'), type_=list, default=['default'])
        aidcfg = {}
        aidcfg['qcfg'] = parse_cfgstr_list2(acfgstr_list, named_cfg_dict['qcfg'], 'qcfg', alias_keys)
        aidcfg['dcfg'] = parse_cfgstr_list2(acfgstr_list, named_cfg_dict['dcfg'], 'dcfg', alias_keys)
    else:
        aidcfg = {}
        aidcfg['qcfg'] = parse_cfgstr_list2(qcfgstr_list, named_cfg_dict['qcfg'], 'qcfg', alias_keys)
        aidcfg['dcfg'] = parse_cfgstr_list2(dcfgstr_list, named_cfg_dict['dcfg'], 'dcfg', alias_keys)
    return aidcfg


def ensure_flatiterable(input_):
    if isinstance(input_, six.string_types):
        input_ = ut.fuzzy_int(input_)
    if isinstance(input_, int) or not ut.isiterable(input_):
        return [input_]
    elif isinstance(input_, (list, tuple)):
        #print(input_)
        if len(input_) > 0 and ut.isiterable(input_[0]):
            return ut.flatten(input_)
        return input_
    else:
        raise TypeError('cannot ensure %r input_=%r is iterable', (type(input_), input_))


def ensure_flatlistlike(input_):
    #if isinstance(input_, slice):
    #    pass
    iter_ = ensure_flatiterable(input_)
    return list(iter_)


def get_test_qaids(ibs, default_qaids=None, return_annot_info=False, aidcfg=None):
    """
    Gets test annot_rowids based on command line arguments

    DEPRICATE

    Args:
        ibs (IBEISController):  ibeis controller object
        default_qaids (None): if list then used only if no other aids are available (default = [1])
           as a string it mimics the command line

    Returns:
        list: available_qaids

    CommandLine:
        python -m ibeis.init.main_helpers --test-get_test_qaids
        python -m ibeis.init.main_helpers --test-get_test_qaids --controlled --db PZ_Master0
        python -m ibeis.init.main_helpers --test-get_test_qaids --controlled --db PZ_Master0 --qaid 1
        python -m ibeis.init.main_helpers --test-get_test_qaids --allgt --db PZ_MTEST
        python -m ibeis.init.main_helpers --test-get_test_qaids --qaid 4 5 8  --verbmhelp
        python -m ibeis.init.main_helpers --test-get_test_qaids --controlled --db PZ_MTEST
        python -m ibeis.init.main_helpers --test-get_test_qaids --controlled --db PZ_MTEST --qaid 2 --verbmhelp
        python -m ibeis.init.main_helpers --test-get_test_qaids --controlled --db PZ_MTEST --qaid 2
        python -m ibeis.init.main_helpers --test-get_test_qaids --controlled --db PZ_Master0 --qindex 0:10 --verbmhelp
        python -m ibeis.init.main_helpers --exec-get_test_qaids --controlled --db PZ_Master0 --exec-mode
        python -m ibeis.init.main_helpers --exec-get_test_qaids --db testdb1 --allgt --qindex 0:256

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.init.main_helpers import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> default_qaids = None
        >>> available_qaids = get_test_qaids(ibs, default_qaids)
        >>> ibeis.other.dbinfo.get_dbinfo(ibs, aid_list=available_qaids, with_contrib=False, short=True)
        >>> result = 'available_qaids = ' + ut.obj_str(available_qaids, truncate=True, nl=False)
        >>> print('len(available_qaids) = %d' % len(available_qaids))
        >>> print(result)
        available_qaids = [1]
    """
    qaid_request_info = {}
    if VERB_MAIN_HELPERS:
        print('[get_test_qaids] + --- GET_TEST_QAIDS ---')

    # Old version of this function
    if VERB_MAIN_HELPERS:
        print('[get_test_qaids] + --- GET_TEST_QAIDS ---')
        print('[get_test_qaids] * default_qaids = %s' % (ut.obj_str(default_qaids, truncate=True, nl=False)))

    valid_aids = ibs.get_valid_aids()

    if len(valid_aids) == 0:
        print('[get_test_qaids] WARNING no annotations available')

    # ---- INCLUDING STEP
    if VERB_MAIN_HELPERS:
        print('[get_test_qaids] * include step')

    available_qaids = []

    #ut.get_argflag(('--all-cases', '--all'))
    #ut.get_argflag(('--all-gt-cases', '--allgt'))
    #ut.get_argflag(('--all-hard-cases', '--allhard'))
    #ut.get_argflag(('--qaid', '--qaids'))
    #ut.get_argflag('--controlled') or ut.get_argflag('--controlled_qaids')
    #not ut.get_argflag('--junk')

    ALL_CASES = params.args.all_cases or default_qaids == 'all'
    GT_CASES = params.args.all_gt_cases or default_qaids == 'gt'
    HARD_CASES = params.args.all_hard_cases or ut.get_argflag(('--all-hard-cases', '--allhard', '--hard'))
    NO_JUNK = not ut.get_argflag('--junk')
    CONTROLLED_CASES = ut.get_argflag('--controlled') or ut.get_argflag('--controlled_qaids')
    NO_REVIEWED = ut.get_argflag('--unreviewed')
    species = ut.get_argval('--species')
    QAID = params.args.qaid
    QINDEX = params.args.qindex
    QSHUFFLE = ut.get_argval('--qshuffle')

    if QAID is not None:
        if VERB_MAIN_HELPERS:
            print('[get_test_qaids] * Including cmdline specified qaids')
        try:
            args_qaid = ensure_flatlistlike(QAID)
        except Exception:
            args_qaid = QAID
        available_qaids.extend(args_qaid)
        qaid_request_info['custom_commandline'] = args_qaid

    if ALL_CASES:
        if VERB_MAIN_HELPERS:
            print('[get_test_qaids] * Including all qaids')
        available_qaids.extend(valid_aids)
        qaid_request_info['all_cases'] = True

    if HARD_CASES:
        if VERB_MAIN_HELPERS:
            print('[get_test_qaids] * Including hard qaids')
        is_hard_list = ibs.get_annot_is_hard(valid_aids)
        hard_aids = ut.filter_items(valid_aids, is_hard_list)
        available_qaids.extend(hard_aids)
        qaid_request_info['hard_cases'] = True

    if GT_CASES:
        if VERB_MAIN_HELPERS:
            print('[get_test_qaids] * Including groundtruth qaids')
        has_gt_list = ibs.get_annot_has_groundtruth(valid_aids)
        hasgt_aids = ut.filter_items(valid_aids, has_gt_list)
        print('[get_test_qaids] Adding all %d/%d ground-truthed test cases' % (len(hasgt_aids), len(valid_aids)))
        available_qaids.extend(hasgt_aids)
        qaid_request_info['gt_cases'] = True

    if CONTROLLED_CASES:
        if VERB_MAIN_HELPERS:
            print('[get_test_qaids] * Including controlled qaids')
        from ibeis import ibsfuncs
        # Override all other gts with controlled
        controlled_qaids = ibsfuncs.get_two_annots_per_name_and_singletons(ibs, onlygt=True)
        available_qaids.extend(controlled_qaids)
        qaid_request_info['controlled'] = True
    else:
        qaid_request_info['controlled'] = False

    # ---- CHECK_DEFAULTS QUERY
    if VERB_MAIN_HELPERS:
        print('[get_test_qaids] * len(available_qaids) = %r' % (len(available_qaids)))

    if len(available_qaids) == 0:
        print('[get_test_qaids] * ... defaulting, no available qaids on command line.')
        if default_qaids is None:
            default_qaids = valid_aids[0:1]
            qaid_request_info['default_one'] = True
        elif isinstance(default_qaids, six.string_types):
            if default_qaids == 'gt' or default_qaids == 'allgt':
                default_qaids = ibs.get_valid_aids(hasgt=True)
                qaid_request_info['default_gt'] = True
        available_qaids = default_qaids
    else:
        if VERB_MAIN_HELPERS:
            print('[get_test_qaids] * ... not defaulting')

    available_qaids = ut.unique_keep_order2(available_qaids)

    # ---- EXCLUSION STEP
    if VERB_MAIN_HELPERS:
        print('[get_test_qaids] * len(available_qaids) = %r' % (len(available_qaids)))
        print('[get_test_qaids] * exclude step')

    if NO_JUNK:
        if VERB_MAIN_HELPERS:
            print('[get_test_qaids] * Filtering junk')
        available_qaids = ibs.filter_junk_annotations(available_qaids)
        qaid_request_info['has_junk'] = False

    if NO_REVIEWED:
        if VERB_MAIN_HELPERS:
            print('[get_test_qaids] * Filtering unreviewed')
        isreviewed_list = ibs.get_annot_has_reviewed_matching_aids(available_qaids)
        available_qaids = ut.filterfalse_items(available_qaids, isreviewed_list)
        qaid_request_info['has_unreviewed'] = False

    if species is not None:
        if species == 'primary':
            if VERB_MAIN_HELPERS:
                print('[get_test_qaids] * Finiding primary species')
            #species = ibs.get_primary_database_species(available_qaids)
            species = ibs.get_primary_database_species()
            qaid_request_info['primary_species'] = True

        if VERB_MAIN_HELPERS:
            print('[get_test_qaids] * Filtering to species=%r' % (species,))
        isvalid_list = np.array(ibs.get_annot_species(available_qaids)) == species
        available_qaids = ut.filter_items(available_qaids, isvalid_list)
        qaid_request_info['species_filter'] = species

    if VERB_MAIN_HELPERS:
        print('[get_test_qaids] * len(available_qaids) = %r' % (len(available_qaids)))
        print('[get_test_qaids] * subindex step')

    # ---- INDEX SUBSET

    #ut.get_argval('--qshuffle')
    if QSHUFFLE:
        # Determenistic shuffling
        available_qaids = ut.list_take(available_qaids, ut.random_indexes(len(available_qaids), seed=42))
        qaid_request_info['shuffled'] = True

    # Sample a large pool of chosen query qindexes
    if QINDEX is not None:
        # FIXME: should use a slice of the list or a sublist
        qindexes = ensure_flatlistlike(QINDEX)
        _test_qaids = [available_qaids[qx] for qx in qindexes if qx < len(available_qaids)]
        print('[get_test_qaids] Chose subset of size %d/%d' % (len(_test_qaids), len(available_qaids)))
        available_qaids = _test_qaids
        qaid_request_info['subset'] = qindexes

    if VERB_MAIN_HELPERS:
        print('[get_test_qaids] * len(available_qaids) = %r' % (len(available_qaids)))
        print('[get_test_qaids] L ___ GET_TEST_QAIDS ___')
    if return_annot_info:
        return available_qaids, qaid_request_info
    else:
        return available_qaids


def get_test_daids(ibs, default_daids='all', qaid_list=None, return_annot_info=False, aidcfg=None):
    """ Gets database annot_rowids based on command line arguments

    DEPRICATE

    CommandLine:
        python dev.py --db PZ_MTEST -t best --exclude-query --qaid 72 -r 0 -c 0 --show --va --vf --dump-extra

    Args:
        ibs (IBEISController):  ibeis controller object
        default_daids (str): (default = 'all')
        qaid_list (list): list of chosen qaids that may affect daids (default = None)

    Returns:
        list: available_daids

    CommandLine:
        python -m ibeis.init.main_helpers --test-get_test_daids
        python -m ibeis.init.main_helpers --test-get_test_daids --db PZ_MTEST  --verbmhelp
        python -m ibeis.init.main_helpers --test-get_test_daids --db PZ_MTEST --exclude-query
        python -m ibeis.init.main_helpers --test-get_test_daids --db PZ_MTEST --daid-exclude 2 3 4
        python -m ibeis.init.main_helpers --test-get_test_daids --db PZ_MTEST --species=zebra_grevys
        python -m ibeis.init.main_helpers --test-get_test_daids --db PZ_Master0 --species=zebra_grevys
        python -m ibeis.init.main_helpers --test-get_test_daids --db PZ_Master0 --controlled --verbmhelp
        python -m ibeis.init.main_helpers --exec-get_test_daids --controlled --db PZ_Master0 --exec-mode

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.init.main_helpers import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> default_daids = 'all'
        >>> qaid_list = [1]
        >>> available_daids = get_test_daids(ibs, default_daids, qaid_list)
        >>> ibeis.other.dbinfo.get_dbinfo(ibs, aid_list=available_daids, with_contrib=False, short=True)
        >>> result = 'available_daids = ' + ut.obj_str(available_daids, truncate=True, nl=False)
        >>> print('len(available_daids) %d' % len(available_daids))
        >>> print(result)
        available_daids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    """
    daid_request_info = {}

    if VERB_MAIN_HELPERS:
        print('[get_test_daids] + --- GET_TEST_DAIDS ---')
        print('[get_test_daids] * default_daids = %s' % (ut.obj_str(default_daids, truncate=True, nl=False)))
        print('[get_test_daids] * qaid_list = %s' % (ut.obj_str(qaid_list, truncate=True, nl=False)))

    # ---- INCLUDING STEP
    if VERB_MAIN_HELPERS:
        print('[get_test_daids] * include step')

    available_daids = []

    CONTROLLED_CASES = ut.get_argflag('--controlled') or ut.get_argflag('--controlled_daids')
    DSHUFFLE = ut.get_argval('--dshuffle')
    DINDEX = params.args.dindex
    NO_JUNK = not ut.get_argflag('--junk')
    EXCLUDE_QUERY = ut.get_argflag('--exclude-query')
    daids_exclude = params.args.daid_exclude

    if CONTROLLED_CASES:
        print('[get_test_daids] * Including controlled daids')
        from ibeis import ibsfuncs
        controlled_daids = ibsfuncs.get_two_annots_per_name_and_singletons(ibs, onlygt=False)
        available_daids.extend(controlled_daids)
        daid_request_info['controlled'] = True
    else:
        daid_request_info['controlled'] = False

    # ---- CHECK_DEFAULTS DATA
    if VERB_MAIN_HELPERS:
        print('[get_test_daids] * len(available_daids) = %r' % (len(available_daids)))

    if len(available_daids) == 0:
        print('[get_test_daids] * ... defaulting, no available daids on command line.')
        if isinstance(default_daids, six.string_types):
            if default_daids == 'all':
                default_daids = ibs.get_valid_aids()
                daid_request_info['default_daids'] = 'all'
            elif default_daids == 'gt':
                default_daids = ut.flatten(ibs.get_annot_groundtruth(qaid_list))
                daid_request_info['default_daids'] = 'gt'
        #available_qaids = valid_aids[0:1]
        assert not isinstance(available_daids, six.string_types)
        available_daids = default_daids
    else:
        if VERB_MAIN_HELPERS:
            print('[get_test_daids] * ... not defaulting')

    available_daids = ut.unique_keep_order2(available_daids)

    # ---- EXCLUSION STEP
    if VERB_MAIN_HELPERS:
        print('[get_test_daids] * len(available_daids) = %r' % (len(available_daids)))
        print('[get_test_daids] * exclude step')

    species = ut.get_argval('--species', type_=str, default=None)

    if NO_JUNK:
        if VERB_MAIN_HELPERS:
            print('[get_test_daids] * Filtering junk')
        available_daids = ibs.filter_junk_annotations(available_daids)

    if EXCLUDE_QUERY:
        if VERB_MAIN_HELPERS:
            print('[get_test_daids] * Excluding query qaids')
        assert qaid_list is not None, 'must specify qaids to exclude'
        available_daids = ut.setdiff_ordered(available_daids, qaid_list)

    if daids_exclude is not None:
        if VERB_MAIN_HELPERS:
            print('[get_test_daids] * Excluding specified daids')
        available_daids = ut.setdiff_ordered(available_daids, daids_exclude)

    if species is not None:
        if species == 'primary':
            if VERB_MAIN_HELPERS:
                print('[get_test_qaids] * Finiding primary species')
            #species = ibs.get_primary_database_species(available_daids)
            species = ibs.get_primary_database_species()
        if VERB_MAIN_HELPERS:
            print('[get_test_daids] * Filtering to species=%r' % (species,))
        import numpy as np
        isvalid_list = np.array(ibs.get_annot_species(available_daids)) == species
        available_daids = ut.filter_items(available_daids, isvalid_list)

    # ---- SUBINDEXING STEP
    if VERB_MAIN_HELPERS:
        print('[get_test_daids] * len(available_daids) = %r' % (len(available_daids)))
        print('[get_test_daids] * subindex step')

    #ut.get_argval('--qshuffle')
    if DSHUFFLE:
        # Determenistic shuffling
        available_daids = ut.list_take(available_daids, ut.random_indexes(len(available_daids), seed=43))
        daid_request_info['shuffled'] = True

    if DINDEX is not None:
        dindexes = ensure_flatlistlike(DINDEX)
        _test_daids = [available_daids[dx] for dx in dindexes if dx < len(available_daids)]
        print('[get_test_daids] Chose subset of size %d/%d' % (len(_test_daids), len(available_daids)))
        available_daids = _test_daids

    if VERB_MAIN_HELPERS:
        print('[get_test_daids] * len(available_daids) = %r' % (len(available_daids)))
        print('[get_test_daids] L ___ GET_TEST_DAIDS ___')

    if return_annot_info:
        return available_daids, daid_request_info
    else:
        return available_daids


#def expand_aidcfg_dict(ibs, aidcfg=None, reference_aids=None):
#    """
#    DEPRICATE
#    """

#    # ---- INCLUDING STEP
#    available_aids = expand_to_default_aids(ibs, aidcfg)

#    # ---- FILTERING STEP
#    available_aids = filter_independent_properties(ibs, available_aids, aidcfg)

#    # ---- FILTERING WITH RESPECT TO WHAT GROUNDTRUTH IS ABAILABLE (what the daid will contain)

#    available_aids = filter_reference_properties(ibs, available_aids, aidcfg, reference_aids)

#    # ---- SAMPLE SELECTION
#    available_aids = sample_available_aids(ibs, available_aids, aidcfg, reference_aids)

#    return available_aids

## ------
#    if OLD:
#        # extract qaid list
#        if VERB_MAIN_HELPERS:
#            print('\n[expand_aidcfg] + --- GET_TEST_QAIDS ---')
#        qaid_list = expand_aidcfg_dict(ibs, aidcfg=qcfg)
#        if VERB_MAIN_HELPERS:
#            print('[expand_aidcfg] L ___ GET_TEST_QAIDS')

#        # extract daid list
#        if VERB_MAIN_HELPERS:
#            print('[expand_aidcfg] + --- GET_TEST_DAIDS ---')
#        daid_list = expand_aidcfg_dict(ibs, aidcfg=dcfg, reference_aids=qaid_list)
#        if VERB_MAIN_HELPERS:
#            print('[expand_aidcfg] L ___ GET_TEST_DAIDS \n')
#    else:
