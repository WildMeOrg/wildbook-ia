# -*- coding: utf-8 -*-
"""
The AID configuration selection is getting a mjor update right now
"""
from __future__ import absolute_import, division, print_function
import utool as ut
import six  # NOQA
from ibeis.init import old_main_helpers
(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[main_helpers]')


# DEPRICATE
get_test_daids = old_main_helpers.get_test_daids
get_test_qaids = old_main_helpers.get_test_qaids

VERB_TESTDATA, VERYVERB_TESTDATA = ut.get_verbflag('testdata', 'td')
VERYVERB_MAIN_HELPERS = VERYVERB_TESTDATA
VERB_MAIN_HELPERS = VERB_TESTDATA

#VERB_TESTDATA = ut.get_argflag(('--verbose-testdata', '--verbtd')) or VERYVERB_TESTDATA
#VERB_MAIN_HELPERS = ut.get_argflag(('--verbose-main-helpers', '--verbmhelp')) or ut.VERBOSE or VERB_TESTDATA


def testdata_pipecfg():
    r"""
    Returns:
        dict: pcfgdict

    CommandLine:
        python -m ibeis.init.main_helpers --exec-testdata_pipecfg

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.init.main_helpers import *  # NOQA
        >>> pcfgdict = testdata_pipecfg()
        >>> result = ('pcfgdict = %s' % (str(pcfgdict),))
        >>> print(result)
    """
    from ibeis.experiments import experiment_helpers
    test_cfg_name_list = ut.get_argval('-t', type_=list, default=['default'])
    pcfgdict_list = experiment_helpers.get_pipecfg_list(test_cfg_name_list)[0]
    assert len(pcfgdict_list) == 1, 'can only specify one pipeline config here'
    pcfgdict = pcfgdict_list[0]
    return pcfgdict


def testdata_filtcfg(default=None):
    from ibeis.experiments import cfghelpers
    if default is None:
        default = ['']
    filt_cfg = cfghelpers.parse_argv_cfg(('--filt', '-f'), default=default)[0]
    return filt_cfg


def testdata_qres(defaultdb='testdb1'):
    r"""
    Args:
        defaultdb (str): (default = 'testdb1')

    Returns:
        tuple: (ibs, test_result)

    CommandLine:
        python -m ibeis.init.main_helpers --exec-testdata_qres
        python -m ibeis.init.main_helpers --exec-testdata_qres --qaid 1

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.init.main_helpers import *  # NOQA
        >>> defaultdb = 'testdb1'
        >>> (ibs, qreq_, qres) = testdata_qres(defaultdb)
        >>> result = ('(ibs, qreq_, qres) = %s' % (str((ibs, qreq_, qres)),))
        >>> print(result)
    """
    ibs, qaids, daids = testdata_ibeis(defaultdb=defaultdb)
    pcfgdict = testdata_pipecfg()
    qreq_ = ibs.new_query_request(qaids, daids, cfgdict=pcfgdict)
    print('qaids = %r' % (qaids,))
    assert len(qaids) == 1, 'only one qaid for this tests'
    qres = qreq_.load_cached_qres(qaids[0])
    print('qreq_ = %r' % (qreq_,))
    return ibs, qreq_, qres


def testdata_expts(defaultdb='testdb1',
                   default_acfgstr_name_list=['default'],
                   #default_acfgstr_name_list=['controlled:qsize=20,dper_name=1,dsize=10',
                   #                           'controlled:qsize=20,dper_name=10,dsize=100'],
                   #default_test_cfg_name_list=['default', 'default:fg_on=False']
                   default_test_cfg_name_list=['default'],
                   a=None,
                   t=None,
                   qaid_override=None,
                   ):
    """
    Command line interface to quickly get testdata for test_results
    """
    import ibeis
    from ibeis.experiments import experiment_harness
    from ibeis.experiments import experiment_storage
    if a is not None:
        default_acfgstr_name_list = a
    if t is not None:
        default_test_cfg_name_list = t

    #from ibeis.experiments import experiment_helpers
    ibs = ibeis.opendb(defaultdb=defaultdb)
    acfg_name_list = ut.get_argval(('--aidcfg', '--acfg', '-a'), type_=list, default=default_acfgstr_name_list)
    test_cfg_name_list = ut.get_argval('-t', type_=list, default=default_test_cfg_name_list)
    test_result_list = experiment_harness.run_test_configurations2(
        ibs, acfg_name_list, test_cfg_name_list, qaid_override=qaid_override)
    test_result = experiment_storage.combine_test_results(ibs, test_result_list)
    return ibs, test_result
    #return ibs, test_result_list


def testdata_ibeis(default_qaids=[1], default_daids='all', defaultdb='testdb1', ibs=None, verbose=False, return_annot_info=False):
    r"""
    Args:
        default_qaids (list): (default = [1])
        default_daids (str): (default = 'all')
        defaultdb (str): (default = 'testdb1')
        ibs (IBEISController):  ibeis controller object(default = None)
        verbose (bool):  verbosity flag(default = False)
        return_annot_info (bool): (default = False)

    Returns:
        ibs, qaid_list, daid_list, annot_info:

    CommandLine:
        python -m ibeis.init.main_helpers --exec-testdata_ibeis --db NNP_Master3
        python -m ibeis.init.main_helpers --exec-testdata_ibeis --db PZ_MTEST --acfg default:aids=gt,shuffle,index=0:25 --verbose-testdata
        python -m ibeis.init.main_helpers --exec-testdata_ibeis --db PZ_MTEST --acfg default:aids=gt,index=0:25 --verbose-testdata
        python -m ibeis.init.main_helpers --exec-testdata_ibeis --db NNP_Master3 --verbose-testdata -a controlled
        python -m ibeis.init.main_helpers --exec-testdata_ibeis --db NNP_Master3 --verbose-testdata --aidcfg controlled
        python -m ibeis.init.main_helpers --exec-testdata_ibeis --db NNP_Master3 --verbose-testdata --aidcfg default:species=None

        python -m ibeis.init.main_helpers --exec-testdata_ibeis --db NNP_Master3 --acfg controlled --verbose-testdata
        python -m ibeis.init.main_helpers --exec-testdata_ibeis --db PZ_Master0 --acfg controlled --verbose-testdata
        python -m ibeis.init.main_helpers --exec-testdata_ibeis --db GZ_ALL --acfg controlled --verbose-testdata

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.init.main_helpers import *  # NOQA
        >>> import ibeis
        >>> from ibeis.experiments import annotation_configs
        >>> default_qaids = [1]
        >>> default_daids = 'all'
        >>> defaultdb = 'testdb1'
        >>> ibs = None
        >>> verbose = False
        >>> return_annot_info = True
        >>> ibs, qaid_list, daid_list, aidcfg = testdata_ibeis(default_qaids, default_daids, defaultdb, ibs, verbose, return_annot_info)
        >>> print('Printing annot config')
        >>> annotation_configs.print_acfg(aidcfg)
        >>> print('Printing annotconfig stats')
        >>> #print('qaid_list = %r' % (np.array(qaid_list),))
        >>> ibs.get_annotconfig_stats(qaid_list, daid_list)
        >>> print('Combined annotconfig stats')
        >>> ibs.print_annot_stats(qaid_list + daid_list, yawtext_isect=True)
    """
    print('[testdata_ibeis] Getting test annot configs')
    import ibeis
    if ibs is None:
        ibs = ibeis.opendb(defaultdb=defaultdb)
    # TODO: rectify command line with function arguments
    from ibeis.experiments import experiment_helpers
    aidcfg_name_list = ut.get_argval(('--aidcfg', '--acfg', '-a'), type_=list, default=['default'])
    acfg_list, expanded_aids_list = experiment_helpers.get_annotcfg_list(ibs, aidcfg_name_list)

    #aidcfg = old_main_helpers.get_commandline_aidcfg()
    assert len(acfg_list) == 1, 'multiple acfgs specified, but this function is built to return only 1. len(acfg_list)=%r' % (len(acfg_list),)
    aidcfg = acfg_list[0]

    qaid_list, daid_list = expanded_aids_list[0]

    #ibs.get_annotconfig_stats(qaid_list, daid_list)

    if ut.VERYVERBOSE:
        ibeis.other.dbinfo.print_qd_info(ibs, qaid_list, daid_list, verbose=True)
    if return_annot_info:
        return ibs, qaid_list, daid_list, aidcfg
    else:
        return ibs, qaid_list, daid_list


#def register_utool_aliases():
#    """
#    registers commmon class names with utool so they are printed nicely
#    """
#    #print('REGISTER UTOOL ALIASES')
#    import utool as ut
#    import matplotlib as mpl
#    from ibeis.control import IBEISControl, SQLDatabaseControl
#    from ibeis.gui import guiback
#    #from ibeis.gui import guifront
#    ut.extend_global_aliases([
#        (SQLDatabaseControl.SQLDatabaseController, 'sqldb'),
#        (IBEISControl.IBEISController, 'ibs'),
#        (guiback.MainWindowBackend, 'back'),
#        #(guifront.MainWindowFrontend, 'front'),
#        (mpl.figure.Figure, 'fig')
#    ])


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.init.main_helpers
        python -m ibeis.init.main_helpers --allexamples
        python -m ibeis.init.main_helpers --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
