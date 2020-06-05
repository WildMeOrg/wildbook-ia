# -*- coding: utf-8 -*-
"""
TODO:
    replace with dtool
    Rename to IdentifyRequest

    python -m utool.util_inspect check_module_usage --pat="query_request.py"
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from os.path import join
from wbia import dtool
import itertools as it
import hashlib
import vtool as vt
import utool as ut
import numpy as np
from wbia.algo.hots import neighbor_index_cache

# from wbia.algo.hots import multi_index
# from wbia.algo.hots import scorenorm
# from wbia.algo.hots import distinctiveness_normalizer
from wbia.algo.hots import query_params
from wbia.algo.hots import chip_match
from wbia.algo.hots import _pipeline_helpers as plh  # NOQA
import wbia.constants as const

# import warnings
(print, rrr, profile) = ut.inject2(__name__)

VERBOSE_QREQ, VERYVERBOSE_QREQ = ut.get_module_verbosity_flags('qreq')


def testdata_newqreq(defaultdb='testdb1'):
    """
    Returns:
        (wbia.IBEISController, list, list)
    """
    import wbia

    ibs = wbia.opendb(defaultdb=defaultdb)
    qaid_list = [1]
    daid_list = [1, 2, 3, 4, 5]
    return ibs, qaid_list, daid_list


@profile
def new_wbia_query_request(
    ibs,
    qaid_list,
    daid_list,
    cfgdict=None,
    verbose=None,
    unique_species=None,
    use_memcache=True,
    query_cfg=None,
    custom_nid_lookup=None,
):
    """
    wbia entry point to create a new query request object

    Args:
        ibs (wbia.IBEISController):  image analysis api
        qaid_list (list): query ids
        daid_list (list): database ids
        cfgdict (dict): pipeline dictionary config
        query_cfg (dtool.Config): Pipeline Config Object
        unique_species (None): (default = None)
        use_memcache (bool): (default = True)
        verbose (bool):  verbosity flag(default = True)

    Returns:
        wbia.QueryRequest

    CommandLine:
        python -m wbia.algo.hots.query_request --test-new_wbia_query_request:0
        python -m wbia.algo.hots.query_request --test-new_wbia_query_request:1

    Example0:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.hots.query_request import *  # NOQA
        >>> ibs, qaid_list, daid_list = testdata_newqreq('PZ_MTEST')
        >>> unique_species = None
        >>> verbose = ut.NOT_QUIET
        >>> cfgdict = {'sv_on': False, 'fg_on': True}  # 'fw_detector': 'rf'}
        >>> qreq_ = new_wbia_query_request(ibs, qaid_list, daid_list, cfgdict=cfgdict)
        >>> print(qreq_.get_cfgstr())
        >>> assert qreq_.qparams.sv_on is False, (
        ...     'qreq_.qparams.sv_on = %r ' % qreq_.qparams.sv_on)
        >>> result = ibs.get_dbname() + qreq_.get_data_hashid()
        >>> print(result)
        PZ_MTEST_DPCC_UUIDS-a5-n2-vpkyggtpzbqbecuq

    Example1:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.hots.query_request import *  # NOQA
        >>> ibs, qaid_list, daid_list = testdata_newqreq('NAUT_test')
        >>> unique_species = None
        >>> verbose = ut.NOT_QUIET
        >>> cfgdict = {'sv_on': True, 'fg_on': True}
        >>> qreq_ = new_wbia_query_request(ibs, qaid_list, daid_list, cfgdict=cfgdict)
        >>> assert qreq_.query_config2_.featweight_enabled is False
        >>> # Featweight should be off because there is no Naut detector
        >>> print(qreq_.qparams.query_cfgstr)
        >>> assert qreq_.qparams.sv_on is True, (
        ...     'qreq_.qparams.sv_on = %r ' % qreq_.qparams.sv_on)
        >>> result = ibs.get_dbname() + qreq_.get_data_hashid()
        >>> print(result)
        NAUT_test_DPCC_UUIDS-a5-n3-rtuyggvzpczvmjcw

    Example2:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.hots.query_request import *  # NOQA
        >>> ibs, qaid_list, daid_list = testdata_newqreq('PZ_MTEST')
        >>> unique_species = None
        >>> verbose = ut.NOT_QUIET
        >>> cfgdict = {'sv_on': False, 'query_rotation_heuristic': True}
        >>> qreq_ = new_wbia_query_request(ibs, qaid_list, daid_list, cfgdict=cfgdict)
        >>> # Featweight should be off because there is no Naut detector
        >>> print(qreq_.qparams.query_cfgstr)
        >>> assert qreq_.qparams.sv_on is False, (
        ...     'qreq_.qparams.sv_on = %r ' % qreq_.qparams.sv_on)
        >>> result = ibs.get_dbname() + qreq_.get_data_hashid()
        >>> print(result)
        PZ_MTEST_DPCC_UUIDS-a5-n2-vpkyggtpzbqbecuq

    Ignore:
        # This is supposed to be the beginings of the code to transition the
        # pipeline configuration into the new minimal dict based structure that
        # supports different configs for query and database annotations.
        dcfg = qreq_.extern_data_config2
        qcfg = qreq_.extern_query_config2
        ut.dict_intersection(qcfg.__dict__, dcfg.__dict__)
        from wbia.expt import cfghelpers
        cfg_list = [qcfg.__dict__, dcfg.__dict__]
        nonvaried_cfg, varied_cfg_list = ut.partition_varied_cfg_list(
            cfg_list, recursive=True)
        qvaried, dvaried = varied_cfg_list
    """
    if verbose is None:
        verbose = int(ut.NOT_QUIET)
        # verbose = VERBOSE_QREQ
    if verbose:
        print('[qreq] +--- New IBEIS QRequest --- ')

    if ut.SUPER_STRICT:
        ibs.assert_valid_aids(qaid_list, msg='error in new qreq qaids')
        ibs.assert_valid_aids(daid_list, msg='error in new qreq daids')

    qresdir = ibs.get_qres_cachedir()
    cfgdict = {} if cfgdict is None else cfgdict.copy()

    # try:
    piperoot = cfgdict.get('pipeline_root', cfgdict.get('proot', None))
    if piperoot is None and query_cfg is not None:
        try:
            piperoot = query_cfg.get('pipeline_root', query_cfg.get('proot', None))
        except AttributeError:
            pass
    # except Exception:
    #     piperoot = None

    if verbose > 2:
        print('[qreq] piperoot = %r' % (piperoot,))
    if piperoot is not None and piperoot in ['smk']:
        from wbia.algo.smk import smk_pipeline

        if query_cfg is None:
            config = cfgdict
        else:
            # Another config hack
            config = dict(query_cfg.parse_items())
            config.update(**cfgdict)

        assert custom_nid_lookup is None, 'unsupported'
        qreq_ = smk_pipeline.SMKRequest(ibs, qaid_list, daid_list, config)
    # HACK FOR DEPC REQUESTS including flukes
    elif query_cfg is not None and isinstance(query_cfg, dtool.Config):
        if verbose > 2:
            print('[qreq] dtool.Config HACK')
        tablename = query_cfg.get_config_name()
        cfgdict = dict(query_cfg.parse_items())
        requestclass = ibs.depc_annot.requestclass_dict[tablename]
        assert custom_nid_lookup is None, 'unsupported'
        qreq_ = request = requestclass.new(  # NOQA
            ibs.depc_annot, qaid_list, daid_list, cfgdict, tablename=tablename
        )
    elif piperoot is not None and piperoot not in ['vsmany']:
        # Hack to ensure that correct depcache style request gets called
        if verbose > 2:
            print('[qreq] piperoot HACK')
        requestclass = ibs.depc_annot.requestclass_dict[piperoot]
        assert custom_nid_lookup is None, 'unsupported'
        qreq_ = request = requestclass.new(  # NOQA
            ibs.depc_annot, qaid_list, daid_list, cfgdict, tablename=piperoot
        )
        # assert qreq_.qparams.pipeline_root != 'vsone', 'pipeline vsone is depricated'
    else:
        if verbose > 2:
            print('[qreq] default hots config HACK')

        # <HACK>
        if not hasattr(ibs, 'generate_species_background_mask'):
            print('HACKING FG OFF')
            cfgdict['fg_on'] = False

        if unique_species is None:
            unique_species_ = apply_species_with_detector_hack(
                ibs, cfgdict, qaid_list, daid_list
            )
        else:
            unique_species_ = unique_species
        # </HACK>
        if query_cfg is None:
            cfg = ibs.cfg.query_cfg
        else:
            cfg = query_cfg

        qparams = query_params.QueryParams(cfg, cfgdict)
        data_config2_ = qparams
        #
        # <HACK>
        # MAKE A SECOND CONFIG FOR QUERIES AND DATABASE VECTORS ONLY
        # allow query and database annotations to have different feature configs
        if qparams.query_rotation_heuristic:
            query_cfgdict = cfgdict.copy()
            query_cfgdict['augment_orientation'] = True
            query_config2_ = query_params.QueryParams(cfg, query_cfgdict)
        else:
            query_config2_ = qparams
        # </HACK>
        _indexer_request_params = dict(use_memcache=use_memcache)
        qreq_ = QueryRequest.new_query_request(
            qaid_list,
            daid_list,
            qparams,
            qresdir,
            ibs,
            query_config2_,
            data_config2_,
            _indexer_request_params,
            custom_nid_lookup=custom_nid_lookup,
        )
        # qreq_.query_config2_ = query_config2_
        # qreq_.data_config2_ = data_config2_
        qreq_.unique_species = unique_species_  # HACK
        if verbose > 1:
            print('[qreq] * unique_species = %s' % (qreq_.unique_species,))
    if verbose:
        print('[qreq] * pipe_cfg = %s' % (qreq_.get_pipe_cfgstr()))
        print('[qreq] * data_hashid  = %s' % (qreq_.get_data_hashid(),))
        print('[qreq] * query_hashid = %s' % (qreq_.get_query_hashid(),))
        print('[qreq] L___ New IBEIS QRequest ___ ')
    return qreq_


@profile
def apply_species_with_detector_hack(ibs, cfgdict, qaids, daids, verbose=None):
    """
    HACK turns of featweights if they cannot be applied
    """
    if verbose is None:
        verbose = VERBOSE_QREQ
    if True:
        # Hack for test speed
        if ibs.dbname in {'PZ_MTEST', 'GZ_Master1', 'PZ_Master1'}:
            return True
    # Only apply the hack with repsect to the queried annotations
    aid_list = set(it.chain(qaids, daids))
    unique_species = ibs.get_database_species(aid_list)
    # turn off featureweights when not absolutely sure they are ok to us,)
    # candetect = (len(unique_species) == 1 and
    #              ibs.has_species_detector(unique_species[0]))

    ut.cprint('unique_species = %r' % (unique_species,), 'yellow')
    candetect = True
    for species in set(unique_species):
        if species == const.UNKNOWN:
            continue
        if not ibs.has_species_detector(species):
            ut.cprint(
                'Species %r is not supported with ibs.has_species_detector()'
                % (species,),
                'yellow',
            )
            candetect = False
            break

    if not candetect:
        if ut.NOT_QUIET:
            ut.cprint(
                '[qreq] HACKING FG_WEIGHT OFF (db species is not supported)', 'yellow'
            )
            if verbose > 1:
                if len(unique_species) != 1:
                    print('[qreq]  * len(unique_species) = %r' % len(unique_species))
                else:
                    print('[qreq]  * unique_species = %r' % (unique_species,))
        # print('[qreq]  * valid species = %r' % (
        #    ibs.get_species_with_detectors(),))
        # cfg._featweight_cfg.featweight_enabled = 'ERR'
        cfgdict['featweight_enabled'] = False  # 'ERR'
        cfgdict['fg_on'] = False
    else:
        # print(ibs.get_annot_species_texts(aid_list))
        if verbose:
            print('[qreq] Using fgweights of unique_species=%r' % (unique_species,))
    return unique_species


@ut.reloadable_class
class QueryRequest(ut.NiceRepr):
    """
    Request object for pipline parameter run
    """

    _isnewreq = False

    def __init__(qreq_):
        # Conceptually immutable State
        qreq_.unique_species = None  # num categories
        qreq_.internal_qspeciesid_list = None  # category species id label list
        qreq_.internal_qaids = None
        qreq_.internal_daids = None
        # Conceptually mutable state
        qreq_.internal_qaids_mask = None
        qreq_.internal_daids_mask = None
        # Loaded Objects
        # Handle to parent IBEIS Controller

        # HACK: jedi type hinting. Need to have non-obvious condition
        try:
            qreq_.ibs = None
        except Exception:
            import wbia

            qreq_.ibs = wbia.IBEISController()

        qreq_.indexer = None  # The nearest neighbor mechanism
        qreq_.normalizer = None  # The scoring normalization mechanism
        qreq_.dstcnvs_normer = None
        qreq_.hasloaded = False
        # Pipeline configuration
        qreq_.qparams = None  # Parameters relating to pipeline execution
        qreq_.query_config2_ = None
        qreq_.data_config2_ = None
        qreq_._indexer_request_params = None
        # Set values
        qreq_.unique_species = None  # HACK
        qreq_.qresdir = None
        qreq_.prog_hook = None
        qreq_.lnbnn_normer = None

        # Keeps internal name state
        qreq_.unique_aids = None
        qreq_.unique_nids = None
        qreq_.aid_to_idx = None
        qreq_.nid_to_groupuuid = None

    @classmethod
    @profile
    def new_query_request(
        cls,
        qaid_list,
        daid_list,
        qparams,
        qresdir,
        ibs,
        query_config2_,
        data_config2_,
        _indexer_request_params,
        custom_nid_lookup=None,
    ):
        """
        old way of calling new

        Args:
            qaid_list (list):
            daid_list (list):
            qparams (QueryParams):  query hyper-parameters
            qresdir (str):
            ibs (wbia.IBEISController):  image analysis api
            _indexer_request_params (dict):

        Returns:
            wbia.QueryRequest
        """
        qreq_ = cls()
        qreq_.ibs = ibs
        qreq_.qparams = qparams  # Parameters relating to pipeline execution
        qreq_.query_config2_ = query_config2_
        qreq_.data_config2_ = data_config2_
        qreq_.qresdir = qresdir
        qreq_._indexer_request_params = _indexer_request_params
        qreq_.set_external_daids(daid_list)
        qreq_.set_external_qaids(qaid_list)

        # Load name information so it can change in the database and that's ok.
        # I'm not 100% liking how this works.
        qreq_.unique_aids = np.union1d(qreq_.qaids, qreq_.daids)
        qreq_.unique_aids.sort()

        # Internal caching objects and views
        _annots = ibs.annots(qreq_.unique_aids)
        # I think the views copy the original cache
        qreq_._unique_annots = _annots.view(_annots.aids)
        qreq_._unique_dannots = qreq_._unique_annots.view(sorted(qreq_.daids))

        qreq_.aid_to_idx = ut.make_index_lookup(qreq_.unique_aids)
        if custom_nid_lookup is None:
            qreq_.unique_nids = ibs.get_annot_nids(qreq_.unique_aids)
        else:
            qreq_.unique_nids = ut.dict_take(custom_nid_lookup, qreq_.unique_aids)
        qreq_.unique_nids = np.array(qreq_.unique_nids)

        # qreq_.nid_to_groupuuid = qreq_._make_namegroup_uuids()
        # qreq_.dnid_to_groupuuid = qreq_._make_namegroup_data_uuids()
        qreq_.nid_to_grouphash = qreq_._make_namegroup_hashes()
        qreq_.dnid_to_grouphash = qreq_._make_namegroup_data_hashes()
        return qreq_

    @profile
    def _make_namegroup_data_hashes(qreq_):
        """
        Makes dynamically created uuid groups only for database annotations
        (hacks for iccv).
        """
        annots = qreq_._unique_dannots
        nids = np.array(qreq_.get_qreq_annot_nids(annots._rowids))
        nid_to_grouphash = qreq_._make_anygroup_hashes(annots, nids)
        return nid_to_grouphash

    @profile
    def _make_namegroup_hashes(qreq_):
        """
        dynamically creates uuid groups
        """
        # annots = qreq_.ibs.annots(qreq_.unique_aids)
        annots = qreq_._unique_annots
        nids = qreq_.unique_nids
        nid_to_grouphash = qreq_._make_anygroup_hashes(annots, nids)
        return nid_to_grouphash

    @staticmethod
    def _make_anygroup_hashes(annots, nids):
        """ helper function

            import wbia
            qreq_ = wbia.testdata_qreq_(
                defaultdb='PZ_MTEST',
                qaid_override=[1, 2, 3, 4, 5, 6, 10, 11],
                daid_override=[2, 3, 5, 6, 20, 21, 22, 23, 24],
                )

            import wbia
            qreq_ = wbia.testdata_qreq_(defaultdb='PZ_Master1')
            %timeit qreq_._make_namegroup_data_hashes()
            %timeit qreq_._make_namegroup_data_uuids()

        """
        # make sure items are sorted to ensure same assignment
        # gives same uuids
        # annots = qreq_.ibs.annots(sorted(qreq_.daids))
        unique_nids, groupxs = vt.group_indices(nids)
        grouped_visual_uuids = ut.apply_grouping(annots.visual_uuids, groupxs)
        group_hashes = [
            ut.combine_hashes(sorted(u.bytes for u in uuids), hasher=hashlib.sha1())
            for uuids in grouped_visual_uuids
        ]
        nid_to_grouphash = dict(zip(unique_nids, group_hashes))
        return nid_to_grouphash

    def get_qreq_annot_visual_uuids(qreq_, aids):
        visual_uuids = qreq_._unique_annots.view(aids).visual_uuids
        return visual_uuids

    @profile
    def get_qreq_pcc_uuids(qreq_, aids):
        """
        TODO. dont use uuids anymore. they are slow
        """
        import uuid

        for bytes_ in qreq_.get_qreq_pcc_hashes(aids):
            yield uuid.UUID(bytes=bytes_[0:16])

    @profile
    def get_qreq_pcc_hashes(qreq_, aids):
        """
        aids = [1, 2, 3]
        """
        nids = qreq_.get_qreq_annot_nids(aids)
        b = ut.util_hash.b
        zero = b('\x00' * 16)
        # Should we just be combining with a hash that represents the entire
        # database PCC state? Maybe.
        # For now, only considers grouping of database names
        dannot_name_hashes = ut.dict_take(qreq_.dnid_to_grouphash, nids, zero)
        dannot_visual_uuids = qreq_.get_qreq_annot_visual_uuids(aids)
        dannot_visual_hashes = (u.bytes for u in dannot_visual_uuids)
        for vuuid, nuuid in zip(dannot_visual_hashes, dannot_name_hashes):
            bytes_ = ut.combine_hashes((vuuid, nuuid), hasher=hashlib.sha1())
            yield bytes_

    @profile
    def get_qreq_pcc_hashid(qreq_, aids, prefix='', with_nids=False):
        """
        Gets a combined hash of a group of aids. Each aid hash represents
        itself in the context of the query database.

        only considers grouping of database names

        CommandLine:
            python -m wbia.algo.hots.query_request --test-get_qreq_pcc_hashid:0

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.algo.hots.query_request import *  # NOQA
            >>> import wbia
            >>> p = ['default:K=2,nameknn=True']
            >>> defaultdb = 'testdb1'
            >>> # Test that UUIDS change when you change the name lookup
            >>> new_ = ut.partial(wbia.testdata_qreq_, defaultdb=defaultdb, p=p,
            >>>                   verbose=False)
            >>> # All diff names
            >>> qreq1 = new_(daid_override=[2, 3, 5, 6],
            >>>              qaid_override=[1, 2, 4],
            >>>              custom_nid_lookup={a: a for a in range(14)})
            >>> # All same names
            >>> qreq2 = new_(daid_override=[2, 3, 5, 6],
            >>>              qaid_override=[1, 2, 4],
            >>>              custom_nid_lookup={a: 1 for a in range(14)})
            >>> # Change the PCC, removing a query (data should NOT change)
            >>> # because the thing being queried against is the same
            >>> qreq3 = new_(daid_override=[2, 3, 5, 6],
            >>>              qaid_override=[1, 2],
            >>>              custom_nid_lookup={a: 1 for a in range(14)})
            >>> # Now remove a database object (query SHOULD change)
            >>> # because the results are different depending on
            >>> # nameing of database (maybe they shouldnt change...)
            >>> qreq4 = new_(daid_override=[2, 3, 6],
            >>>              qaid_override=[1, 2, 4],
            >>>              custom_nid_lookup={a: 1 for a in range(14)})
            >>> print(qreq1.get_cfgstr(with_input=True, with_pipe=False))
            >>> print(qreq2.get_cfgstr(with_input=True, with_pipe=False))
            >>> print(qreq3.get_cfgstr(with_input=True, with_pipe=False))
            >>> print(qreq4.get_cfgstr(with_input=True, with_pipe=False))
            >>> assert qreq3.get_data_hashid() == qreq2.get_data_hashid()
            >>> assert qreq1.get_data_hashid() != qreq2.get_data_hashid()

        """
        # TODO: pcc based hashing should only be used if name dependant
        # attributes are used in the pipeline.
        label = ''.join(('_', prefix, 'PCC_UUIDS'))
        pcc_hashes = qreq_.get_qreq_pcc_hashes(sorted(aids))
        pcc_hash = ut.combine_hashes(pcc_hashes, hasher=hashlib.sha1())
        pcc_hashstr = ut.convert_bytes_to_bigbase(pcc_hash)
        pcc_hashstr = pcc_hashstr[0:16]

        sep = '-'

        n_aids = 'a' + str(len(aids))

        if with_nids:
            unique_nids = set(qreq_.get_qreq_annot_nids(aids))
            n_nids = 'n' + str(len(unique_nids))
            pcc_hashid = sep.join([label, n_aids, n_nids, pcc_hashstr])
        else:
            pcc_hashid = sep.join([label, n_aids, pcc_hashstr])
        return pcc_hashid

    def __getstate__(qreq_):
        """
        Make QueryRequest pickleable

        CommandLine:
            python -m wbia.dev -t candidacy --db testdb1

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.algo.hots.query_request import *  # NOQA
            >>> import wbia
            >>> from six.moves import cPickle as pickle
            >>> qreq_ = wbia.testdata_qreq_()
            >>> qreq_dump = pickle.dumps(qreq_)
            >>> qreq2_ = pickle.loads(qreq_dump)
        """
        state = qreq_.__dict__.copy()
        # Unload attributes that should not be saved
        # state['ibs'] = None
        state['prog_hook'] = None
        state['indexer'] = None
        state['normalizer'] = None
        state['dstcnvs_normer'] = None
        state['hasloaded'] = False
        state['lnbnn_normer'] = False
        state['_internal_dannots'] = None
        state['_internal_qannots'] = None
        state['_unique_annots'] = None
        state['_unique_dannots'] = None
        # qreq_._unique_annots = _annots.view(_annots.aids)
        # qreq_._unique_dannots = qreq_._unique_annots.view(sorted(qreq_.daids))

        # Hack for the actual wbia object
        # (The ibs object itself should now do this hack)
        # state['dbdir'] = qreq_.ibs.get_dbdir()
        return state

    def __setstate__(qreq_, state):
        # Hack for the actual wbia object
        # (The ibs object itself should now do this hack)
        # import wbia
        # dbdir = state['dbdir']
        # del state['dbdir']
        # state['ibs'] = wbia.opendb(dbdir=dbdir, web=False)
        qreq_.__dict__.update(state)

        # Internal caching objects and views
        _annots = qreq_.ibs.annots(qreq_.unique_aids)
        # I think the views copy the original cache
        qreq_._unique_annots = _annots.view(_annots.aids)
        qreq_._unique_dannots = qreq_._unique_annots.view(sorted(qreq_.daids))

    def _custom_str(qreq_):
        r"""
        CommandLine:
            python -m wbia.algo.hots.query_request --exec-_custom_str --show

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.algo.hots.query_request import *  # NOQA
            >>> import wbia
            >>> qreq_ = wbia.testdata_qreq_()
            >>> print(repr(qreq_))
        """
        typestr = qreq_.__class__.__name__
        parts = qreq_.get_shortinfo_parts()
        # print('parts = %r' % (parts,))
        custom_str = '%s(%s) %s %s %s' % ((typestr,) + tuple(parts))
        return custom_str

    def get_shortinfo_parts(qreq_):
        """ Rename to get_nice_parts """
        parts = []
        parts.append(qreq_.ibs.get_dbname())
        parts.append('nQ=%d' % len(qreq_.qaids))
        parts.append('nD=%d' % len(qreq_.daids))
        parts.append(qreq_.get_pipe_hashid())
        return parts

    def get_shortinfo_cfgstr(qreq_):
        shortinfo_cfgstr = '_'.join(qreq_.get_shortinfo_parts())
        return shortinfo_cfgstr

    def get_big_cacher(qreq_):
        bc_dpath, bc_fname, bc_cfgstr = qreq_.get_bigcache_info()
        cacher = ut.Cacher(bc_fname, bc_cfgstr, cache_dir=bc_dpath)
        return cacher

    @profile
    def get_bigcache_info(qreq_):
        bc_dpath = qreq_.ibs.get_big_cachedir()
        bc_fname = 'BIG_MC4_' + qreq_.get_shortinfo_cfgstr()
        bc_cfgstr = qreq_.get_full_cfgstr()
        bc_info = bc_dpath, bc_fname, bc_cfgstr
        return bc_info

    def get_full_cfgstr(qreq_):
        """ main cfgstring used to identify the 'querytype'
        FIXME: name
        params + data + query
        """
        full_cfgstr = qreq_.get_cfgstr(with_input=True)
        return full_cfgstr

    def __nice__(qreq_):
        parts = qreq_.get_shortinfo_parts()
        return ' '.join(parts)

    # def __repr__(qreq_):
    #     return '<' + qreq_._custom_str() + ' at %s>' % (hex(id(qreq_)),)

    # def __str__(qreq_):
    #     return '<' + qreq_._custom_str() + '>'

    def set_external_daids(qreq_, daid_list):
        qreq_._set_internal_daids(daid_list)

    def set_external_qaids(qreq_, qaid_list):
        qreq_._set_internal_qaids(qaid_list)

    def _set_internal_daids(qreq_, daid_list):
        qreq_.internal_daids_mask = None  # Invalidate mask
        qreq_.internal_daids = np.array(daid_list)
        # Use new annotation objects
        config = qreq_.get_internal_data_config2()
        qreq_._internal_dannots = qreq_.ibs.annots(qreq_.internal_daids, config=config)

    def _set_internal_qaids(qreq_, qaid_list):
        qreq_.internal_qaids_mask = None  # Invalidate mask
        qreq_.internal_qaids = np.array(qaid_list)
        # Use new annotation objects
        config = qreq_.get_internal_query_config2()
        qreq_._internal_qannots = qreq_.ibs.annots(qreq_.internal_qaids, config=config)

    def shallowcopy(qreq_, qaids=None):
        """
        Creates a copy of qreq with the same qparams object and a subset of the
        qx and dx objects.  used to generate chunks of vsmany queries

        CommandLine:
            python -m wbia.algo.hots.query_request QueryRequest.shallowcopy

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.algo.hots.query_request import *  # NOQA
            >>> import wbia
            >>> qreq_ = wbia.testdata_qreq_(default_qaids=[1, 2])
            >>> qreq2_ = qreq_.shallowcopy(qaids=1)
            >>> assert qreq_.daids is qreq2_.daids, 'should be the same'
            >>> assert len(qreq_.qaids) != len(qreq2_.qaids), 'should be diff'
            >>> #assert qreq_.metadata is not qreq2_.metadata
        """
        # qreq2_ = copy.copy(qreq_)  # copy calls setstate and getstate
        qreq2_ = QueryRequest()
        qreq2_.__dict__.update(qreq_.__dict__)
        qaids = [qaids] if not ut.isiterable(qaids) else qaids
        _intersect = np.intersect1d(qaids, qreq2_.qaids)
        assert len(_intersect) == len(qaids), 'not a subset'
        qreq2_.set_external_qaids(qaids)
        # The shallow copy does not bring over output / query data
        qreq2_.indexer = None
        # qreq2_.metadata = {}
        qreq2_.hasloaded = False
        return qreq2_

    # --- State Modification ---

    # def remove_internal_daids(qreq_, remove_daids):
    #    r"""
    #    DEPRICATE

    #    State Modification: remove daids from the query request.  Do not call
    #    this function often. It invalidates the indexer, which is very slow to
    #    rebuild.  Should only be done between query pipeline runs.

    #    CommandLine:
    #        python -m wbia.algo.hots.query_request --test-remove_internal_daids

    #    Example:
    #        >>> # ENABLE_DOCTEST
    #        >>> from wbia.algo.hots.query_request import *  # NOQA
    #        >>> import wbia
    #        >>> # build test data
    #        >>> ibs = wbia.opendb('testdb1')
    #        >>> species = wbia.const.TEST_SPECIES.ZEB_PLAIN
    #        >>> daids = ibs.get_valid_aids(species=species, is_exemplar=True)
    #        >>> qaids = ibs.get_valid_aids(species=species, is_exemplar=False)
    #        >>> qreq_ = ibs.new_query_request(qaids, daids)
    #        >>> remove_daids = daids[0:1]
    #        >>> # execute function
    #        >>> assert len(qreq_.internal_daids) == 4, 'bad setup data'
    #        >>> qreq_.remove_internal_daids(remove_daids)
    #        >>> # verify results
    #        >>> assert len(qreq_.internal_daids) == 3, 'did not remove'
    #    """
    #    # Invalidate the current indexer, mask and metadata
    #    qreq_.indexer = None
    #    qreq_.internal_daids_mask = None
    #    #qreq_.metadata = {}
    #    # Find indices to remove
    #    delete_flags = vt.get_covered_mask(qreq_.internal_daids, remove_daids)
    #    delete_indices = np.where(delete_flags)[0]
    #    assert len(delete_indices) == len(remove_daids), (
    #        'requested removal of nonexistant daids')
    #    # Remove indices
    #    qreq_.internal_daids = np.delete(qreq_.internal_daids, delete_indices)
    #    # TODO: multi-indexer delete support
    #    if qreq_.indexer is not None:
    #        warnings.warn('Implement point removal from trees')
    #        qreq_.indexer.remove_wbia_support(qreq_, remove_daids)

    # def add_internal_daids(qreq_, new_daids):
    #    """
    #    DEPRICATE

    #    State Modification: add new daid to query request. Should only be
    #    done between query pipeline runs
    #    """
    #    if ut.DEBUG2:
    #        species = qreq_.ibs.get_annot_species(new_daids)
    #        assert set(qreq_.unique_species) == set(species), (
    #            'inconsistent species')
    #    qreq_.internal_daids_mask = None
    #    #qreq_.metadata = {}
    #    qreq_.internal_daids = np.append(qreq_.internal_daids, new_daids)
    #    # TODO: multi-indexer add support
    #    if qreq_.indexer is not None:
    #        #qreq_.load_indexer(verbose=True)
    #        qreq_.indexer.add_wbia_support(qreq_, new_daids)

    def set_external_qaid_mask(qreq_, masked_qaid_list):
        r"""
        Args:
            qaid_list (list):

        CommandLine:
            python -m wbia.algo.hots.query_request --test-set_external_qaid_mask

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.algo.hots.query_request import *  # NOQA
            >>> import wbia
            >>> ibs = wbia.opendb(db='testdb1')
            >>> qaid_list = [1, 2, 3, 4, 5]
            >>> daid_list = [1, 2, 3, 4, 5]
            >>> qreq_ = ibs.new_query_request(qaid_list, daid_list)
            >>> masked_qaid_list = [2, 4, 5]
            >>> qreq_.set_external_qaid_mask(masked_qaid_list)
            >>> result = np.array_str(qreq_.qaids)
            >>> print(result)
            [1 3]
        """
        qreq_.set_internal_masked_qaids(masked_qaid_list)

    # --- Internal Annotation ID Masks ----

    def set_internal_masked_daids(qreq_, masked_daid_list):
        """ used by the pipeline to execute a subset of the query request
        without modifying important state """
        if masked_daid_list is None or len(masked_daid_list) == 0:
            qreq_.internal_daids_mask = None
        else:
            # with ut.EmbedOnException():
            # input denotes invalid elements mark all elements not in that
            # list as True
            flags = vt.get_uncovered_mask(qreq_.internal_daids, masked_daid_list)
            assert len(flags) == len(qreq_.internal_daids), 'unequal len internal daids'
            qreq_.internal_daids_mask = flags

    def set_internal_masked_qaids(qreq_, masked_qaid_list):
        r"""
        used by the pipeline to execute a subset of the query request
        without modifying important state

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.algo.hots.query_request import *  # NOQA
            >>> import utool as ut
            >>> import wbia
            >>> qaid_list = [1, 2, 3, 4]
            >>> daid_list = [1, 2, 3, 4]
            >>> qreq_ = wbia.testdata_qreq_(qaid_override=qaid_list, daid_override=daid_list, p='default:sv_on=True')
            >>> qaids = qreq_.get_internal_qaids()
            >>> ut.assert_lists_eq(qaid_list, qaids)
            >>> masked_qaid_list = [1, 2, 3,]
            >>> qreq_.set_internal_masked_qaids(masked_qaid_list)
            >>> new_internal_aids = qreq_.get_internal_qaids()
            >>> ut.assert_lists_eq(new_internal_aids, [4])
        """
        if masked_qaid_list is None or len(masked_qaid_list) == 0:
            qreq_.internal_qaids_mask = None
        else:
            # input denotes invalid elements mark all elements not in that
            # list as True
            flags = vt.get_uncovered_mask(qreq_.internal_qaids, masked_qaid_list)
            assert len(flags) == len(qreq_.internal_qaids), 'unequal len internal qaids'
            qreq_.internal_qaids_mask = flags

    # --- INTERNAL INTERFACE ---
    # For within pipeline use only

    @property
    def internal_qannots(qreq_):
        if qreq_.internal_qaids_mask is None:
            return qreq_._internal_qannots
        else:
            return qreq_._internal_qannots.compress(qreq_.internal_qaids_mask)

    @property
    def internal_dannots(qreq_):
        if qreq_.internal_daids_mask is None:
            return qreq_._internal_dannots
        else:
            return qreq_._internal_dannots.compress(qreq_.internal_daids_mask)

    def get_internal_daids(qreq_):
        # return np.array(qreq_.internal_dannots.aid)
        if qreq_.internal_daids_mask is None:
            return qreq_.internal_daids
        else:
            return qreq_.internal_daids.compress(qreq_.internal_daids_mask, axis=0)

    def get_internal_qaids(qreq_):
        # return np.array(qreq_.internal_qannots.aid)
        if qreq_.internal_qaids_mask is None:
            return qreq_.internal_qaids
        else:
            return qreq_.internal_qaids.compress(qreq_.internal_qaids_mask, axis=0)

    def get_internal_data_config2(qreq_):
        return qreq_.data_config2_

    def get_internal_query_config2(qreq_):
        return qreq_.query_config2_

    # --- EXTERNAL INTERFACE ---

    def get_unique_species(qreq_):
        return qreq_.unique_species

    # External id-lists

    @property
    def qannots(qreq_):
        """ internal query annotation objects """
        return qreq_.internal_qannots

    @property
    def dannots(qreq_):
        """ external query annotation objects """
        return qreq_._internal_dannots

    @property
    def daids(qreq_):
        """ These are the users daids in vsone mode """
        return qreq_.get_internal_daids()

    @property
    def qaids(qreq_):
        """ These are the users qaids in vsone mode """
        return qreq_.get_internal_qaids()

    @ut.accepts_numpy
    def get_qreq_annot_nids(qreq_, aids):
        # Hack uses own internal state to grab name rowids
        # instead of using wbia.
        idxs = ut.take(qreq_.aid_to_idx, aids)
        nids = ut.take(qreq_.unique_nids, idxs)
        return nids

    def get_qreq_qannot_kpts(qreq_, qaids):
        return qreq_.ibs.get_annot_kpts(qaids, config2_=qreq_.extern_query_config2)

    def get_qreq_dannot_kpts(qreq_, daids):
        return qreq_.ibs.get_annot_kpts(daids, config2_=qreq_.extern_data_config2)

    def get_qreq_dannot_fgweights(qreq_, daids):
        return qreq_.ibs.get_annot_fgweights(
            daids, config2_=qreq_.extern_data_config2, ensure=False
        )

    def get_qreq_qannot_fgweights(qreq_, qaids):
        return qreq_.ibs.get_annot_fgweights(
            qaids, config2_=qreq_.extern_query_config2, ensure=False
        )

    @property
    def dnids(qreq_):
        """ TODO: save dnids in qreq_ state """
        return qreq_.get_qreq_annot_nids(qreq_.daids)

    @property
    def qnids(qreq_):
        """ TODO: save qnids in qreq_ state """
        return qreq_.get_qreq_annot_nids(qreq_.qaids)

    @property
    def extern_data_config2(qreq_):
        return qreq_.data_config2_

    @property
    def extern_query_config2(qreq_):
        return qreq_.query_config2_

    def get_external_query_groundtruth(qreq_, qaids):
        """ gets groundtruth that are accessible via this query """
        external_daids = qreq_.daids
        gt_aids = qreq_.ibs.get_annot_groundtruth(qaids, daid_list=external_daids)
        return gt_aids

    # External id-hashes

    def get_data_hashid(qreq_):
        r"""
        CommandLine:
            python -m wbia.algo.hots.query_request --exec-QueryRequest.get_query_hashid --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.algo.hots.query_request import *  # NOQA
            >>> import wbia
            >>> qreq_ = wbia.testdata_qreq_()
            >>> data_hashid = qreq_.get_data_hashid()
            >>> result = ('data_hashid = %s' % (ut.repr2(data_hashid),))
            >>> print(result)
        """
        data_hashid = qreq_.get_qreq_pcc_hashid(qreq_.daids, prefix='D', with_nids=True)
        return data_hashid

    def get_query_hashid(qreq_):
        r"""
        CommandLine:
            python -m wbia.algo.hots.query_request --exec-QueryRequest.get_query_hashid --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.algo.hots.query_request import *  # NOQA
            >>> import wbia
            >>> qreq_ = wbia.testdata_qreq_()
            >>> query_hashid = qreq_.get_query_hashid()
            >>> result = ('query_hashid = %s' % (ut.repr2(query_hashid),))
            >>> print(result)
        """
        query_hashid = qreq_.get_qreq_pcc_hashid(qreq_.qaids, prefix='Q')
        return query_hashid

    def get_pipe_cfgstr(qreq_):
        """
        FIXME: name
        params only """
        pipe_cfgstr = qreq_.qparams.query_cfgstr
        return pipe_cfgstr

    def get_pipe_hashid(qreq_):
        pipe_hashstr = ut.hashstr27(qreq_.get_pipe_cfgstr())
        return pipe_hashstr

    @profile
    def get_cfgstr(
        qreq_, with_input=False, with_data=True, with_pipe=True, hash_pipe=False
    ):
        r"""
        main cfgstring used to identify the 'querytype'
        FIXME: name params + data

        TODO:
            rename query_cfgstr to pipe_cfgstr or pipeline_cfgstr EVERYWHERE

        Args:
            with_input (bool): (default = False)

        Returns:
            str: cfgstr

        CommandLine:
            python -m wbia.algo.hots.query_request --exec-get_cfgstr

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.algo.hots.query_request import *  # NOQA
            >>> import wbia
            >>> qreq_ = wbia.testdata_qreq_(defaultdb='testdb1',
            >>>                              p='default:fgw_thresh=.3',
            >>>                              a='default:species=zebra_plains')
            >>> with_input = True
            >>> cfgstr = qreq_.get_cfgstr(with_input)
            >>> result = ('cfgstr = %s' % (str(cfgstr),))
            >>> print(result)
        """
        cfgstr_list = []
        if with_input:
            cfgstr_list.append(qreq_.get_query_hashid())
        if with_data:
            cfgstr_list.append(qreq_.get_data_hashid())
        if with_pipe:
            if hash_pipe:
                cfgstr_list.append(qreq_.get_pipe_hashid())
            else:
                cfgstr_list.append(qreq_.get_pipe_cfgstr())
        cfgstr = ''.join(cfgstr_list)
        return cfgstr

    def get_qresdir(qreq_):
        return qreq_.qresdir

    # --- Lazy Loading ---

    @profile
    def lazy_preload(qreq_, prog_hook=None, verbose=ut.NOT_QUIET):
        """
        feature weights and normalizers should be loaded before vsone queries
        are issued. They do not depened only on qparams

        Load non-query specific normalizers / weights
        """
        if verbose >= 2:
            print('[qreq] lazy preloading')
        if prog_hook is not None:
            prog_hook.initialize_subhooks(4)

        qreq_.qannots.preload('nids')
        qreq_.dannots.preload('nids')

        subhook = None if prog_hook is None else prog_hook.next_subhook()
        qreq_.ensure_features(verbose=verbose, prog_hook=subhook)

        subhook = None if prog_hook is None else prog_hook.next_subhook()
        if subhook is not None:
            subhook(0, 1, 'preload featweights')
        if qreq_.qparams.fg_on is True:
            qreq_.ensure_featweights(verbose=verbose)

        subhook = None if prog_hook is None else prog_hook.next_subhook()
        if subhook is not None:
            subhook(0, 1, 'finishing preload')

        subhook = None if prog_hook is None else prog_hook.next_subhook()
        if subhook is not None:
            subhook(0, 1, 'finished preload')
        # if hook is not None:
        #    hook.set_progress(4, 4, lbl='preloading features')

    @profile
    def lazy_load(qreq_, verbose=ut.NOT_QUIET):
        """
        Performs preloading of all data needed for a batch of queries
        """
        print('[qreq] lazy loading')
        qreq_.hasloaded = True
        qreq_.lazy_preload(verbose=verbose)
        if qreq_.qparams.pipeline_root in ['vsmany']:
            qreq_.load_indexer(verbose=verbose)

    # load query data structures
    @profile
    def ensure_chips(qreq_, verbose=ut.NOT_QUIET, num_retries=1):
        r"""
        ensure chips are computed (used in expt, not used in pipeline)

        Args:
            verbose (bool):  verbosity flag(default = True)
            num_retries (int): (default = 0)

        CommandLine:
            python -m wbia.algo.hots.query_request --test-ensure_chips

        Example:
            >>> # ENABLE_DOCTEST
            >>> # Delete chips (accidentally), then try to run a query
            >>> from wbia.algo.hots.query_request import *  # NOQA
            >>> import wbia
            >>> ibs = wbia.opendb(defaultdb='testdb1')
            >>> daids = ibs.get_valid_aids()[0:3]
            >>> qaids = ibs.get_valid_aids()[0:6]
            >>> qreq_ = ibs.new_query_request(qaids, daids)
            >>> verbose = True
            >>> num_retries = 1
            >>> qchip_fpaths = ibs.get_annot_chip_fpath(qaids, config2_=qreq_.extern_query_config2)
            >>> dchip_fpaths = ibs.get_annot_chip_fpath(daids, config2_=qreq_.extern_data_config2)
            >>> ut.remove_file_list(qchip_fpaths)
            >>> ut.remove_file_list(dchip_fpaths)
            >>> result = qreq_.ensure_chips(verbose, num_retries)
            >>> print(result)
        """
        if verbose:
            print('[qreq] ensure_chips')
        external_qaids = qreq_.qaids
        external_daids = qreq_.daids
        # np.union1d(external_qaids, external_daids)
        # TODO check if configs are the same
        externgetkw = dict(
            ensure=True, check_external_storage=True, num_retries=num_retries
        )
        q_chip_fpath = qreq_.ibs.get_annot_chip_fpath(  # NOQA
            external_qaids, config2_=qreq_.extern_query_config2, **externgetkw
        )
        d_chip_fpath = qreq_.ibs.get_annot_chip_fpath(  # NOQA
            external_daids, config2_=qreq_.extern_data_config2, **externgetkw
        )

    @profile
    def ensure_features(qreq_, verbose=ut.NOT_QUIET, prog_hook=None):
        r""" ensure features are computed
        Args:
            verbose (bool):  verbosity flag(default = True)

        CommandLine:
            python -m wbia.algo.hots.query_request --test-ensure_features

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.algo.hots.query_request import *  # NOQA
            >>> import wbia
            >>> ibs = wbia.opendb(defaultdb='testdb1')
            >>> daids = ibs.get_valid_aids()[0:2]
            >>> qaids = ibs.get_valid_aids()[0:3]
            >>> qreq_ = ibs.new_query_request(qaids, daids)
            >>> ibs.delete_annot_feats(qaids,  config2_=qreq_.extern_query_config2)  # Remove the chips
            >>> ut.remove_file_list(ibs.get_annot_chip_fpath(qaids, config2_=qreq_.extern_query_config2))
            >>> verbose = True
            >>> result = qreq_.ensure_features(verbose)
            >>> print(result)
        """
        # with ut.EmbedOnException():
        if verbose:
            print('[qreq] ensure_features')
        if prog_hook is not None:
            prog_hook(0, 3, 'ensure features')
        if prog_hook is not None:
            prog_hook(1, 3, 'ensure query features')
        qreq_.qannots.preload('kpts', 'vecs')
        if prog_hook is not None:
            prog_hook(2, 3, 'ensure database features')
        qreq_.dannots.preload('kpts')
        if prog_hook is not None:
            prog_hook(3, 3, 'computed features')

    @profile
    def ensure_featweights(qreq_, verbose=ut.NOT_QUIET):
        """ ensure feature weights are computed """
        if verbose:
            print('[qreq] ensure_featweights')
        qreq_.qannots.preload('fgweights')
        qreq_.dannots.preload('fgweights')

    @profile
    def load_indexer(qreq_, verbose=ut.NOT_QUIET, force=False, prog_hook=None):
        if not force and qreq_.indexer is not None:
            if prog_hook is not None:
                prog_hook.set_progress(1, 1, lbl='Indexer is loaded')
            return False
        else:
            index_method = qreq_.qparams.index_method
            if prog_hook is not None:
                prog_hook.set_progress(0, 1, lbl='Loading %s indexer' % (index_method,))
            if index_method == 'single':
                # TODO: SYSTEM updatable indexer
                if ut.VERYVERBOSE or verbose:
                    print('[qreq] loading single indexer normalizer')
                indexer = neighbor_index_cache.request_wbia_nnindexer(
                    qreq_,
                    verbose=verbose,
                    prog_hook=prog_hook,
                    **qreq_._indexer_request_params,
                )
            # elif index_method == 'multi':
            #    if ut.VERYVERBOSE or verbose:
            #        print('[qreq] loading multi indexer normalizer')
            #    indexer = multi_index.request_wbia_mindexer(
            #        qreq_, verbose=verbose)
            else:
                raise ValueError('unknown index_method=%r' % (index_method,))
            # if qreq_.prog_hook is not None:
            #    hook.set_progress(4, 4, lbl='building indexer')
            qreq_.indexer = indexer
            return True

    def get_infostr(qreq_):
        infostr_list = []
        app = infostr_list.append
        qaid_internal = qreq_.get_internal_qaids()
        daid_internal = qreq_.get_internal_daids()
        qd_intersection = ut.intersect_ordered(daid_internal, qaid_internal)
        app(' * len(internal_daids) = %r' % len(daid_internal))
        app(' * len(internal_qaids) = %r' % len(qaid_internal))
        app(' * len(qd_intersection) = %r' % len(qd_intersection))
        infostr = '\n'.join(infostr_list)
        return infostr

    def get_chipmatch_fpaths(qreq_, qaid_list, super_qres_cache=False):
        r"""
        Generates chipmatch paths for input query annotation rowids
        """
        dpath = qreq_.get_qresdir()
        if super_qres_cache:
            cfgstr = 'supercache'
        else:
            cfgstr = qreq_.get_cfgstr(with_input=False, with_data=True, with_pipe=True)
        qauuid_list = qreq_.get_qreq_pcc_uuids(qaid_list)
        for qaid, qauuid in zip(qaid_list, qauuid_list):
            fname = chip_match.get_chipmatch_fname(
                qaid, qreq_, qauuid=qauuid, cfgstr=cfgstr
            )
            fpath = join(dpath, fname)
            yield fpath

    def execute(
        qreq_, qaids=None, prog_hook=None, use_cache=None, invalidate_supercache=None
    ):
        r"""
        Runs the hotspotter pipeline and returns chip match objects.

        CommandLine:
            python -m wbia.algo.hots.query_request execute --show

        Example:
            >>> # SLOW_DOCTEST
            >>> # xdoctest: +SKIP
            >>> from wbia.algo.hots.query_request import *  # NOQA
            >>> import wbia
            >>> qreq_ = wbia.testdata_qreq_()
            >>> cm_list = qreq_.execute()
            >>> ut.quit_if_noshow()
            >>> cm = cm_list[0]
            >>> cm.ishow_analysis(qreq_)
            >>> ut.show_if_requested()
        """
        if qaids is not None:
            shallow_qreq_ = qreq_.shallowcopy(qaids=qaids)
            cm_list = shallow_qreq_.execute(
                prog_hook=prog_hook,
                use_cache=use_cache,
                invalidate_supercache=invalidate_supercache,
            )
            # cm_list = qreq_.ibs.query_chips(
            #    qreq_=shallow_qreq_, use_bigcache=False )
        else:
            from wbia.algo.hots import match_chips4 as mc4

            # Send query to hotspotter (runs the query)
            qreq_.prog_hook = prog_hook
            cm_list = mc4.submit_query_request(
                qreq_,
                use_cache=use_cache,
                use_bigcache=use_cache,
                verbose=True,
                save_qcache=use_cache,
                use_supercache=False,
                invalidate_supercache=invalidate_supercache,
            )
        return cm_list


def cfg_deepcopy_test():
    """
    TESTING FUNCTION

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.hots.query_request import *  # NOQA
        >>> result = cfg_deepcopy_test()
        >>> print(result)
    """
    import wbia

    ibs = wbia.opendb('testdb1')
    cfg1 = ibs.cfg.query_cfg
    cfg2 = cfg1.deepcopy()
    cfg3 = cfg2
    assert cfg1.get_cfgstr() == cfg2.get_cfgstr()
    assert cfg2.sv_cfg is not cfg1.sv_cfg
    assert cfg3.sv_cfg is cfg2.sv_cfg
    cfg2.update_query_cfg(sv_on=False)
    assert cfg1.get_cfgstr() != cfg2.get_cfgstr()
    assert cfg2.get_cfgstr() == cfg3.get_cfgstr()


if __name__ == '__main__':
    """
    CommandLine:
        utprof.sh -m wbia.algo.hots.query_request --test-QueryParams
        python -m wbia.algo.hots.query_request
        python -m wbia.algo.hots.query_request --allexamples
        python -m wbia.algo.hots.query_request --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    ut.doctest_funcs()
