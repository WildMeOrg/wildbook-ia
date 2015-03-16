from __future__ import absolute_import, division, print_function
import numpy as np
import utool as ut
import vtool as vt
from operator import xor
from vtool import matching
import six
#from ibeis.model.hots import hstypes
#from collections import namedtuple, defaultdict
from ibeis.model.hots import old_chip_match
from ibeis.model.hots import scoring
from ibeis.model.hots import name_scoring
from ibeis.model.hots import _pipeline_helpers as plh  # NOQA
print, print_,  printDBG, rrr, profile = ut.inject(__name__, '[chip_match]', DEBUG=False)


DEBUG_CHIPMATCH = False

#import six


def test_from_qres(qres):
    """
    CommandLine:
        python -m ibeis.model.hots.chip_match --test-test_from_qres

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.chip_match import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('testdb1')
        >>> qres = ibs._query_chips4([1], [2, 3, 4, 5], cfgdict=dict(), verbose=True)[1]
        >>> cm = ChipMatch2.from_qres(qres)
        >>> cm.print_csv(ibs=ibs)
    """
    pass


@six.add_metaclass(ut.ReloadingMetaclass)
class ChipMatch2(old_chip_match._OldStyleChipMatchSimulator):
    """
    behaves as as the ChipMatchOldTup named tuple until we
    completely replace the old structure
    """

    # Alternative  Cosntructors

    @classmethod
    def from_qres(cls, qres):
        r"""
        """
        aid2_fm_    = qres.aid2_fm
        aid2_fsv_   = qres.aid2_fsv
        aid2_fk_    = qres.aid2_fk
        aid2_score_ = qres.aid2_score
        aid2_H_     = qres.aid2_H
        qaid        = qres.qaid
        cmtup_old = (aid2_fm_, aid2_fsv_, aid2_fk_, aid2_score_, aid2_H_)
        fsv_col_lbls = qres.filtkey_list
        cm = ChipMatch2.from_cmtup_old(cmtup_old, qaid, fsv_col_lbls)
        fs_list = [fsv.T[cm.fsv_col_lbls.index('lnbnn')] for fsv in cm.fsv_list]
        cm.fs_list = fs_list
        return cm

    @classmethod
    def from_unscored(cls, prior_cm, fm_list, fs_list, H_list=None, fsv_col_lbls=None):
        qaid = prior_cm.qaid
        daid_list = prior_cm.daid_list
        fsv_list = matching.ensure_fsv_list(fs_list)
        if fsv_col_lbls is None:
            fsv_col_lbls = ['unknown']
            #fsv_col_lbls = [str(count) for count in range(num_cols)]
            #fsv_col_lbls
        #score_list = [fsv.prod(axis=1).sum() for fsv in fsv_list]
        score_list = [-1 for fsv in fsv_list]
        #fsv.prod(axis=1).sum() for fsv in fsv_list]
        cm = ChipMatch2(qaid, daid_list, fm_list, fsv_list, None, score_list, H_list, fsv_col_lbls)
        cm.fs_list = fs_list
        return cm

    @classmethod
    def from_vsmany_match_tup(cls, valid_match_tup, qaid=None, fsv_col_lbls=None):
        # Vsmany - create new cmtup_old
        (valid_daid, valid_qfx, valid_dfx, valid_scorevec, valid_rank) = valid_match_tup
        valid_fm = np.vstack((valid_qfx, valid_dfx)).T
        daid_list, daid_groupxs = vt.group_indices(valid_daid)
        fm_list  = vt.apply_grouping(valid_fm, daid_groupxs)
        fsv_list = vt.apply_grouping(valid_scorevec, daid_groupxs)
        fk_list  = vt.apply_grouping(valid_rank, daid_groupxs)
        cm = ChipMatch2(qaid, daid_list, fm_list, fsv_list, fk_list, fsv_col_lbls=fsv_col_lbls)
        return cm

    @classmethod
    def from_vsone_match_tup(cls, valid_match_tup_list, daid_list=None, qaid=None, fsv_col_lbls=None):
        assert all(list(map(ut.list_allsame, ut.get_list_column(valid_match_tup_list, 0)))),\
            'internal daids should not have different daids for vsone'
        qfx_list = ut.get_list_column(valid_match_tup_list, 1)
        dfx_list = ut.get_list_column(valid_match_tup_list, 2)
        fm_list  = [np.vstack(dfx_qfx).T for dfx_qfx in zip(dfx_list, qfx_list)]
        fsv_list = ut.get_list_column(valid_match_tup_list, 3)
        fk_list  = ut.get_list_column(valid_match_tup_list, 4)
        cm = ChipMatch2(qaid, daid_list, fm_list, fsv_list, fk_list, fsv_col_lbls=fsv_col_lbls)
        return cm

    # Standard Contstructor

    def __init__(cm, qaid=None, daid_list=None, fm_list=None, fsv_list=None, fk_list=None,
                 score_list=None, H_list=None, fsv_col_lbls=None, dnid_list=None, qnid=None):
        """
        qaid and daid_list are not optional. fm_list and fsv_list are strongly
        encouraged and will probalby break things if they are not there.
        """
        if DEBUG_CHIPMATCH:
            assert daid_list is not None, 'must give daids'
            assert fm_list is None or len(fm_list) == len(daid_list), 'incompatable data'
            assert fsv_list is None or len(fsv_list) == len(daid_list), 'incompatable data'
            assert fk_list is None or len(fk_list) == len(daid_list), 'incompatable data'
            assert H_list is None or len(H_list) == len(daid_list), 'incompatable data'
            assert score_list is None or len(score_list) == len(daid_list), 'incompatable data'
            assert dnid_list is None or len(dnid_list) == len(daid_list), 'incompatable data'
        cm.qaid         = qaid
        cm.daid_list    = np.array(daid_list)
        cm.fm_list      = fm_list
        cm.fsv_list     = fsv_list
        cm.fk_list      = (fk_list if fk_list is not None else
                           [np.zeros(fm.shape[0]) for fm in cm.fm_list])
        cm.score_list   = score_list
        cm.H_list       = H_list
        cm.fsv_col_lbls = fsv_col_lbls
        cm.daid2_idx    = None
        cm.fs_list = None
        # TODO
        cm.dnid_list = dnid_list
        # standard groupings
        cm.unique_nids = None  # belongs to name_groupxs
        cm.nid2_nidx = None
        cm.name_groupxs = None
        cm.qnid = qnid
        # Name probabilities
        cm.prob_list = None
        # Annot scores
        cm.annot_score_list = None
        # Name scores
        cm.name_score_list = None
        # TODO: have subclass or dict for special scores
        cm.csum_score_list = None
        cm.nsum_score_list = None
        cm.acov_score_list = None
        cm.ncov_score_list = None
        #
        cm._update_daid_index()

    #------------------
    # Modification / Evaluation Functions
    #------------------

    def _update_daid_index(cm):
        cm.daid2_idx = (None if cm.daid_list is None else
                        {daid: idx for idx, daid in enumerate(cm.daid_list)})

    def evaluate_dnids(cm, ibs):
        cm.qnid = ibs.get_annot_name_rowids(cm.qaid)
        cm.dnid_list = np.array(ibs.get_annot_name_rowids(cm.daid_list))
        # evaluate name groupings as well
        unique_nids, name_groupxs = vt.group_indices(cm.dnid_list)
        cm.unique_nids  = unique_nids
        cm.name_groupxs = name_groupxs
        cm.nid2_nidx    = ut.make_index_lookup(cm.unique_nids)

    def sortself(cm):
        """ reorders the internal data using cm.score_list """
        sortx               = cm.argsort()
        cm.daid_list        = vt.trytake(cm.daid_list, sortx)
        cm.dnid_list        = vt.trytake(cm.dnid_list, sortx)
        cm.fm_list          = vt.trytake(cm.fm_list, sortx)
        cm.fsv_list         = vt.trytake(cm.fsv_list, sortx)
        cm.fs_list          = vt.trytake(cm.fs_list, sortx)
        cm.fk_list          = vt.trytake(cm.fk_list, sortx)
        cm.score_list       = vt.trytake(cm.score_list, sortx)
        cm.csum_score_list  = vt.trytake(cm.csum_score_list, sortx)
        cm.H_list           = vt.trytake(cm.H_list, sortx)
        cm._update_daid_index()

    def shortlist_subset(cm, top_aids):
        """ returns a new cmtup_old with only the requested daids """
        qaid         = cm.qaid
        qnid         = cm.qnid
        idx_list     = ut.dict_take(cm.daid2_idx, top_aids)
        daid_list    = vt.list_take_(cm.daid_list, idx_list)
        fm_list      = vt.list_take_(cm.fm_list, idx_list)
        fsv_list     = vt.list_take_(cm.fsv_list, idx_list)
        fk_list      = vt.trytake(cm.fk_list, idx_list)
        #score_list   = vt.trytake(cm.score_list, idx_list)
        score_list   = None  # don't transfer scores
        H_list       = vt.trytake(cm.H_list, idx_list)
        dnid_list    = vt.trytake(cm.dnid_list, idx_list)
        fsv_col_lbls = cm.fsv_col_lbls
        cm_subset = ChipMatch2(qaid, daid_list, fm_list, fsv_list, fk_list,
                               score_list, H_list, fsv_col_lbls, dnid_list, qnid)
        return cm_subset

    #------------------
    # Getter Functions
    #------------------

    def get_num_feat_score_cols(cm):
        return len(cm.fsv_col_lbls)

    def get_fs(cm, idx=None, colx=None, daid=None, col=None):
        assert xor(idx is None, daid is None)
        assert xor(colx is None or col is None)
        if daid is not None:
            idx = cm.daid2_idx[daid]
        if col is not None:
            colx = cm.fsv_col_lbls.index(col)
        fs = cm.fsv_list[idx][colx]
        return fs

    def get_fsv_prod_list(cm):
        return [fsv.prod(axis=1) for fsv in cm.fsv_list]

    def get_annot_fm(cm, daid):
        idx = ut.dict_take(cm.daid2_idx, daid)
        fm  = ut.list_take(cm.fm_list, idx)
        return fm

    def get_fs_list(cm, colx=None, col=None):
        assert xor(colx is None, col is None)
        if col is not None:
            colx = cm.fsv_col_lbls.index(col)
        fs_list = [fsv.T[colx].T for fsv in cm.fsv_list]
        return fs_list

    def get_groundtruth_flags(cm):
        assert cm.dnid_list is not None, 'run cm.evaluate_dnids'
        gt_flags = cm.dnid_list == cm.qnid
        return gt_flags

    def get_groundtruth_daids(cm):
        gt_flags = cm.get_groundtruth_flags()
        gt_daids = vt.list_compress_(cm.daid_list, gt_flags)
        return gt_daids

    def get_nid_scores(cm, nid_list):
        nidx_list = ut.dict_take(cm.nid2_nidx, nid_list)
        name_scores = vt.list_take_(cm.name_score_list, nidx_list)
        return name_scores

    def get_num_matches_list(cm):
        num_matches_list = list(map(len, cm.fm_list))
        return num_matches_list

    def argsort(cm):
        if cm.score_list is None:
            num_matches_list = cm.get_num_matches_list()
            sortx = ut.list_argsort(num_matches_list, reverse=True)
        else:
            sortx = ut.list_argsort(cm.score_list, reverse=True)
        return sortx

    def get_name_shortlist_aids(cm, nNameShortList, nAnnotPerName):
        """
        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.model.hots.chip_match import *  # NOQA
            >>> ibs, qreq_, cm_list = plh.testdata_pre_sver('PZ_MTEST', qaid_list=[18])
            >>> cm = cm_list[0]
            >>> cm.score_nsum(qreq_)
            >>> top_daids = cm.get_name_shortlist_aids(5, 2)
            >>> assert cm.qnid in ibs.get_annot_name_rowids(top_daids)
        """
        top_daids = scoring.get_name_shortlist_aids(cm.daid_list, cm.dnid_list,
                                                    cm.annot_score_list, cm.name_score_list,
                                                    cm.nid2_nidx, nNameShortList, nAnnotPerName)
        return top_daids

    #+=================
    # Scoring Functions
    #------------------

    # Cannonical Setters

    def set_cannonical_annot_score(cm, annot_score_list):
        cm.annot_score_list = annot_score_list
        #cm.name_score_list  = None
        cm.score_list       = annot_score_list

    def set_cannonical_name_score(cm, annot_score_list, name_score_list):
        cm.annot_score_list = annot_score_list
        cm.name_score_list  = name_score_list
        # align with score_list
        cm.score_list = name_scoring.align_name_scores_with_annots(
            cm.annot_score_list, cm.daid_list, cm.daid2_idx, cm.name_groupxs, cm.name_score_list)

    # ChipSum Score

    def evaluate_csum_score(cm, qreq_):
        csum_score_list = scoring.compute_csum_score(cm)
        cm.csum_score_list = csum_score_list

    def score_csum(cm, qreq_):
        """
        CommandLine:
            python -m ibeis.model.hots.chip_match --test-score_csum --show
            python -m ibeis.model.hots.chip_match --test-score_csum --show --qaid 18

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.model.hots.chip_match import *  # NOQA
            >>> ibs, qreq_, cm_list = plh.testdata_post_sver()
            >>> cm = cm_list[0]
            >>> cm.score_csum(qreq_)
            >>> cm.testshow_ranked(qreq_, figtitle='score_csum')
        """
        cm.evaluate_csum_score(qreq_)
        cm.set_cannonical_annot_score(cm.csum_score_list)

    # NameSum Score

    def evaluate_nsum_score(cm, qreq_):
        cm.evaluate_dnids(qreq_.ibs)
        nsum_nid_list, nsum_score_list = name_scoring.compute_nsum_score(cm, qreq_=qreq_)
        assert np.all(cm.unique_nids == nsum_nid_list), 'name score not in alignment'
        cm.nsum_score_list = nsum_score_list

    def score_nsum(cm, qreq_):
        """
        CommandLine:
            python -m ibeis.model.hots.chip_match --test-score_nsum --show --qaid 1
            python -m ibeis.model.hots.chip_match --test-score_nsum --show --qaid 18

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.model.hots.chip_match import *  # NOQA
            >>> ibs, qreq_, cm_list = plh.testdata_post_sver('PZ_MTEST', qaid_list=[18])
            >>> cm = cm_list[0]
            >>> cm.score_nsum(qreq_)
            >>> gt_score = cm.score_list.compress(cm.get_groundtruth_flags()).max()
            >>> cm.print_csv()
            >>> assert cm.get_top_nids()[0] == cm.unique_nids[cm.name_score_list.argmax()], 'bug in alignment'
            >>> cm.testshow_ranked(qreq_, figtitle='score_nsum')
            >>> assert cm.get_top_nids()[0] == cm.qnid, 'is this case truely hard?'
        """
        cm.evaluate_csum_score(qreq_)
        cm.evaluate_nsum_score(qreq_)
        cm.set_cannonical_name_score(cm.csum_score_list, cm.nsum_score_list)

    # ChipCoverage Score

    def evaluate_acov_score(cm, qreq_):
        daid_list, acov_score_list = scoring.compute_annot_coverage_score(qreq_, cm, qreq_.qparams)
        assert np.all(daid_list == np.array(cm.daid_list)), 'daids out of alignment'
        cm.acov_score_list = acov_score_list

    def score_annot_coverage(cm, qreq_):
        """
        CommandLine:
            python -m ibeis.model.hots.chip_match --test-score_annot_coverage --show
            python -m ibeis.model.hots.chip_match --test-score_annot_coverage --show --qaid 18

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.model.hots.chip_match import *  # NOQA
            >>> ibs, qreq_, cm_list = plh.testdata_post_sver()
            >>> cm = cm_list[0]
            >>> cm.fs_list = cm.get_fs_list(col='lnbnn')
            >>> cm.score_annot_coverage(qreq_)
            >>> cm.testshow_ranked(qreq_, figtitle='score_annot_coverage')
        """
        cm.evaluate_acov_score(qreq_)
        cm.set_cannonical_annot_score(cm.acov_score_list)

    # NameCoverage Score

    def evaluate_ncov_score(cm, qreq_):
        cm.evaluate_dnids(qreq_.ibs)
        ncov_nid_list, ncov_score_list = scoring.compute_name_coverage_score(qreq_, cm, qreq_.qparams)
        assert np.all(cm.unique_nids == ncov_nid_list)
        cm.ncov_score_list = ncov_score_list

    def score_name_coverage(cm, qreq_):
        """
        CommandLine:
            python -m ibeis.model.hots.chip_match --test-score_name_coverage --show
            python -m ibeis.model.hots.chip_match --test-score_name_coverage --show --qaid 18

        Example:
            >>> # ENABLE_DOCTEST
            >>> from ibeis.model.hots.chip_match import *  # NOQA
            >>> ibs, qreq_, cm_list = plh.testdata_post_sver()
            >>> cm = cm_list[0]
            >>> cm.fs_list = cm.get_fs_list(col='lnbnn')
            >>> cm.score_name_coverage(qreq_)
            >>> cm.testshow_ranked(qreq_, figtitle='score_name_coverage')
        """
        if cm.csum_score_list is None:
            cm.evaluate_csum_score(qreq_)
        cm.evaluate_ncov_score(qreq_)
        cm.set_cannonical_name_score(cm.csum_score_list, cm.ncov_score_list)

    #------------------
    # Result Functions
    #------------------

    def get_top_scores(cm, ntop=None):
        sortx = cm.score_list.argsort()[::-1]
        _top_scores = vt.list_take_(cm.score_list, sortx)
        top_scores = ut.listclip(_top_scores, ntop)
        return top_scores

    def get_top_nids(cm, ntop=None):
        sortx = cm.score_list.argsort()[::-1]
        _top_nids = vt.list_take_(cm.dnid_list, sortx)
        top_nids = ut.listclip(_top_nids, ntop)
        return top_nids

    def get_top_aids(cm, ntop=None):
        sortx = cm.score_list.argsort()[::-1]
        _top_aids = vt.list_take_(cm.daid_list, sortx)
        top_aids = ut.listclip(_top_aids, ntop)
        return top_aids

    #------------------
    # String Functions
    #------------------

    def print_rawinfostr(cm):
        print(cm.get_rawinfostr())

    def print_csv(cm, *args, **kwargs):
        print(cm.get_cvs_str(*args, **kwargs))

    def get_rawinfostr(cm):
        def varinfo(varname, forcerepr=False):
            import utool as ut
            varval = getattr(cm, varname.replace('cm.', ''))
            if not forcerepr and ut.isiterable(varval):
                varinfo_list = [
                    '    * varinfo(cm.%s):' % (varname,),
                    '        depth = %r' % (ut.depth_profile(varval),),
                    '        types = %s' % (ut.list_type_profile(varval),),
                ]
                #varinfo = '\n'.join(ut.align_lines(varinfo_list, '='))
                varinfo = '\n'.join(ut.align_lines(varinfo_list, '='))
            else:
                varinfo = '    * cm.%s = %r' % (varname, varval)
            return varinfo
        str_list = []
        append = str_list.append
        append('ChipMatch2:')
        append('    * cm.qaid = %r' % (cm.qaid,))
        append('    * cm.qnid = %r' % (cm.qnid,))
        append('    * len(cm.daid2_idx) = %r' % (len(cm.daid2_idx),))
        append(varinfo('cm.fsv_col_lbls', forcerepr=True))
        append(varinfo('cm.daid_list'))
        append(varinfo('cm.dnid_list'))
        append(varinfo('cm.fs_list'))
        append(varinfo('cm.fm_list'))
        append(varinfo('cm.fk_list'))
        append(varinfo('cm.fsv_list'))
        append(varinfo('cm.H_list'))
        append(varinfo('cm.score_list'))
        append(varinfo('cm.name_score_list'))
        append(varinfo('cm.annot_score_list'))
        #
        append(varinfo('cm.csum_score_list'))
        append(varinfo('cm.nsum_score_list'))
        append(varinfo('cm.acov_score_list'))
        append(varinfo('cm.ncov_score_list'))
        #append(varinfo('cm.annot_score_dict[\'csum\']'))
        #append(varinfo('cm.annot_score_dict[\'acov\']'))
        #append(varinfo('cm.name_score_dict[\'nsum\']'))
        #append(varinfo('cm.name_score_dict[\'ncov\']'))
        #infostr = '\n'.join(ut.align_lines(str_list, '='))
        infostr = '\n'.join(str_list)
        return infostr

    def get_cvs_str(cm,  numtop=6, ibs=None, sort=True):
        """
        Notes:
        Very weird that it got a score

        qaid 6 vs 41 has
            [72, 79, 0, 17, 6, 60, 15, 36, 63]
            [72, 79, 0, 17, 6, 60, 15, 36, 63]
            [72, 79, 0, 17, 6, 60, 15, 36, 63]
            [0.060, 0.053, 0.0497, 0.040, 0.016, 0, 0, 0, 0]
            [7, 40, 41, 86, 103, 88, 8, 101, 35]

        makes very little sense
        """
        if not sort or cm.score_list is None:
            if sort:
                print('Warning: cm.score_list is None and sort is True')
            sortx = list(range(len(cm.daid_list)))
        else:
            sortx = ut.list_argsort(cm.score_list, reverse=True)
        if ibs is not None:
            qnid = ibs.get_annot_nids(cm.qaid)
            dnid_list = ibs.get_annot_nids(cm.daid_list)
        else:
            qnid = cm.qnid
            dnid_list = cm.dnid_list
        # Build columns for the csv, filtering out unavailable information
        column_lbls_ = ['daid', 'dnid', 'score', 'num_matches', 'annot_scores', 'fm_depth', 'fsv_depth']
        column_list_ = [
            vt.list_take_(cm.daid_list,  sortx),
            None if dnid_list is None else vt.list_take_(dnid_list, sortx),
            None if cm.score_list is None else vt.list_take_(cm.score_list, sortx),
            vt.list_take_(cm.get_num_matches_list(), sortx),
            None if cm.annot_score_list is None else vt.list_take_(cm.annot_score_list, sortx),
            #None if cm.name_score_list is None else vt.list_take_(cm.name_score_list, sortx),
            ut.lmap(str, ut.depth_profile(vt.list_take_(cm.fm_list,  sortx))),
            ut.lmap(str, ut.depth_profile(vt.list_take_(cm.fsv_list, sortx))),
        ]
        isnone_list = ut.flag_None_items(column_list_)
        column_lbls = ut.filterfalse_items(column_lbls_, isnone_list)
        column_list = ut.filterfalse_items(column_list_, isnone_list)
        # Clip to the top results
        if numtop is not None:
            column_list = [ut.listclip(col, numtop) for col in column_list]
        # hard case for python text parsing
        # better know about quoted hash symbols
        header = ut.codeblock(
            '''
            # qaid = {qaid}
            # qnid = {qnid}
            # fsv_col_lbls = {fsv_col_lbls}
            '''
        ).format(qaid=cm.qaid, qnid=qnid, fsv_col_lbls=cm.fsv_col_lbls)

        csv_str = ut.make_csv_table(column_list, column_lbls, header, comma_repl=';')
        return csv_str

    #------------------
    # Testing Functions
    #------------------

    def assert_self(cm, qreq_=None, strict=False, verbose=ut.NOT_QUIET):
        assert cm.qaid is not None, 'must have qaid'
        assert cm.daid_list is not None, 'must give daids'
        assert cm.fm_list is None or len(cm.fm_list) == len(cm.daid_list), 'incompatable data'
        assert cm.fsv_list is None or len(cm.fsv_list) == len(cm.daid_list), 'incompatable data'
        assert cm.fk_list is None or len(cm.fk_list) == len(cm.daid_list), 'incompatable data'
        assert cm.H_list is None or len(cm.H_list) == len(cm.daid_list), 'incompatable data'
        assert cm.score_list is None or len(cm.score_list) == len(cm.daid_list), 'incompatable data'
        assert cm.dnid_list is None or len(cm.dnid_list) == len(cm.daid_list), 'incompatable data'

        if strict or cm.unique_nids is not None:
            assert np.all(cm.unique_nids[ut.dict_take(cm.nid2_nidx, cm.unique_nids)] == cm.unique_nids)
            print('[cm] unique nid alignment is ok')

        if strict or qreq_ is not None and cm.dnid_list is not None:
            assert np.all(cm.dnid_list == qreq_.ibs.get_annot_name_rowids(cm.daid_list)), 'bad nids'
            print('[cm] annot aligned nids are ok')

        if verbose:
            print('[cm] lengths are ok')
        try:
            assert ut.list_all_eq_to([fsv.shape[1] for fsv in cm.fsv_list], len(cm.fsv_col_lbls))
        except Exception as ex:
            cm.print_rawinfostr()
            raise
        assert ut.list_all_eq_to([fm.shape[1] for fm in cm.fm_list], 2), 'bad fm'
        if verbose:
            print('[cm] shapes are ok')
        if strict or qreq_ is not None:
            external_qaids = qreq_.get_external_qaids().tolist()
            external_daids = qreq_.get_external_daids().tolist()
            if qreq_.qparams.pipeline_root == 'vsone':
                assert len(external_qaids) == 1, 'only one external qaid for vsone'
                if strict or qreq_.indexer is not None:
                    nExternalQVecs = qreq_.ibs.get_annot_vecs(external_qaids[0], config2_=qreq_.get_external_query_config2()).shape[0]
                    assert qreq_.indexer.idx2_vec.shape[0] == nExternalQVecs, (
                        'did not index query descriptors properly')
                if verbose:
                    print('[cm] vsone daids are ok are ok')

            nFeats1 = qreq_.ibs.get_annot_num_feats(cm.qaid, config2_=qreq_.get_external_query_config2())
            nFeats2_list = np.array(qreq_.ibs.get_annot_num_feats(cm.daid_list, config2_=qreq_.get_external_data_config2()))
            try:
                assert ut.list_issubset(cm.daid_list, external_daids), 'cmtup_old must be subset of daids'
            except AssertionError as ex:
                ut.printex(ex, keys=['daid_list', 'external_daids'])
                raise
            try:
                fm_list = cm.fm_list
                fx2s_list = [fm_.T[1] for fm_ in fm_list]
                fx1s_list = [fm_.T[0] for fm_ in fm_list]
                max_fx1_list = np.array([-1 if len(fx1s) == 0 else fx1s.max() for fx1s in fx1s_list])
                max_fx2_list = np.array([-1 if len(fx2s) == 0 else fx2s.max() for fx2s in fx2s_list])
                ut.assert_lessthan(max_fx2_list, nFeats2_list, 'max feat index must be less than num feats')
                ut.assert_lessthan(max_fx1_list, nFeats1, 'max feat index must be less than num feats')
            except AssertionError as ex:
                ut.printex(ex, keys=['qaid', 'daid_list', 'nFeats1',
                                     'nFeats2_list', 'max_fx1_list',
                                     'max_fx2_list', ])
                raise
            if verbose:
                print('[cm] nFeats are ok in fm')

    def testshow_single(cm, qreq_, daid=None, lastcall=True, **kwargs):
        if ut.show_was_requested():
            import plottool as pt
            cm.show_single(qreq_, daid=None, **kwargs)
            if lastcall:
                pt.show_if_requested()

    def testshow_ranked(cm, qreq_, daid=None, lastcall=True, **kwargs):
        if ut.show_was_requested():
            import plottool as pt
            cm.show_ranked_matches(qreq_, **kwargs)
            if lastcall:
                pt.show_if_requested()

    def show_single(cm, qreq_, daid=None, fnum=None, pnum=None, homog=ut.get_argflag('--homog'), **kwargs):
        from ibeis.viz import viz_matches
        if daid is None:
            idx = cm.argsort()[0]
            daid = cm.daid_list[idx]
        else:
            idx = cm.daid2_idx[daid]
        fm   = cm.fm_list[idx]
        H1   = None if not homog or cm.H_list is None else cm.H_list[idx]
        fsv  = None if cm.fsv_list is None else cm.fsv_list[idx]
        fs   = None if fsv is None else fsv.prod(axis=1)
        showkw = dict(fm=fm, fs=fs, H1=H1, fnum=fnum, pnum=pnum, **kwargs)
        viz_matches.show_matches2(qreq_.ibs, cm.qaid, daid, qreq_=qreq_, **showkw)

    def show_ranked_matches(cm, qreq_, clip_top=6, *args, **kwargs):
        idx_list  = ut.listclip(cm.argsort(), clip_top)
        cm.show_index_matches(qreq_, idx_list, *args, **kwargs)

    def show_daids_matches(cm, qreq_, daids, *args, **kwargs):
        idx_list = ut.dict_take(cm.daid2_idx, daids)
        cm.show_index_matches(qreq_, idx_list, *args, **kwargs)

    def show_index_matches(cm, qreq_, idx_list, fnum=None, figtitle=None, **kwargs):
        import plottool as pt
        if fnum is None:
            fnum = pt.next_fnum()
        nRows, nCols  = pt.get_square_row_cols(len(idx_list), fix=True)
        next_pnum     = pt.make_pnum_nextgen(nRows, nCols)
        for idx in idx_list:
            daid  = cm.daid_list[idx]
            pnum = next_pnum()
            cm.show_single(qreq_, daid, fnum=fnum, pnum=pnum, **kwargs)
            score = vt.trytake(cm.score_list, idx)
            annot_score = vt.trytake(cm.annot_score_list, idx)
            score_str = ('score = %.3f' % (score,)
                         if score is not None else
                         'score = None')
            annot_score_str = ('annot_score = %.3f' % (annot_score,)
                               if annot_score is not None else
                               'annot_score = None')
            title = score_str + '\n' + annot_score_str
            pt.set_title(title)
        if figtitle is not None:
            pt.set_figtitle(figtitle)


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.model.hots.chip_match
        python -m ibeis.model.hots.chip_match --allexamples
        python -m ibeis.model.hots.chip_match --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
