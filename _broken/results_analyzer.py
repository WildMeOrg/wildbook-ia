# -*- coding: utf-8 -*-
"""
not really used
most things in here can be depricated
"""
from __future__ import absolute_import, division, print_function
import utool as ut
import numpy as np
import six
from six.moves import zip
from ibeis.other import ibsfuncs
#from ibeis.expt import results_organizer
print, rrr, profile = ut.inject2(__name__, '[resorg]')


def print_desc_distances_map(orgres2_distmap):
    print('+-----------------------------')
    print('| DESCRIPTOR MATCHE DISTANCES:')
    for orgtype, distmap in six.iteritems(orgres2_distmap):
        print('| orgtype(%r)' % (orgtype,))
        for disttype, dists in six.iteritems(distmap):
            print('|     disttype(%12r): %s' % (disttype, ut.get_stats_str(dists)))
    print('L-----------------------------')


def print_annotationmatch_scores_map(orgres2_scores):
    print('+-----------------------------')
    print('| CHIPMATCH SCORES:')
    for orgtype, scores in six.iteritems(orgres2_scores):
        print('| orgtype(%r)' % (orgtype,))
        print('|     scores: %s' % (ut.get_stats_str(scores)))
    print('L-----------------------------')


def get_orgres_annotationmatch_scores(allres, orgtype_list=['false', 'true'], verbose=True):
    orgres2_scores = {}
    for orgtype in orgtype_list:
        if verbose:
            print('[rr2] getting orgtype=%r distances between sifts' % orgtype)
        orgres = allres.get_orgtype(orgtype)
        ranks  = orgres.ranks
        scores = orgres.scores
        valid_scores = scores[ranks >= 0]  # None is less than 0
        orgres2_scores[orgtype] = valid_scores
    return orgres2_scores


def get_orgres_desc_match_dists(allres, orgtype_list=['false', 'true'],
                                distkey_list=['L2'],
                                verbose=True):
    r"""
    computes distances between matching descriptors of orgtypes in allres

    Args:
        allres (AllResults): AllResults object
        orgtype_list (list): of strings denoting the type of results to compare
        distkey_list (list): list of requested distance types

    Returns:
        dict: orgres2_descmatch_dists mapping from orgtype to dicts of distances (ndarrays)

    Notes:
        Just SIFT distance seems to have a very interesting property

    CommandLine:
        python -m ibeis.expt.results_analyzer --test-get_orgres_desc_match_dists
        python -m ibeis.expt.results_analyzer --test-get_orgres_desc_match_dists --show
        python -m ibeis.expt.results_analyzer --test-get_orgres_desc_match_dists --db PZ_Master0 --show
        python -m ibeis.expt.results_analyzer --test-get_orgres_desc_match_dists --db PZ_Master0 --distkeys=fs,lnbnn,bar_L2_sift --show
        python -m ibeis.expt.results_analyzer --test-get_orgres_desc_match_dists --db PZ_MTEST --distkeys=fs,lnbnn,bar_L2_sift,cos_sift --show
        python -m ibeis.expt.results_analyzer --test-get_orgres_desc_match_dists --db PZ_Master0 --distkeys=fs,lnbnn,bar_L2_sift,cos_sift --show
        python -m ibeis.expt.results_analyzer --test-get_orgres_desc_match_dists --db PZ_MTEST --distkeys=cos_sift --show
        python -m ibeis.expt.results_analyzer --test-get_orgres_desc_match_dists --db PZ_Master0 --distkeys=fs,lnbnn,bar_L2_sift,cos_sift --show --nosupport

        python -m ibeis.expt.results_analyzer --test-get_orgres_desc_match_dists --db PZ_MTEST --distkeys=lnbnn --show --feat_type=hesaff+siam128
        python -m ibeis.expt.results_analyzer --test-get_orgres_desc_match_dists --db PZ_MTEST --distkeys=lnbnn --show --feat_type=hesaff+sift

        python -m ibeis.expt.results_analyzer --test-get_orgres_desc_match_dists --db PZ_MTEST --distkeys=lnbnn --show --feat_type=hesaff+sift --num-top-fs=2
        python -m ibeis.expt.results_analyzer --test-get_orgres_desc_match_dists --db PZ_MTEST --distkeys=lnbnn --show --feat_type=hesaff+sift --num-top-fs=10
        python -m ibeis.expt.results_analyzer --test-get_orgres_desc_match_dists --db PZ_MTEST --distkeys=lnbnn --show --feat_type=hesaff+sift --num-top-fs=1000
        python -m ibeis.expt.results_analyzer --test-get_orgres_desc_match_dists --db PZ_MTEST --distkeys=lnbnn --show --feat_type=hesaff+siam128 --num-top-fs=1

    Example:
        >>> # SLOW_DOCTEST
        >>> from ibeis.expt.results_analyzer import *  # NOQA
        >>> from ibeis.expt import results_all
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='PZ_MTEST')
        >>> qaid_list = ibs.get_valid_aids(hasgt=True)
        >>> from ibeis.model import Config
        >>> cfgdict = ut.argparse_dict(dict(Config.parse_config_items(Config.QueryConfig())), only_specified=True)
        >>> allres = results_all.get_allres(ibs, qaid_list, cfgdict=cfgdict)
        >>> # {'feat_type': 'hesaff+siam128'})
        >>> orgtype_list = ['false', 'top_true']
        >>> verbose = True
        >>> distkey_list = ut.get_argval('--distkeys', type_=list, default=['fs', 'lnbnn', 'bar_L2_sift'])
        >>> #distkey_list = ['hist_isect']
        >>> #distkey_list = ['L2_sift', 'bar_L2_sift']
        >>> # execute function
        >>> orgres2_descmatch_dists = get_orgres_desc_match_dists(allres, orgtype_list, distkey_list, verbose)
        >>> #print('orgres2_descmatch_dists = ' + ut.dict_str(orgres2_descmatch_dists, truncate=-1, precision=3))
        >>> stats_ = {key: ut.dict_val_map(val, ut.get_stats) for key, val in orgres2_descmatch_dists.items()}
        >>> print('orgres2_descmatch_dists = ' + ut.dict_str(stats_, truncate=2, precision=3, nl=4))
        >>> # ------ VISUALIZE ------------
        >>> ut.quit_if_noshow()
        >>> import vtool as vt
        >>> # If viewing a large amount of data this might help on OverFlowError
        >>> #ut.embed()
        >>> # http://stackoverflow.com/questions/20330475/matplotlib-overflowerror-allocated-too-many-blocks
        >>> # http://matplotlib.org/1.3.1/users/customizing.html
        >>> limit_ = len(qaid_list) > 100
        >>> if limit_ or True:
        >>>     import matplotlib as mpl
        >>>     mpl.rcParams['agg.path.chunksize'] = 100000
        >>> # visualize the descriptor scores
        >>> for fnum, distkey in enumerate(distkey_list, start=1):
        >>>     encoder = vt.ScoreNormalizer()
        >>>     tn_scores, tp_scores = ut.get_list_column(ut.dict_take(orgres2_descmatch_dists, orgtype_list), distkey)
        >>>     encoder.fit_partitioned(tp_scores, tn_scores, verbose=False)
        >>>     figtitle = 'Descriptor Distance: %r. db=%r\norgtype_list=%r' % (distkey, ibs.get_dbname(), orgtype_list)
        >>>     use_support = not ut.get_argflag('--nosupport')
        >>>     encoder.visualize(figtitle=figtitle, use_stems=not limit_, fnum=fnum, with_normscore=use_support, with_scores=use_support)
        >>> ut.show_if_requested()
    """
    import vtool as vt
    orgres2_descmatch_dists = {}
    desc_dist_xs, other_xs = vt.index_partition(distkey_list, vt.VALID_DISTS)
    distkey_list1 = ut.list_take(distkey_list, desc_dist_xs)
    distkey_list2 = ut.list_take(distkey_list, other_xs)

    for orgtype in orgtype_list:
        if verbose:
            print('[rr2] getting orgtype=%r distances between vecs' % orgtype)
        orgres = allres.get_orgtype(orgtype)
        qaids = orgres.qaids
        aids  = orgres.aids
        # DO distance that need real computation
        if len(desc_dist_xs) > 0:
            try:
                stacked_qvecs, stacked_dvecs = get_matching_descriptors(allres, qaids, aids)
            except Exception as ex:
                orgres.printme3()
                ut.printex(ex)
                raise
            if verbose:
                print('[rr2]  * stacked_qvecs.shape = %r' % (stacked_qvecs.shape,))
                print('[rr2]  * stacked_dvecs.shape = %r' % (stacked_dvecs.shape,))
            #distkey_list = ['L1', 'L2', 'hist_isect', 'emd']
            #distkey_list = ['L1', 'L2', 'hist_isect']
            #distkey_list = ['L2', 'hist_isect']
            hist1 = np.asarray(stacked_qvecs, dtype=np.float32)
            hist2 = np.asarray(stacked_dvecs, dtype=np.float32)
            # returns an ordered dictionary
            distances1 = vt.compute_distances(hist1, hist2, distkey_list1)
        else:
            distances1 = {}
        # DO precomputed distances like fs (true weights) or lnbnn
        if len(other_xs) > 0:
            distances2 = ut.odict([(disttype, []) for disttype in distkey_list2])
            for qaid, daid in zip(qaids, aids):
                try:
                    qres = allres.qaid2_qres[qaid]
                    for disttype in distkey_list2:
                        if disttype == 'fs':
                            # hack in full fs
                            assert disttype == 'fs', 'unimplemented'
                            vals = qres.aid2_fs[daid]
                        else:
                            assert disttype in qres.filtkey_list, 'no score labeled %' % (disttype,)
                            index = qres.filtkey_list.index(disttype)
                            vals = qres.aid2_fsv[daid].T[index]
                        if len(vals) == 0:
                            continue
                        else:
                            # individual score component
                            pass
                        #num_top_vec_scores = None
                        num_top_vec_scores = ut.get_argval('--num-top-fs', type_=int, default=None)
                        if num_top_vec_scores is not None:
                            # Take only the best matching descriptor scores for each pair in this analysis
                            # This tries to see how deperable the BEST descriptor score is for each match
                            vals = vals[vals.argsort()[::-1][0:num_top_vec_scores]]
                            vals = vals[vals.argsort()[::-1][0:num_top_vec_scores]]
                        distances2[disttype].extend(vals)
                except KeyError:
                    continue
            # convert to numpy array
            for disttype in distkey_list2:
                distances2[disttype] = np.array(distances2[disttype])
        else:
            distances2 = {}
        # Put things back in expected order
        dist1_vals = ut.dict_take(distances1, distkey_list1)
        dist2_vals = ut.dict_take(distances2, distkey_list2)
        dist_vals = vt.rebuild_partition(dist1_vals, dist2_vals, desc_dist_xs, other_xs)
        distances = ut.odict(list(zip(distkey_list, dist_vals)))
        orgres2_descmatch_dists[orgtype] = distances
    return orgres2_descmatch_dists


def get_matching_descriptors(allres, qaid_list, daid_list):
    """
    returns a set of matching descriptors from queries to database annotations
    in allres

    Args:
        allres (AllResults): all results object
        qaid_list (list): query annotation id list
        daid_list (list): database annotation id list

    Returns:
        tuple: (stacked_qvecs, stacked_dvecs)

    Example:
        >>> from ibeis.expt.results_analyzer import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> allres = results_all.get_allres(ibs, ibs.get_valid_aids(hasgt=True))
        >>> orgres = allres.get_orgtype('top_true')
        >>> qaid_list = orgres.qaids
        >>> daid_list  = orgres.aids
        >>> (stacked_qvecs, stacked_dvecs) = get_matching_descriptors(allres, qaid_list, daid_list)
        >>> print((stacked_qvecs, stacked_dvecs))
    """
    ibs = allres.ibs
    qvecs_cache = ibsfuncs.get_annot_vecs_cache(ibs, qaid_list)
    dvecs_cache = ibsfuncs.get_annot_vecs_cache(ibs, daid_list)
    qvecs_list = []
    dvecs_list = []
    for qaid, daid in zip(qaid_list, daid_list):
        try:
            #fm = get_feat_matches(allres, qaid, daid)
            #def get_feat_matches(allres, qaid, aid):
            try:
                qres = allres.qaid2_qres[qaid]
                fm = qres.aid2_fm[daid]
            except KeyError:
                if ut.VERBOSE:
                    #if 'suppress_further' not in vars():
                    print('[rr2] Failed qaid=%r, daid=%r' % (qaid, daid))
                    # Probably just a match that doesnt exist
                    # not a big deal.
                    #suppress_further = True  # NOQA
                raise
            if len(fm) == 0:
                continue
        except KeyError:
            continue
        qvecs_m = qvecs_cache[qaid][fm.T[0]]
        dvecs_m = dvecs_cache[daid][fm.T[1]]
        qvecs_list.append(qvecs_m)
        dvecs_list.append(dvecs_m)
    del dvecs_cache
    del qvecs_cache
    try:
        stacked_qvecs = np.vstack(qvecs_list)
        stacked_dvecs = np.vstack(dvecs_list)
    except Exception as ex:
        ut.printex(ex, '[!!!] get_matching_descriptors',
                   keys=['qaid_list', 'daid_list', 'qvecs_list', 'dvecs_list'
                         'stacked_qvecs', 'stacked_dvecs'])
        raise
    return stacked_qvecs, stacked_dvecs


#def get_score_stuff_pdfish(allres):
#    """ In development """
#    true_orgres  = allres.get_orgtype('true')
#    false_orgres = allres.get_orgtype('false')
#    orgres = true_orgres
#    orgres = false_orgres
#    def get_interesting_annotationpairs(orgres):
#        orgres2 = results_organizer._score_sorted_ranks_top(orgres, 2)


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.expt.results_analyzer
        python -m ibeis.expt.results_analyzer --allexamples
        python -m ibeis.expt.results_analyzer --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
