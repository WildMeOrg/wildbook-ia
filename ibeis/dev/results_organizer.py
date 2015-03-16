"""
not really used
most things in here can be depricated
"""
from __future__ import absolute_import, division, print_function
# Python
import six
import utool as ut  # NOQA
from six.moves import zip
import numpy as np
# Tools
import utool
from utool import DynStruct
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[resorg]', DEBUG=False)


#
#
# OrganizedResult Class
#---------------------

class OrganizedResult(DynStruct):
    """
    What chips are populated depends on the type of organization

    Maintains an organized list of::
        * query annotation indexes
        * their top matching result
        * their score
        * their rank
    """
    def __init__(self, orgtype=''):
        super(DynStruct, self).__init__()
        self.orgtype = orgtype
        self.qaids   = []  # query annotation indexes
        self.aids    = []  # a matching result
        self.scores  = []  # the matching score
        self.ranks   = []  # the matching rank

    def append(self, qaid, aid, rank, score):
        self.qaids.append(qaid)
        self.aids.append(aid)
        self.scores.append(score)
        self.ranks.append(rank)

    def freeze(self):
        """ No more appending """
        self.qaids  = np.array(self.qaids)
        self.aids   = np.array(self.aids)
        self.scores = np.array(self.scores)
        self.ranks  = np.array(self.ranks)

    def where_ranks_lt(orgres, num):
        """ get new orgres where all the ranks are less or equal to """
        # Remove None ranks
        return _where_ranks_lt(orgres, num)

    def iter_sorted(self):
        qaids  = np.array(self.qaids)
        aids   = np.array(self.aids)
        scores = np.array(self.scores)
        ranks  = np.array(self.ranks)
        #
        sortx  = ranks.argsort()
        sorted_qaids  =  qaids[sortx]
        sorted_aids   =   aids[sortx]
        sorted_scores = scores[sortx]
        sorted_ranks  =  ranks[sortx]
        return (sorted_qaids, sorted_aids, sorted_scores, sorted_ranks)

    def __len__(self):
        num_qcxs   = len(self.qaids)
        num_aids   = len(self.aids)
        num_scores = len(self.scores)
        num_ranks  = len(self.ranks)
        assert num_qcxs == num_aids
        assert num_aids == num_scores
        assert num_scores == num_ranks
        return num_qcxs

    def iter(self):
        """ useful for plotting """
        result_iter = zip(self.qaids, self.aids, self.scores, self.ranks)
        for qaid, aid, score, rank in result_iter:
            yield qaid, aid, score, rank

    def printme3(self):
        column_list = [self.qaids, self.aids, self.scores, self.ranks]
        column_lbls = ['qaids', 'aids', 'scores', 'ranks']
        header = 'Orgres %s' % (self.orgtype)
        print(utool.make_csv_table(column_list, column_lbls, header, column_type=None))


def _where_ranks_lt(orgres, num):
    """ get new orgres where all the ranks are less or equal to """
    # Remove None ranks
    isvalid = [rank is not None and rank <= num and rank != -1
                for rank in orgres.ranks]
    orgres2 = OrganizedResult(orgres.orgtype + ' < %d' % num)
    orgres2.qaids  = utool.filter_items(orgres.qaids, isvalid)
    orgres2.aids   = utool.filter_items(orgres.aids, isvalid)
    orgres2.scores = utool.filter_items(orgres.scores, isvalid)
    orgres2.ranks  = utool.filter_items(orgres.ranks, isvalid)
    return orgres2


def _sorted_by_score(orgres):
    """ get new orgres where arrays are sorted by score """
    orgres2 = OrganizedResult(orgres.orgtype + ' score-sorted')
    sortx = np.array(orgres.scores).argsort()[::-1]
    orgres2.qaids  = np.array(orgres.qaids)[sortx]
    orgres2.aids   = np.array(orgres.aids)[sortx]
    orgres2.scores = np.array(orgres.scores)[sortx]
    orgres2.ranks  = np.array(orgres.ranks)[sortx]
    return orgres2


def _score_sorted_ranks_lt(orgres, num):
    orgres2 = _where_ranks_lt(orgres, num)
    orgres3 = _sorted_by_score(orgres2)
    return orgres3


def qres2_true_and_false(ibs, qres):
    """

    Organizes chip-vs-chip results into true positive set and false positive set

    a set is a query, its best match, and a score

    qres2_true_and_false

    Args:
        ibs (IBEISController):
        qres (QueryResult): object of feature correspondences and scores

    Returns:
        tuple: (true_tup, false_tup)
            * true_tup  = (true_aids,  true_scores,  true_ranks)
            * false_tup = (false_aids, false_scores, false_ranks)

    Example:
        >>> from ibeis.dev.results_organizer import *   # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> aid_list = ibs.get_valid_aids()
        >>> cfgdict = dict(codename='nsum',fg_on=True)
        >>> qaid_list = aid_list[0:1]
        >>> qaid2_qres = ibs._query_chips4(qaid_list, aid_list, cfgdict=cfgdict)
        >>> qres = qaid2_qres[qaid_list[0]]
        >>> (true_tup, false_tup) = qres2_true_and_false(ibs, qres)
        >>> print((true_tup, false_tup))
    """
    # Get top chip indexes and scores
    top_aids  = qres.get_top_aids()
    top_score = qres.get_aid_scores(top_aids)
    top_ranks = range(len(top_aids))
    # True Rids / Scores / Ranks
    true_ranks, true_aids = qres.get_gt_ranks(ibs=ibs, return_gtaids=True)
    true_scores  = [-1 if rank is None else top_score[rank] for rank in true_ranks]
    # False Rids / Scores / Ranks
    false_ranks = list(set(top_ranks) - set(true_ranks))
    false_aids   = [-1 if rank is None else top_aids[rank]  for rank in false_ranks]
    false_scores = [-1 if rank is None else top_score[rank] for rank in false_ranks]
    # Construct the true positive tuple
    NEW = True
    if NEW:
        def sort_tup(tup):
            # Sort tup by rank
            (aids, scores, ranks) = tup
            aids   = np.array(aids)
            scores = np.array(scores)
            ranks  = np.array(ranks)
            sortx  = scores.argsort()[::-1]
            sorted_aids   = aids[sortx].tolist()
            sorted_scores = scores[sortx].tolist()
            sorted_ranks  = ranks[sortx].tolist()
            sorted_tup = (sorted_aids, sorted_scores, sorted_ranks)
            return sorted_tup
        true_tup     = sort_tup((true_aids, true_scores, true_ranks))
        false_tup    = sort_tup((false_aids, false_scores, false_ranks))
    else:
        true_tup     = (true_aids, true_scores, true_ranks)
        false_tup    = (false_aids, false_scores, false_ranks)
    # Return tuples
    return true_tup, false_tup


def organize_results(ibs, qaid2_qres):
    """
    Sorts query result annotations, score, and ranks.


    TODO:
        somehow use the ratio of scores to try and make a prediction

    CommandLine:
        ib
        python dev.py -t scores --db PZ_MTEST --allgt -w --show
        python dev.py -t scores --db PZ_MTEST --allgt -w --show --cfg codename='vsone' fg_on:True --index 0:3
        python dev.py -t scores --db PZ_MTEST --allgt -w --show --cfg fg_on:True
        python dev.py -t scores --db GZ_ALL --allgt -w --show --cfg fg_on:True
        python dev.py -t scores --db GZ_ALL --allgt -w --show

    Example:
        >>> from ibeis.dev.results_organizer import *   # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> aid_list = ibs.get_valid_aids()
        >>> cfgdict = dict(codename='vsmany', fg_on=True)
        >>> qaid2_qres = ibs._query_chips4(aid_list, aid_list, cfgdict=cfgdict)
    """
    print('organize_results()')
    org_true          = OrganizedResult('true')
    org_false         = OrganizedResult('false')
    org_top_true      = OrganizedResult('top_true')   # highest ranked true matches
    org_top_false     = OrganizedResult('top_false')  # highest ranked false matches
    org_bot_true      = OrganizedResult('bot_true')
    org_problem_true  = OrganizedResult('problem_true')
    org_problem_false = OrganizedResult('problem_false')
    org_rank0_true    = OrganizedResult('rank0_true')
    org_rank0_false   = OrganizedResult('rank0_false')

    def _organize_result(qres):
        # Use ground truth to sort into true/false
        # * true_tup  = (true_aids,  true_scores,  true_ranks)
        # * false_tup = (false_aids, false_scores, false_ranks)
        true_tup, false_tup = qres2_true_and_false(ibs, qres)
        last_rank     = -1
        skipped_ranks = set([])
        #
        # Record: all_true, missed_true, top_true, bot_true
        topx = 0
        for topx, ttup in enumerate(zip(*true_tup)):
            (aid, score, rank) = ttup
            # Record all true results
            org_true.append(qaid, aid, rank, score)
            # Record non-top (a.k.a problem) true results
            if rank is None or last_rank is None or rank - last_rank > 1:
                if rank is not None:
                    skipped_ranks.add(rank - 1)
                org_problem_true.append(qaid, aid, rank, score)
            # Record the best results
            if topx == 0:
                org_top_true.append(qaid, aid, rank, score)
            if rank == 0:
                org_rank0_true.append(qaid, aid, rank, score)
            last_rank = rank
        # Record the worse true result
        if topx > 1:
            org_bot_true.append(qaid, aid, rank, score)
        #
        # Record the all_false, false_positive, top_false
        for topx, ftup in enumerate(zip(*false_tup)):
            (aid, score, rank) = ftup
            org_false.append(qaid, aid, rank, score)
            if rank in skipped_ranks:
                org_problem_false.append(qaid, aid, rank, score)
            if topx == 0:
                org_top_false.append(qaid, aid, rank, score)
            if rank == 0:
                org_rank0_false.append(qaid, aid, rank, score)

    # -----------------
    # Query result loop
    for qaid, qres in six.iteritems(qaid2_qres):
        if qres is not None:
            _organize_result(qres)
    #print('[rr2] len(org_true)          = %r' % len(org_true))
    #print('[rr2] len(org_false)         = %r' % len(org_false))
    #print('[rr2] len(org_top_true)      = %r' % len(org_top_true))
    #print('[rr2] len(org_top_false)     = %r' % len(org_top_false))
    #print('[rr2] len(org_bot_true)      = %r' % len(org_bot_true))
    #print('[rr2] len(org_problem_true)  = %r' % len(org_problem_true))
    #print('[rr2] len(org_problem_false) = %r' % len(org_problem_false))
    # qaid arrays for ttbttf
    allorg = dict([
        ('true',          org_true),
        ('false',         org_false),
        ('top_true',      org_top_true),
        ('top_false',     org_top_false),
        ('bot_true',      org_bot_true),
        ('problem_true',  org_problem_true),
        ('problem_false', org_problem_false),
        ('rank0_true',    org_rank0_true),
        ('rank0_false',   org_rank0_false),
    ])

    for org in six.itervalues(allorg):
        org.freeze()
    return allorg


def get_automatch_candidates(qaid2_qres, ranks_lt=5, directed=True,
                             name_scoring=False, ibs=None, filter_reviewed=False):
    """
    Returns a list of matches that should be inspected
    This function is more lightweight than orgres or allres.

    Used in inspect_gui and interact_qres2

    Args:
        qaid2_qres (dict): mapping from query annotaiton id to query result object
        ranks_lt (int): put all ranks less than this number into the graph
        directed (bool):

    Returns:
        tuple: candidate_matches = (qaid_arr, aid_arr, score_arr, rank_arr)

    Example0:
        >>> from ibeis.dev.results_organizer import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> qaid_list = ibs.get_valid_aids()[0:5]
        >>> daid_list = ibs.get_valid_aids()[0:20]
        >>> qaid2_qres = ibs._query_chips4(qaid_list, daid_list)
        >>> ranks_lt = 5
        >>> directed = True
        >>> name_scoring = False
        >>> candidate_matches = get_automatch_candidates(qaid2_qres, ranks_lt, directed)
        >>> print(candidate_matches)

    Example1:
        >>> from ibeis.dev.results_organizer import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> qaid_list = ibs.get_valid_aids()[0:5]
        >>> daid_list = ibs.get_valid_aids()[0:20]
        >>> qaid2_qres = ibs._query_chips4(qaid_list, daid_list)
        >>> ranks_lt = 5
        >>> directed = False
        >>> name_scoring = False
        >>> candidate_matches = get_automatch_candidates(qaid2_qres, ranks_lt, directed)
        >>> print(candidate_matches)
    """
    qaids_stack  = []
    aids_stack   = []
    ranks_stack  = []
    scores_stack = []

    # For each QueryResult, Extract inspectable candidate matches
    for qaid, qres in six.iteritems(qaid2_qres):
        assert qaid == qres.qaid, 'qaid2_qres and qres disagree on qaid'
        (qaids, aids, scores, ranks) = qres.get_match_tbldata(ranks_lt=ranks_lt,
                                                              name_scoring=name_scoring,
                                                              ibs=ibs)
        qaids_stack.append(qaids)
        aids_stack.append(aids)
        scores_stack.append(scores)
        ranks_stack.append(ranks)

    # Stack them into a giant array
    # utool.embed()
    qaid_arr  = np.hstack(qaids_stack)
    aid_arr   = np.hstack(aids_stack)
    score_arr = np.hstack(scores_stack)
    rank_arr  = np.hstack(ranks_stack)

    # Sort by scores
    sortx = score_arr.argsort()[::-1]
    qaid_arr  = qaid_arr[sortx]
    aid_arr   = aid_arr[sortx]
    score_arr = score_arr[sortx]
    rank_arr  = rank_arr[sortx]

    if filter_reviewed:
        is_reviewed = np.array(ibs.get_annot_pair_is_reviewed(qaid_arr.tolist(), aid_arr.tolist()))
        qaid_arr = qaid_arr.compress(is_reviewed)
        aid_arr = aid_arr.compress(is_reviewed)
        qaid_arr = score_arr.compress(is_reviewed)
        rank_arr = rank_arr.compress(is_reviewed)

    # Remove directed edges
    if not directed:
        #nodes = np.unique(directed_edges.flatten())
        directed_edges = np.vstack((qaid_arr, aid_arr)).T
        import vtool as vt
        idx1, idx2 = vt.intersect2d_indices(directed_edges, directed_edges[:, ::-1])

        def compute_edge_ids(edges):
            # construct a unique id for every edge
            ncols = edges.shape[1]
            # get the number of decimal places to shift
            exp_step = np.ceil(np.log10(edges.max()))
            offsets = [int(10 ** (ix * exp_step)) for ix in reversed(range(0, ncols))]
            edgeid_list = np.array([sum([e * offset for e, offset in zip(edge, offsets)]) for edge in edges])
            return edgeid_list

        def find_best_undirected_edge_indexes(directed_edges, score_arr):
            flipped = qaid_arr < aid_arr
            # standardize edge order
            edges_dupl = directed_edges.copy()
            edges_dupl[flipped, 0:2] = edges_dupl[flipped, 0:2][:, ::-1]
            edgeid_list = compute_edge_ids(edges_dupl)
            unique_edgeids, groupxs = vt.group_indices(edgeid_list)
            # if there is more than one edge in a group take the one with the highest score
            score_groups = vt.apply_grouping(score_arr, groupxs)
            unique_edge_xs = np.array(sorted([groupx[score_group.argmax()] for groupx, score_group in zip(groupxs, score_groups)]))
            return unique_edge_xs

        unique_rowx = find_best_undirected_edge_indexes(directed_edges, score_arr)

        OLD = False
        if OLD:
            flipped = qaid_arr < aid_arr
            # standardize edge order
            edges_dupl = directed_edges.copy()
            edges_dupl[flipped, 0:2] = edges_dupl[flipped, 0:2][:, ::-1]
            # Find unique row indexes
            unique_rowx = utool.unique_row_indexes(edges_dupl)
            #edges_unique = edges_dupl[unique_rowx]
            #flipped_unique = flipped[unique_rowx]
        qaid_arr  = qaid_arr.take(unique_rowx)
        aid_arr   = aid_arr.take(unique_rowx)
        score_arr = score_arr.take(unique_rowx)
        rank_arr  = rank_arr.take(unique_rowx)

    candidate_matches = (qaid_arr, aid_arr, score_arr, rank_arr)

    return candidate_matches
