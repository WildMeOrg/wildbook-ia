from __future__ import absolute_import, division, print_function
# Python
from itertools import izip
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
    Maintains an organized list of:
        * query roi indexes
        * their top matching result
        * their score
        * their rank
    What chips are populated depends on the type of organization
    """
    def __init__(self, orgtype=''):
        super(DynStruct, self).__init__()
        self.orgtype = orgtype
        self.qrids   = []  # query roi indexes
        self.rids    = []  # their top matching result
        self.scores  = []  # their score
        self.ranks   = []  # their rank

    def append(self, qrid, rid, rank, score):
        self.qrids.append(qrid)
        self.rids.append(rid)
        self.scores.append(score)
        self.ranks.append(rank)

    def freeze(self):
        """ No more appending """
        self.qrids  = np.array(self.qrids)
        self.rids   = np.array(self.rids)
        self.scores = np.array(self.scores)
        self.ranks  = np.array(self.ranks)

    def __len__(self):
        num_qcxs   = len(self.qrids)
        num_rids   = len(self.rids)
        num_scores = len(self.scores)
        num_ranks  = len(self.ranks)
        assert num_qcxs == num_rids
        assert num_rids == num_scores
        assert num_scores == num_ranks
        return num_qcxs

    def iter(self):
        """ useful for plotting """
        result_iter = izip(self.qrids, self.rids, self.scores, self.ranks)
        for qrid, rid, score, rank in result_iter:
            yield qrid, rid, score, rank

    def printme3(self):
        column_list = [self.qrids, self.rids, self.scores, self.ranks]
        column_labels = ['qrids', 'rids', 'scores', 'ranks']
        header = 'Orgres %s' % (self.orgtype)
        print(utool.make_csv_table(column_labels, column_list, header, column_type=None))


def qres2_true_and_false(ibs, qres):
    """
    Organizes results into:
        true positive set
        and
        false positive set
    a set is a query, its best match, and a score
    """
    # Get top chip indexes and scores
    top_rids  = qres.get_top_rids()
    top_score = qres.get_rid_scores(top_rids)
    top_ranks = range(len(top_rids))
    # True Rids / Scores / Ranks
    true_ranks, true_rids = qres.get_gt_ranks(ibs=ibs, return_gtrids=True)
    true_scores  = [-1 if rank is None else top_score[rank] for rank in true_ranks]
    # False Rids / Scores / Ranks
    false_ranks = list(set(top_ranks) - set(true_ranks))
    false_rids   = [-1 if rank is None else top_rids[rank]  for rank in false_ranks]
    false_scores = [-1 if rank is None else top_score[rank] for rank in false_ranks]
    # Construct the true positive tuple
    true_tup     = (true_rids, true_scores, true_ranks)
    false_tup    = (false_rids, false_scores, false_ranks)
    # Return tuples
    return true_tup, false_tup


def organize_results(ibs, qrid2_qres):
    print('organize_results()')
    org_true          = OrganizedResult('true')
    org_false         = OrganizedResult('false')
    org_top_true      = OrganizedResult('top_true')
    org_top_false     = OrganizedResult('top_false')
    org_bot_true      = OrganizedResult('bot_true')
    org_problem_true  = OrganizedResult('problem_true')
    org_problem_false = OrganizedResult('problem_false')
    # -----------------
    # Query result loop

    def _organize_result(qres):
        # Use ground truth to sort into true/false
        true_tup, false_tup = qres2_true_and_false(ibs, qres)
        last_rank     = -1
        skipped_ranks = set([])
        #
        # Record: all_true, missed_true, top_true, bot_true
        topx = 0
        for topx, (rid, score, rank) in enumerate(izip(*true_tup)):
            # Record all true results
            org_true.append(qrid, rid, rank, score)
            # Record non-top (a.k.a problem) true results
            if rank is None or last_rank is None or rank - last_rank > 1:
                if rank is not None:
                    skipped_ranks.add(rank - 1)
                org_problem_true.append(qrid, rid, rank, score)
            # Record the best results
            if topx == 0:
                org_top_true.append(qrid, rid, rank, score)
            last_rank = rank
        # Record the worse true result
        if topx > 1:
            org_bot_true.append(qrid, rid, rank, score)
        #
        # Record the all_false, false_positive, top_false
        topx = 0
        for rid, score, rank in zip(*false_tup):
            org_false.append(qrid, rid, rank, score)
            if rank in skipped_ranks:
                org_problem_false.append(qrid, rid, rank, score)
            if topx == 0:
                org_top_false.append(qrid, rid, rank, score)
            topx += 1

    for qrid, qres in qrid2_qres.iteritems():
        if qres is not None:
            _organize_result(qres)
    #print('[rr2] len(org_true)          = %r' % len(org_true))
    #print('[rr2] len(org_false)         = %r' % len(org_false))
    #print('[rr2] len(org_top_true)      = %r' % len(org_top_true))
    #print('[rr2] len(org_top_false)     = %r' % len(org_top_false))
    #print('[rr2] len(org_bot_true)      = %r' % len(org_bot_true))
    #print('[rr2] len(org_problem_true)  = %r' % len(org_problem_true))
    #print('[rr2] len(org_problem_false) = %r' % len(org_problem_false))
    # qrid arrays for ttbttf
    allorg = dict([
        ('true',          org_true),
        ('false',         org_false),
        ('top_true',      org_top_true),
        ('top_false',     org_top_false),
        ('bot_true',      org_bot_true),
        ('problem_true',  org_problem_true),
        ('problem_false', org_problem_false),
    ])

    for org in allorg.itervalues():
        org.freeze()
    return allorg
