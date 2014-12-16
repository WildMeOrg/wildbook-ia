from __future__ import absolute_import, division, print_function
import utool as ut
print, print_, printDBG, rrr, profile = ut.inject(__name__, '[incoracle]')


def oracle_method1(ibs_gt, ibs2, qnid1, aid_list2, aid2_to_aid1, sorted_nids, MAX_LOOK):
    """ METHOD 1: MAKE BEST DECISION FROM GIVEN INFORMATION """
    # Map annotations to ibs_gt annotation rowids
    aid_list1 = ut.dict_take_list(aid2_to_aid1, aid_list2)
    nid_list1 = ibs_gt.get_annot_name_rowids(aid_list1)
    # Using ibs_gt nameids find the correct index in returned results
    correct_rank = ut.listfind(nid_list1, qnid1)
    if correct_rank is None or correct_rank >= MAX_LOOK:
        # If the correct result was not presented create a new name
        name2 = None
    else:
        # Otherwise return the correct result
        nid2 = sorted_nids[correct_rank]
        name2 = ibs2.get_name_texts(nid2)
    return name2


def oracle_method2(ibs_gt, qnid1):
    """ METHOD 2: MAKE THE ABSOLUTE CORRECT DECISION REGARDLESS OF RESULT """
    name2 = ibs_gt.get_name_texts(qnid1)
    return name2


def get_oracle_decision(metatup, ibs2, qaid, choicetup, oracle_method=1):
    """
    Find what the correct decision should be ibs2 is the database we are working
    with ibs_gt has pristine groundtruth
    """
    print('Oracle is making decision using oracle_method=%r' % oracle_method)
    if metatup is None:
        print('WARNING METATUP IS NONE')
        return None
    MAX_LOOK = 3  # the oracle should only see what the user sees
    (sorted_nids, sorted_nscore, sorted_rawscore, sorted_aids, sorted_ascores) = choicetup
    (ibs_gt, aid1_to_aid2) = metatup

    #ut.embed()
    # Get the annotations that the user can see
    aid_list2 = ut.get_list_column(sorted_aids, 0)
    # Get name rowids of the query from ibs_gt
    aid2_to_aid1 = ut.invert_dict(aid1_to_aid2)
    qannot_rowid1 = aid2_to_aid1[qaid]
    qnid1 = ibs_gt.get_annot_name_rowids(qannot_rowid1)
    # Make an oracle decision by choosing a name (like a user would)
    if oracle_method == 1:
        name2 = oracle_method1(ibs_gt, ibs2, qnid1, aid_list2, aid2_to_aid1, sorted_nids, MAX_LOOK)
    elif oracle_method == 2:
        name2 = oracle_method2(ibs_gt, qnid1)
    else:
        raise AssertionError('unknown oracle method %r' % (oracle_method,))
    print('Oracle decision is name2=%r' % (name2,))
    return name2


def get_oracle_name_suggestion(ibs2, autoname_msg, qaid, choicetup,  metatup):
    (sorted_nids, sorted_nscore, sorted_rawscore, sorted_aids, sorted_ascores) = choicetup
    oracle_msg_list = []
    oracle_msg_list.append('The overrided system responce was:\n%s'
                           % (ut.indent(autoname_msg, '  ~~'),))
    name = get_oracle_decision(metatup, ibs2, qaid, choicetup)

    if name is None:
        oracle_msg_list.append('Oracle suggests a new name')
    else:
        nid = ibs2.get_name_rowids_from_text(name)
        rank = ut.listfind(sorted_nids.tolist(), nid)
        if rank is None:
            print('Warning: impossible state if oracle_method == 1')
            score, rawscore = None
        else:
            score = sorted_nscore[rank]
            rawscore = sorted_rawscore[rank][0]
        oracle_msg_list.append(
            'Oracle suggests nid=%r, score=%.2f, rank=%r, rawscore=%.2f' % (nid, score, rank, rawscore))
    autoname_msg = '\n'.join(oracle_msg_list)
    name_confidence = 1.0
    return autoname_msg, name, name_confidence
    #print(autoname_msg)
    #ut.embed()
