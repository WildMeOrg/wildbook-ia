"""
module for making the correct automatic decisions in incremental tests
"""
from __future__ import absolute_import, division, print_function
import numpy as np
import utool as ut
print, print_, printDBG, rrr, profile = ut.inject(__name__, '[incoracle]')


@profile
def oracle_method1(ibs_gt, ibs, qnid1, aid_list2, aid2_to_aid1, sorted_nids, MAX_LOOK):
    """ METHOD 1: MAKE BEST DECISION FROM GIVEN INFORMATION """
    # Map annotations to ibs_gt annotation rowids
    sorted_nids = np.array(sorted_nids)
    aid_list1 = ut.dict_take_list(aid2_to_aid1, aid_list2)
    nid_list1 = np.array(ibs_gt.get_annot_name_rowids(aid_list1))
    # Using ibs_gt nameids find the correct index in returned results
    correct_rank = np.where(nid_list1 == qnid1)[0]
    correct_rank = correct_rank[correct_rank < MAX_LOOK]
    # Return a list of any number of correct names or empty if it is a new name
    nid_list2 = sorted_nids[correct_rank]
    chosen_names = ibs.get_name_texts(nid_list2)
    return chosen_names


@profile
def oracle_method2(ibs_gt, qnid1):
    """ METHOD 2: MAKE THE ABSOLUTE CORRECT DECISION REGARDLESS OF RESULT """
    # use the name from the groundruth database
    name2 = ibs_gt.get_name_texts(qnid1)
    chosen_names = [name2]
    return chosen_names


@profile
def get_oracle_name_decision(metatup, ibs, qaid, choicetup, oracle_method=1):
    """
    Find what the correct decision should be ibs is the database we are working
    with ibs_gt has pristine groundtruth
    """
    if ut.VERBOSE:
        print('Oracle is making decision using oracle_method=%r' % oracle_method)
    if metatup is None:
        print('WARNING METATUP IS NONE')
        return None
    MAX_LOOK = 3  # the oracle should only see what the user sees
    (sorted_nids, sorted_nscore, sorted_rawscore, sorted_aids, sorted_ascores) = choicetup
    (ibs_gt, aid1_to_aid2) = metatup
    # Get the annotations that the user can see
    aid_list2 = ut.get_list_column(sorted_aids, 0)
    # Get the groundtruth name of the query
    aid2_to_aid1 = ut.invert_dict(aid1_to_aid2)
    qnid1 = ibs_gt.get_annot_name_rowids(aid2_to_aid1[qaid])
    # Make an oracle decision by choosing a name (like a user would)
    if oracle_method == 1:
        chosen_names = oracle_method1(ibs_gt, ibs, qnid1, aid_list2, aid2_to_aid1, sorted_nids, MAX_LOOK)
    elif oracle_method == 2:
        chosen_names = oracle_method2(ibs_gt, qnid1)
    else:
        raise AssertionError('unknown oracle method %r' % (oracle_method,))
    if ut.VERBOSE:
        print('Oracle decision is chosen_names=%r' % (chosen_names,))
    return chosen_names


@profile
def get_oracle_name_suggestion(ibs, qaid, choicetup,  metatup):
    """
    main entry point for the oracle
    """
    #system_autoname_msg = system_name_suggest_tup[0]
    (sorted_nids, sorted_nscore, sorted_rawscore, sorted_aids, sorted_ascores) = choicetup
    oracle_msg_list = []
    #oracle_msg_list.append('The overrided system responce was:\n%s'
    #                       % (ut.indent(system_autoname_msg, '  ~~'),))
    chosen_names = get_oracle_name_decision(metatup, ibs, qaid, choicetup)

    if len(chosen_names) == 0:
        oracle_msg_list.append('Oracle suggests a new name')
        # be confident if suggesting a new name
        name_confidence = 1.0
    else:
        #name_confidence = 0.99  # The oracle is confident in its decision
        name_confidence = 1.0  # The oracle is confident in its decision
        oracle_msg_list.append(
            'Oracle suggests chosen_names=%r' % (chosen_names,))
    autoname_msg = '\n'.join(oracle_msg_list)
    oracle_name_suggest_tup = autoname_msg, chosen_names, name_confidence
    return oracle_name_suggest_tup


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.model.hots.automated_oracle
        python -m ibeis.model.hots.automated_oracle --allexamples
        python -m ibeis.model.hots.automated_oracle --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
