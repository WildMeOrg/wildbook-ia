"""
CommandLine:
    python -c "import utool as ut; ut.write_modscript_alias('Tinc.sh', 'ibeis.model.hots.qt_inc_automatch')"

    sh Tinc.sh --test-test_inc_query:0
    sh Tinc.sh --test-test_inc_query:1
    sh Tinc.sh --test-test_inc_query:2
    sh Tinc.sh --test-test_inc_query:3 --num-initial 5000

    python -m ibeis.model.hots.qt_inc_automatch --test-test_inc_query:0

"""
from __future__ import absolute_import, division, print_function
import six
import utool as ut
from ibeis import constants as const
from ibeis.model.hots import automated_oracle as ao
from ibeis.model.hots import automated_helpers as ah
from ibeis.model.hots import automated_params
from ibeis.model.hots import special_query
from ibeis.model.hots import neighbor_index
from ibeis.model.hots import automatch_suggestor
from ibeis.model.hots import user_dialogs
from collections import namedtuple
ut.noinject(__name__, '[inc]')
#profile = ut.profile
print, print_, printDBG, rrr, profile = ut.inject(__name__, '[inc]')

Metatup = namedtuple('Metatup', ('ibs_gt', 'aid1_to_aid2'))


def testdata_automatch(dbname=None):
    if dbname is None:
        dbname = 'testdb1'
    import ibeis
    ibs = ibeis.opendb('testdb1')
    qaid_chunk = ibs.get_valid_aids()[0:1]
    return ibs, qaid_chunk


# ---- ENTRY POINT ----


@profile
def test_generate_incremental_queries(ibs_gt, ibs, aid_list1, aid1_to_aid2,
                                      num_initial=0, incinfo=None):
    """
    TODO: move this somewhere else
    Testing function

    Adds and queries new annotations one at a time with oracle guidance
    ibs1 is ibs_gt, ibs2 is ibs
    """
    print('begin test interactive iter')
    # Transfer some amount of initial data
    print('Transfer %d initial test annotations' % (num_initial,))
    if num_initial > 0:
        aid_sublist1 = aid_list1[0:num_initial]
        aid_sublist2 = ut.dict_take_list(aid1_to_aid2, aid_sublist1)
        #aid_sublist2 = ah.add_annot_chunk(ibs_gt, ibs, aid_sublist1, aid1_to_aid2)
        # Add names from old databse. add all initial as exemplars
        name_list = ibs_gt.get_annot_names(aid_sublist1)
        ibs.set_annot_names(aid_sublist2, name_list)
        ibs.set_annot_exemplar_flags(aid_sublist2, [True] * len(aid_sublist2))
        aid_list1 = aid_list1[num_initial:]
    # Print info
    WITHINFO = ut.get_argflag('--withinfo')
    if WITHINFO:
        print('+-------')
        print('Printing ibs_gt and ibs info before start')
        print('--------')
        print('\nibs info:')
        print(ibs.get_dbinfo_str())
        print('--------')
        print('\nibs_gt info')
        #print(ibs_gt.get_dbinfo_str())
        print('L________')
    # Setup metadata tuple (in the presence of groundtruth, keeps records of
    # accuracy and give information to the oracle decision maker)
    metatup = Metatup(ibs_gt, aid1_to_aid2)
    assert incinfo is not None
    incinfo['metatup'] = metatup
    incinfo['interactive'] = False
    # Begin incremental iteration
    chunksize = 1
    aids_chunk1_iter = ut.progress_chunks(aid_list1, chunksize, lbl='TEST QUERY')
    for count, aids_chunk1 in enumerate(aids_chunk1_iter):
        with ut.Timer('teststep'):
            #sys.stdout.write('\n')
            print('\n==== EXECUTING TESTSTEP %d ====' % (count,))
            print('generator_stack_depth = %r' % ut.get_current_stack_depth())
            #incinfo['interactive'] = (interact_after is not None and count >= interact_after)
            #---
            # ensure new annot is added (most likely it will have been preadded)
            #qaid_chunk = ah.add_annot_chunk(ibs_gt, ibs, aids_chunk1, aid1_to_aid2)
            #---
            # Assume annot has alredy been added
            # Get mapping
            qaid_chunk = ut.dict_take_list(aid1_to_aid2, aids_chunk1)
            #---
            for item in generate_subquery_steps(ibs, qaid_chunk, incinfo=incinfo):
                yield item
                #(ibs, qres, qreq_, incinfo) = item
                # Yeild results for qt interface to call down into user or
                # oracle code and make a decision
    print('ending interactive iter')
    ah.check_results(ibs_gt, ibs, aid1_to_aid2, aid_list1, incinfo)


@profile
def generate_incremental_queries(ibs, qaid_list, incinfo=None):
    r"""
    qt entry point. generates query results for the qt harness to process.

    Args:
        ibs (IBEISController):  ibeis controller object
        qaid_list (list):

    CommandLine:
        python -m ibeis.model.hots.automated_matcher --test-generate_incremental_queries

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.automated_matcher import *  # NOQA
        >>> ibs, qaid_chunk = testdata_automatch()
        >>> generate_incremental_queries(ibs, qaid_list)
    """
    # Execute each query as a test
    chunksize = 1
    #aids_chunk1_iter = ut.ichunks(aid_list1, chunksize)
    qaid_chunk_iter = ut.progress_chunks(qaid_list, chunksize, lbl='TEST QUERY')

    ibs = ibs
    for count, qaid_chunk in enumerate(qaid_chunk_iter):
        #sys.stdout.write('\n')
        print('\n==== EXECUTING QUERY %d ====' % (count,))
        print('generator_stack_depth = %r' % ut.get_current_stack_depth())
        for item in generate_subquery_steps(ibs, qaid_chunk, incinfo=incinfo):
            yield item

# ---- INIT ----


def initialize_persistant_query_request(ibs, qaid_chunk):
    # FIXME: allow for multiple species or make a nicer way of ensuring that
    # there is only one species here
    species_text_set = set(ibs.get_annot_species_texts(qaid_chunk))
    assert len(species_text_set) == 1, 'query chunk has more than one species'
    species_text = list(species_text_set)[0]
    # controller based exemplars
    daid_list = ibs.get_valid_aids(is_exemplar=True, species=species_text, is_known=True, nojunk=True)
    num_names = len(set(ibs.get_annot_nids(daid_list)))
    # TODO: choose vsmany K every time
    # need to be able to update qreq_.qparams
    vsmany_K = automated_params.choose_vsmany_K(num_names, qaid_chunk, daid_list)
    vsmany_cfgdict = dict(
        K=vsmany_K,
        Knorm=3,
        index_method='multi',
        pipeline_root='vsmany',
        #return_expanded_nns=False,
        #return_expanded_nns=True,
    )
    # Create New qreq
    qreq_vsmany_ = ibs.new_query_request(qaid_chunk, daid_list, cfgdict=vsmany_cfgdict)
    return qreq_vsmany_


def load_or_make_qreq(ibs, qreq_vsmany_, qaid_chunk):
    if qreq_vsmany_ is None:
        qreq_vsmany_ = initialize_persistant_query_request(ibs, qaid_chunk)
    else:
        # set new query aids
        qreq_vsmany_.set_internal_qaids(qaid_chunk)
        # state based exemplars
        # daid_list = qreq_vsmany_.get_external_daids()
        # Force indexer reloading if background process is completed we might
        # get a shiny new indexer.
        force = neighbor_index.check_background_process()
        qreq_vsmany_.load_indexer(force=force)
    return qreq_vsmany_

# ---- QUERY ----


def execute_query_batch(ibs, qaid_chunk, qreq_vsmany_, incinfo):
    """
    TODO: remove special query
    """
    from ibeis.model.hots import match_chips4 as mc4
    USE_SPECIAL_QUERY = False
    if USE_SPECIAL_QUERY:
        qaid2_qres, qreq_, qreq_vsmany_ = special_query.query_vsone_verified(
            ibs, qaid_chunk, qreq_vsmany_.get_external_daids(), qreq_vsmany__=qreq_vsmany_, incinfo=incinfo)
    else:
        qreq_ = qreq_vsmany_
        qaid2_qres = mc4.submit_query_request_nocache(ibs,  qreq_=qreq_)
    if incinfo.get('qreq_vsmany_', None) is None:
        incinfo['qreq_vsmany_'] = qreq_vsmany_
    else:
        assert incinfo.get('qreq_vsmany_') is qreq_vsmany_, 'bad statefulness'
    return qaid2_qres, qreq_


@profile
def generate_subquery_steps(ibs, qaid_chunk, incinfo=None):
    """
    Generats query results for the qt harness to then send into the next
    decision steps.

    Args:
        ibs (IBEISController):  ibeis controller object
        qaid_chunk (?):
        incinfo (dict):

    CommandLine:
        python -m ibeis.model.hots.automated_matcher --test-generate_subquery_steps

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.automated_matcher import *  # NOQA
        >>> ibs, qaid_chunk = testdata_automatch()
        >>> generate_subquery_steps(ibs, qaid_chunk)
    """
    # Use either state-based exemplars or controller based exemplars
    qreq_vsmany_ = incinfo.get('qreq_vsmany_', None)
    # Execute actual queries
    qreq_vsmany_ = load_or_make_qreq(ibs, qreq_vsmany_, qaid_chunk)
    qaid2_qres, qreq_ = execute_query_batch(ibs, qaid_chunk, qreq_vsmany_, incinfo)

    #try_decision_callback = incinfo.get('try_decision_callback', None)
    for qaid, qres in six.iteritems(qaid2_qres):
        yield [ibs, qres, qreq_, incinfo]

# ---- DECISION ---


def get_name_suggestion(ibs, qaid, choicetup, incinfo):
    # ---------------------------------------------
    # Get oracle suggestion if we have the metadata
    # override the system suggestion
    metatup = incinfo.get('metatup', None)
    if incinfo.get('use_oracle', False) and metatup is not None:
        oracle_name_suggest_tup = ao.get_oracle_name_suggestion(
            ibs, qaid, choicetup, metatup)
        name_suggest_tup = oracle_name_suggest_tup
    else:
        # Get system suggested name
        system_name_suggest_tup = automatch_suggestor.get_system_name_suggestion(ibs, choicetup)
        name_suggest_tup = system_name_suggest_tup
    return name_suggest_tup


@profile
def run_until_name_decision_signal(ibs, qres, qreq_, incinfo=None):
    r"""
    DECISION STEP 1)

    Either the system or the user makes a decision about the name of the query
    annotation.

    CommandLine:
        python -m ibeis.model.hots.automated_matcher --test-run_until_name_decision_signal

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.automated_matcher import *  # NOQA
        >>> ibs, qaid_chunk = testdata_automatch()
        >>> exemplar_aids = ibs.get_valid_aids(is_exemplar=True)
        >>> incinfo = {}
        >>> gen = generate_subquery_steps(ibs, qaid_chunk, incinfo)
        >>> item = six.next(gen)
        >>> ibs, qres, qreq_, incinfo = item
        >>> # verify results
        >>> run_until_name_decision_signal(ibs, qres, qreq_, incinfo)

    Ignore::
        qres.ishow_top(ibs, sidebyside=False, show_query=True)
    """
    print('--- Identifying Query Animal ---')
    qaid = qres.get_qaid()
    choicetup = automatch_suggestor.get_qres_name_choices(ibs, qres)
    name_suggest_tup = get_name_suggestion(ibs, qaid, choicetup, incinfo)
    # Have the system ask the user if it is not confident in its decision
    autoname_msg, chosen_names, name_confidence = name_suggest_tup
    print('autoname_msg=')
    print(autoname_msg)
    print('... checking confidence=%r in name decision.' % (name_confidence,))
    if name_confidence < incinfo.get('name_confidence_thresh', 1.0):
        print('... confidence is too low. need user input')
        if incinfo.get('interactive', False):
            print('... asking user for input')
            if qreq_.normalizer is not None:
                VIZ_SCORE_NORM = False
                #VIZ_SCORE_NORM = ut.is_developer()
                if VIZ_SCORE_NORM:
                    qreq_.normalizer.visualize(fnum=511, verbose=False)
            user_dialogs.wait_for_user_name_decision(ibs, qres, qreq_, choicetup,
                                                     name_suggest_tup,
                                                     incinfo=incinfo)
        else:
            run_until_finish(incinfo=incinfo)
            print('... cannot ask user for input. doing nothing')
    else:
        print('... confidence is above threshold. Making decision')
        #return ('CALLBACK', chosen_names)
        name_decision_callback = incinfo['name_decision_callback']
        name_decision_callback(chosen_names)
        #exec_name_decision_and_continue(chosen_names, ibs, qres, qreq_, incinfo=incinfo)


@profile
def exec_name_decision_and_continue(chosen_names, ibs, qres, qreq_,
                                    incinfo=None):
    """
    DECISION STEP 2)

    The name decision from the previous step is executed and the score
    normalizer is updated. Then execution continues to the exemplar decision
    step.
    """
    print('--- Updating Exemplars ---')
    qaid = qres.get_qaid()
    #assert ibs.is_aid_unknown(qaid), 'animal is already known'
    try_update_normalizer = False
    was_junk = False
    if chosen_names is None or len(chosen_names) == 0 or chosen_names == 'newname':
        newname = ibs.make_next_name()
        print('identifying qaid=%r as a new animal. newname=%r' % (qaid, newname))
        ibs.set_annot_names((qaid,), (newname,))
    elif  chosen_names == 'junk':
        print('setting query qaid=%r as a junk annotations')
        ibs.set_annot_qualities((qaid,), (const.QUALITY_TEXT_TO_INT['junk'],))
        was_junk = True
    elif len(chosen_names) == 1:
        print('identifying qaid=%r as name=%r' % (qaid, chosen_names,))
        ibs.set_annot_names((qaid,), chosen_names)
        try_update_normalizer = True
    elif len(chosen_names) > 1:
        merge_name = chosen_names[0]
        other_names = chosen_names[1:]
        print('identifying qaid=%r as name=%r and merging with %r ' %
                (qaid, merge_name, other_names))
        ibs.merge_names(merge_name, other_names)
        ibs.set_annot_names((qaid,), (merge_name,))
    # STATE MAINTENANCE
    # TODO make sure update normalizer works
    if qreq_.normalizer is not None and try_update_normalizer:
        update_normalizer(ibs, qres, qreq_, chosen_names)
    # Do update callback so the name updates in the main GUI
    interactive = incinfo.get('interactive', False)
    update_callback = incinfo.get('update_callback', None)
    if interactive and not was_junk and update_callback is not None:
        # Update callback repopulates the names tree
        update_callback()
    run_until_exemplar_decision_signal(ibs, qres, qreq_, incinfo=incinfo)


@profile
def run_until_exemplar_decision_signal(ibs, qres, qreq_, incinfo=None):
    """
    DECISION STEP 3)

    Either the system or the user decides if the query should be added to the
    database as an exemplar.
    """
    qaid = qres.get_qaid()
    exemplar_confidence_thresh = ut.get_sys_maxfloat()
    exmplr_suggestion = automatch_suggestor.get_system_exemplar_suggestion(ibs, qaid)
    (autoexemplar_msg, exemplar_decision, exemplar_condience) = exmplr_suggestion
    print('autoexemplar_msg=')
    print(autoexemplar_msg)
    # HACK: Disable asking users a about exemplars
    need_user_input = False and (incinfo.get('interactive', False) and exemplar_condience < exemplar_confidence_thresh)
    if need_user_input:
        user_dialogs.wait_for_user_exemplar_decision(
            autoexemplar_msg, exemplar_decision, exemplar_condience)
    else:
        # May need to execute callback whereas whatever the interaction was
        # would issue it otherwise
        exec_exemplar_decision_and_continue(exemplar_decision, ibs, qres, qreq_, incinfo=incinfo)


@profile
def exec_exemplar_decision_and_continue(exemplar_decision, ibs, qres, qreq_,
                                        incinfo=None):
    """
    DECISION STEP 4)

    The exemplar decision in the previous step is executed.  The persistant
    vsmany query request is updated if needbe and the execution continues.
    (currently to the end of this iteration)
    """
    #qaid = qres.get_qaid()
    new_aids, remove_aids = exemplar_decision
    #if exemplar_decision:
    if len(new_aids) > 0:
        ibs.set_annot_exemplar_flags(new_aids, [True] * len(new_aids))
    if len(remove_aids) > 0:
        ibs.set_annot_exemplar_flags(remove_aids, [False] * len(remove_aids))
    if 'qreq_vsmany_' in incinfo:
        qreq_vsmany_ = incinfo.get('qreq_vsmany_')
        # STATE_MAINTENANCE
        # Add new query as a database annotation
        if len(new_aids) > 0:
            qreq_vsmany_.add_internal_daids(new_aids)
        if len(remove_aids) > 0:
            qreq_vsmany_.remove_internal_daids(remove_aids)
    run_until_finish(incinfo=incinfo)


@profile
def run_until_finish(incinfo=None):
    """
    DECISION STEP 5)

    """
    if incinfo is not None:
        # This query run as eneded
        next_query_key = 'next_query_callback'
        if next_query_key in incinfo:
            incinfo[next_query_key]()
        else:
            print('Warning: no next_query_key=%r' % (next_query_key,))


# ---- POST DECISION ---


@profile
def update_normalizer(ibs, qres, qreq_, chosen_names):
    r"""
    adds new support data to the current normalizer

    FIXME: a miss-save in vim will trigger module unloading

    Args:
        ibs (IBEISController):  ibeis controller object
        qreq_ (QueryRequest):  query request object with hyper-parameters
        choicetup (?):
        name (?):

    Returns:
        tuple: (tp_rawscore, tn_rawscore)

    CommandLine:
        python -m ibeis.model.hots.automated_matcher --test-update_normalizer

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.automated_matcher import *  # NOQA
        >>> ibs, qaid_chunk = testdata_automatch()
        >>> exemplar_aids = ibs.get_valid_aids(is_exemplar=True)
        >>> incinfo = {}
        >>> gen = generate_subquery_steps(ibs, qaid_chunk, incinfo)
        >>> item = six.next(gen)
        >>> ibs, qres, qreq_, incinfo = item
        >>> qreq_.load_score_normalizer()
        >>> # verify results
        >>> chosen_names = ['easy']
        >>> update_normalizer(ibs, qres, qreq_, chosen_names)
    """
    # Fixme: duplicate call to get_qres_name_choices
    if qreq_.normalizer is None:
        print('[update_normalizer] NOT UPDATING. qreq_ has not loaded a score normalizer')
        return
    if len(chosen_names) != 1:
        print('[update_normalizer] NOT UPDATING. only updates using simple matches')
        return
    qaid = qres.get_qaid()
    choicetup = automatch_suggestor.get_qres_name_choices(ibs, qres)
    (sorted_nids, sorted_nscore, sorted_rawscore, sorted_aids, sorted_ascores) = choicetup
    # Get new True Negative support data for score normalization
    name = chosen_names[0]
    rank = ut.listfind(ibs.get_name_texts(sorted_nids), name)
    if rank is None:
        return
    nid = sorted_nids[rank]
    tp_rawscore = sorted_rawscore[rank]
    valid_falseranks = set(range(len(sorted_rawscore))) - set([rank])
    if len(valid_falseranks) > 0:
        tn_rank = min(valid_falseranks)
        tn_rawscore = sorted_rawscore[tn_rank][0]
    else:
        tn_rawscore = None
    #return tp_rawscore, tn_rawscore
    canupdate = tp_rawscore is not None and tn_rawscore is not None
    if canupdate:
        # TODO: UPDATE SCORE NORMALIZER HERE
        print('UPDATING! NORMALIZER')
        tp_labels = [ut.deterministic_uuid((qaid, nid))]
        tn_labels = [ut.deterministic_uuid((qaid, nid))]
        print('new normalization example: tp_rawscore={}, tn_rawscore={}'.format(tp_rawscore, tn_rawscore))
        print('new normalization example: tp_labels={}, tn_labels={}'.format(tp_labels, tn_labels))
        tp_scores = [tp_rawscore]
        tn_scores = [tn_rawscore]
        qreq_.normalizer.add_support(tp_scores, tn_scores, tp_labels, tn_labels)
        qreq_.normalizer.retrain()
        species_text = '_'.join(qreq_.get_unique_species())  # HACK
        # TODO: figure out where to save and load the normalizer from
        qreq_.normalizer.save(ibs.get_local_species_scorenorm_cachedir(species_text))
    else:
        print('NOUPDATE! cannot update score normalization')


# --- TESTING ---


if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.model.hots.automated_matcher
        python -m ibeis.model.hots.automated_matcher --allexamples
        python -m ibeis.model.hots.automated_matcher --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    #import utool as ut
    ut.doctest_funcs()
