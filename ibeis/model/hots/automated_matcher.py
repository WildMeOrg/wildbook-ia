from __future__ import absolute_import, division, print_function
import six
import utool as ut
from ibeis.model.hots import automated_oracle as ao
from ibeis.model.hots import automated_helpers as ah
from ibeis.model.hots import special_query
from ibeis.model.hots import system_suggestor
from ibeis.model.hots import user_dialogs
ut.noinject(__name__, '[inc]')
#print, print_, printDBG, rrr, profile = ut.inject(__name__, '[inc]')


# ---- ENTRY POINT ----


def test_generate_incremental_queries(ibs_gt, ibs, aid_list1, aid1_to_aid2, incinfo=None):
    """
    Adds and queries new annotations one at a time with oracle guidance


    CommandLine:
        python -c "import utool as ut; ut.write_modscript_alias('Tinc.sh', 'ibeis.model.hots.interactive_automated_matcher')"
        sh Tinc.sh --test-test_interactive_incremental_queries:0
        sh Tinc.sh --test-test_interactive_incremental_queries:1
        sh Tinc.sh --test-test_interactive_incremental_queries:2

    """
    print('begin test interactive iter')
    #interact_after = 100
    #interact_after = None
    #interact_after = ut.get_argval(('--interactive-after', '--interact-after',),
    #                               type_=int, default=0)
    # Execute each query as a test
    chunksize = 1
    #aids_chunk1_iter = ut.ichunks(aid_list1, chunksize)
    # Query aids in a random order
    #shuffled_aids_list1 = ut.deterministic_shuffle(aid_list1[:])
    #aids_chunk1_iter = ut.progress_chunks(shuffled_aids_list1, chunksize, lbl='TEST QUERY')
    aids_chunk1_iter = ut.progress_chunks(aid_list1, chunksize, lbl='TEST QUERY')
    metatup = (ibs_gt, aid1_to_aid2)
    assert incinfo is not None
    incinfo['metatup'] = metatup
    incinfo['interactive'] = False

    for count, aids_chunk1 in enumerate(aids_chunk1_iter):
        with ut.Timer('teststep'):
            #sys.stdout.write('\n')
            print('\n==== EXECUTING TESTSTEP %d ====' % (count,))
            print('generator_stack_depth = %r' % ut.get_current_stack_depth())
            #if ut.get_current_stack_depth() > 53:
            #    ut.embed()
            #incinfo['interactive'] = (interact_after is not None and count >= interact_after)
            # ensure new annot is added (most likely it will have been preadded)
            qaid_chunk = ah.add_annot_chunk(ibs_gt, ibs, aids_chunk1, aid1_to_aid2)
            for item in generate_subquery_steps(ibs, qaid_chunk, incinfo=incinfo):
                (ibs, qres, qreq_, incinfo) = item
                yield item
    print('ending interactive iter')
    ah.check_results(ibs_gt, ibs, aid1_to_aid2)


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
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
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


# ---- QUERY ----

def generate_subquery_steps(ibs, qaid_chunk, incinfo=None):
    """
    Args:
        ibs (IBEISController):  ibeis controller object
        qaid_chunk (?):
        incinfo (dict):

    CommandLine:
        python -m ibeis.model.hots.automated_matcher --test-generate_subquery_steps

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.automated_matcher import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> qaid_chunk = [1]
        >>> generate_subquery_steps(ibs, qaid_chunk)
    """
    species_text_set = set(ibs.get_annot_species_texts(qaid_chunk))
    assert len(species_text_set) == 1, 'query chunk has more than one species'
    species_text = list(species_text_set)[0]
    daid_list = ibs.get_valid_aids(is_exemplar=True, species=species_text)
    # Execute actual queries
    qaid2_qres, qreq_ = special_query.query_vsone_verified(ibs, qaid_chunk, daid_list)
    #try_decision_callback = incinfo.get('try_decision_callback', None)
    for qaid, qres in six.iteritems(qaid2_qres):
        item = [ibs, qres, qreq_, incinfo]
        yield item

# ---- DECISION ---


def run_until_name_decision_signal(ibs, qres, qreq_, incinfo=None):
    r"""
    Either makes automatic decision or asks user for feedback.

    CommandLine:
        python -m ibeis.model.hots.automated_matcher --test-test_generate_incremental_queries:0
        python -m ibeis.model.hots.automated_matcher --test-test_generate_incremental_queries:1
        python -m ibeis.model.hots.automated_matcher --test-run_until_name_decision_signal

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.automated_matcher import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> qaid_chunk = ibs.get_valid_aids()[0:1]
        >>> exemplar_aids = ibs.get_valid_aids(is_exemplar=True)
        >>> incinfo = {}
        >>> gen = generate_subquery_steps(ibs, qaid_chunk, incinfo)
        >>> item = six.next(gen)
        >>> ibs, qres, qreq_, incinfo = item
        >>> # verify results
        >>> run_until_name_decision_signal(ibs, qres, qreq_, incinfo)

    qres.ishow_top(ibs, sidebyside=False, show_query=True)
    """
    print('--- Identifying Query Animal ---')
    #name_confidence_thresh = incinfo.get('name_confidence_thresh', ut.get_sys_maxfloat())
    name_confidence_thresh = incinfo.get('name_confidence_thresh', 1.0)
    interactive = incinfo.get('interactive', False)
    metatup = incinfo.get('metatup', None)
    #print('id_stack_depth = %r' % ut.get_current_stack_depth())
    qaid = qres.get_qaid()
    choicetup = system_suggestor.get_qres_name_choices(ibs, qres)
    # ---------------------------------------------
    # Get oracle suggestion if we have the metadata
    # override the system suggestion
    if metatup is not None:
        oracle_name_suggest_tup = ao.get_oracle_name_suggestion(
            ibs, qaid, choicetup, metatup)
        name_suggest_tup = oracle_name_suggest_tup
    else:
        # Get system suggested name
        system_name_suggest_tup = system_suggestor.get_system_name_suggestion(ibs, choicetup)
        name_suggest_tup = system_name_suggest_tup
    # ---------------------------------------------
    # Have the system ask the user if it is not confident in its decision
    autoname_msg, chosen_names, name_confidence = name_suggest_tup
    print('autoname_msg=')
    print(autoname_msg)
    print('... checking confidence=%r in name decision.' % (name_confidence,))
    if name_confidence < name_confidence_thresh:
        print('... confidence is too low. need user input')
        if interactive:
            print('... asking user for input')
            if qreq_.normalizer is not None:
                qreq_.normalizer.visualize(fnum=511, verbose=False)
            ut.embed()
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


def exec_name_decision_and_continue(chosen_names, ibs, qres, qreq_,
                                    incinfo=None):
    """
    called either directory or using a callbackfrom the qt harness
    """
    print('--- Updating Exemplars ---')
    if not incinfo.get('dry', False):
        qaid = qres.get_qaid()
        #assert ibs.is_aid_unknown(qaid), 'animal is already known'
        if chosen_names is None or len(chosen_names) == 0:
            newname = ibs.make_next_name()
            print('identifiying qaid=%r as a new animal. newname=%r' % (qaid, newname))
            ibs.set_annot_names((qaid,), (newname,))
        elif len(chosen_names) == 1:
            print('identifiying qaid=%r as name=%r' % (qaid, chosen_names,))
            ibs.set_annot_names((qaid,), chosen_names)
        elif len(chosen_names) > 1:
            merge_name = chosen_names[0]
            other_names = chosen_names[1:]
            print('identifiying qaid=%r as name=%r and merging with %r ' %
                    (qaid, merge_name, other_names))
            ibs.merge_names(merge_name, other_names)
            ibs.set_annot_names((qaid,), (merge_name,))
        # TODO update normalizer
        update_normalizer(ibs, qres, qreq_, chosen_names)
        # Do update callback so the name updates in the main GUI
        interactive = incinfo.get('interactive', False)
        update_callback = incinfo.get('update_callback', None)
        if interactive and update_callback is not None:
            # Update callback repopulates the names tree
            update_callback()
    run_until_exemplar_decision_signal(ibs, qres, qreq_, incinfo=incinfo)


def run_until_exemplar_decision_signal(ibs, qres, qreq_, incinfo=None):
    qaid = qres.get_qaid()
    exemplar_confidence_thresh = ut.get_sys_maxfloat()
    exmplr_suggestion = system_suggestor.get_system_exemplar_suggestion(ibs, qaid)
    (autoexemplar_msg, exemplar_decision, exemplar_condience) = exmplr_suggestion
    print('autoexemplar_msg=')
    print(autoexemplar_msg)
    need_user_input = (
        False and incinfo.get('interactive', False) and
        exemplar_condience < exemplar_confidence_thresh)
    if need_user_input:
        user_dialogs.wait_for_user_exemplar_decision(
            autoexemplar_msg, exemplar_decision, exemplar_condience)
    else:
        # May need to execute callback whereas whatever the interaction was
        # would issue it otherwise
        exec_exemplar_decision_and_continue(exemplar_decision, ibs, qres, qreq_, incinfo=incinfo)


def exec_exemplar_decision_and_continue(exemplar_decision, ibs, qres, qreq_,
                                        incinfo=None):
    qaid = qres.get_qaid()
    if exemplar_decision:
        if not incinfo.get('dry', False):
            ibs.set_annot_exemplar_flags((qaid,), [1])
    run_until_finish(incinfo=incinfo)


def run_until_finish(incinfo=None):
    if incinfo is not None:
        # This query run as eneded
        next_query_attr = 'next_query_callback'
        if next_query_attr in incinfo:
            incinfo[next_query_attr]()
        else:
            print('Warning: no next_query_attr')


# ---- POST DECISION ---


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
        >>> # DISABLE_DOCTEST
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.automated_matcher import *  # NOQA
        >>> import ibeis
        >>> # build test data
        >>> ibs = ibeis.opendb('testdb1')
        >>> qaid_chunk = ibs.get_valid_aids()[0:1]
        >>> exemplar_aids = ibs.get_valid_aids(is_exemplar=True)
        >>> incinfo = {}
        >>> gen = generate_subquery_steps(ibs, qaid_chunk, incinfo)
        >>> item = six.next(gen)
        >>> ibs, qres, qreq_, incinfo = item
        >>> # verify results
        >>> chosen_names = ['easy']
        >>> update_normalizer(ibs, qres, qreq_, chosen_names)
        >>> # verify results
        >>> result = str((tp_rawscore, tn_rawscore))
        >>> print(result)
    """
    # Fixme: duplicate call to get_qres_name_choices
    if len(chosen_names) != 1:
        print('NOT UPDATING normalization. only updates using simple matches')
        return
    qaid = qres.get_qaid()
    choicetup = system_suggestor.get_qres_name_choices(ibs, qres)
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
        print('new normalization example: tp_rawscore={}, tn_rawscore={}'.format(tp_rawscore, tn_rawscore))
        tp_labels = [(qaid, nid)]
        tn_labels = [(qaid, nid)]
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
