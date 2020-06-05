# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import six
import numpy as np
import utool as ut
import ubelt as ub
import pandas as pd
import itertools as it
import wbia.constants as const
from wbia.algo.graph.state import POSTV, NEGTV, INCMP, NULL
from wbia.algo.graph.refresh import RefreshCriteria

print, rrr, profile = ut.inject2(__name__)


PRINCETON_KAIA_EDGE_LIST = None


class InfrLoops(object):
    """
    Algorithm control flow loops
    """

    def main_gen(infr, max_loops=None, use_refresh=True):
        """
        The main outer loop.

        This function is designed as an iterator that will execute the graph
        algorithm main loop as automatically as possible, but if user input is
        needed, it will pause and yield the decision it needs help with. Once
        feedback is given for this item, you can continue the main loop by
        calling next. StopIteration is raised once the algorithm is complete.

        Args:
            max_loops(int): maximum number of times to run the outer loop,
                i.e. ranking is run at most this many times.
            use_refresh(bool): allow the refresh criterion to stop the algo

        Notes:
            Different phases of the main loop are implemented as subiterators

        CommandLine:
            python -m wbia.algo.graph.mixin_loops main_gen

        Doctest:
            >>> from wbia.algo.graph.mixin_loops import *
            >>> from wbia.algo.graph.mixin_simulation import UserOracle
            >>> import wbia
            >>> infr = wbia.AnnotInference('testdb1', aids='all',
            >>>                             autoinit='staging', verbose=4)
            >>> infr.params['manual.n_peek'] = 10
            >>> infr.params['ranking.ntop'] = 1
            >>> infr.oracle = UserOracle(.99, rng=0)
            >>> infr.simulation_mode = False
            >>> infr.reset()
            >>> #infr.load_published()
            >>> gen = infr.main_gen()
            >>> while True:
            >>>     try:
            >>>         reviews = next(gen)
            >>>         edge, priority, data = reviews[0]
            >>>         feedback = infr.request_oracle_review(edge)
            >>>         infr.add_feedback(edge, **feedback)
            >>>     except StopIteration:
            >>>         break
        """
        infr.print('Starting main loop', 1)
        infr.print('infr.params = {}'.format(ut.repr3(infr.params)))
        if max_loops is None:
            max_loops = infr.params['algo.max_outer_loops']
            if max_loops is None:
                max_loops = np.inf

        if infr.test_mode:
            print('------------------ {} -------------------'.format(infr.name))

        # Initialize a refresh criteria
        infr.init_refresh()

        infr.phase = 0
        # Phase 0.1: Ensure the user sees something immediately
        if infr.params['algo.quickstart']:
            infr.loop_phase = 'quickstart_init'
            # quick startup. Yield a bunch of random edges
            num = infr.params['manual.n_peek']
            user_request = []
            for edge in ut.random_combinations(infr.aids, 2, num=num):
                user_request += [infr._make_review_tuple(edge, None)]
                yield user_request

        if infr.params['algo.hardcase']:
            infr.loop_phase = 'hardcase_init'
            # Check previously labeled edges that where the groundtruth and the
            # verifier disagree.
            for _ in infr.hardcase_review_gen():
                yield _

        if infr.params['inference.enabled']:
            infr.loop_phase = 'incon_recover_init'
            # First, fix any inconsistencies
            for _ in infr.incon_recovery_gen():
                yield _

        # Phase 0.2: Ensure positive redundancy (this is generally quick)
        # so the user starts seeing real work after one random review is made
        # unless the graph is already positive redundant.
        if infr.params['redun.enabled'] and infr.params['redun.enforce_pos']:
            infr.loop_phase = 'pos_redun_init'
            # Fix positive redundancy of anything within the loop
            for _ in infr.pos_redun_gen():
                yield _

        infr.phase = 1
        if infr.params['ranking.enabled']:
            for count in it.count(0):

                infr.print('Outer loop iter %d ' % (count,))

                # Phase 1: Try to merge PCCs by searching for LNBNN candidates
                infr.loop_phase = 'ranking_{}'.format(count)
                for _ in infr.ranked_list_gen(use_refresh):
                    yield _

                terminate = infr.refresh.num_meaningful == 0
                if terminate:
                    infr.print('Triggered break criteria', 1, color='red')

                # Phase 2: Ensure positive redundancy.
                infr.phase = 2
                infr.loop_phase = 'posredun_{}'.format(count)
                if all(ut.take(infr.params, ['redun.enabled', 'redun.enforce_pos'])):
                    # Fix positive redundancy of anything within the loop
                    for _ in infr.pos_redun_gen():
                        yield _

                print('prob_any_remain = %r' % (infr.refresh.prob_any_remain(),))
                print(
                    'infr.refresh.num_meaningful = {!r}'.format(
                        infr.refresh.num_meaningful
                    )
                )

                if (count + 1) >= max_loops:
                    infr.print('early stop', 1, color='red')
                    break

                if terminate:
                    infr.print('break triggered')
                    break

        infr.phase = 3
        # Phase 0.3: Ensure positive redundancy (this is generally quick)
        if all(ut.take(infr.params, ['redun.enabled', 'redun.enforce_neg'])):
            # Phase 3: Try to automatically acheive negative redundancy without
            # asking the user to do anything but resolve inconsistency.
            infr.print('Entering phase 3', 1, color='red')
            infr.loop_phase = 'negredun'
            for _ in infr.neg_redun_gen():
                yield _

        infr.phase = 4
        infr.print('Terminate', 1, color='red')
        infr.print('Exiting main loop')

        if infr.params['inference.enabled']:
            infr.assert_consistency_invariant()

    def hardcase_review_gen(infr):
        """
        Subiterator for hardcase review

        Re-review non-confident edges that vsone did not classify correctly
        """
        infr.print('==============================', color='white')
        infr.print('--- HARDCASE PRIORITY LOOP ---', color='white')

        verifiers = infr.learn_evaluation_verifiers()
        verif = verifiers['match_state']

        edges_ = list(infr.edges())
        real_ = list(infr.edge_decision_from(edges_))
        flags_ = [r in {POSTV, NEGTV, INCMP} for r in real_]
        real = ut.compress(real_, flags_)
        edges = ut.compress(edges_, flags_)

        hardness = 1 - verif.easiness(edges, real)

        if True:
            df = pd.DataFrame({'edges': edges, 'real': real})
            df['hardness'] = hardness

            pred = verif.predict(edges)
            df['pred'] = pred.values

            df.sort_values('hardness', ascending=False)
            infr.print('hardness analysis')
            infr.print(str(df))

            infr.print('infr status: ' + ut.repr4(infr.status()))

        # Don't re-review anything that was confidently reviewed
        # CONFIDENCE = const.CONFIDENCE
        # CODE_TO_INT = CONFIDENCE.CODE_TO_INT.copy()
        # CODE_TO_INT[CONFIDENCE.CODE.UNKNOWN] = 0
        # conf = ut.take(CODE_TO_INT, infr.gen_edge_values(
        #     'confidence', edges, on_missing='default',
        #     default=CONFIDENCE.CODE.UNKNOWN))

        # This should only be run with certain params
        assert not infr.params['autoreview.enabled']
        assert not infr.params['redun.enabled']
        assert not infr.params['ranking.enabled']
        assert infr.params['inference.enabled']
        # const.CONFIDENCE.CODE.PRETTY_SURE
        if infr.params['queue.conf.thresh'] is None:
            # != 'pretty_sure':
            infr.print('WARNING: should queue.conf.thresh = "pretty_sure"?')

        # work around add_candidate_edges
        infr.prioritize(metric='hardness', edges=edges, scores=hardness)
        infr.set_edge_attrs('hardness', ut.dzip(edges, hardness))
        for _ in infr._inner_priority_gen(use_refresh=False):
            yield _

    def ranked_list_gen(infr, use_refresh=True):
        """
        Subiterator for phase1 of the main algorithm

        Calls the underlying ranking algorithm and prioritizes the results
        """
        infr.print('============================', color='white')
        infr.print('--- RANKED LIST LOOP ---', color='white')
        n_prioritized = infr.refresh_candidate_edges()
        if n_prioritized == 0:
            infr.print('RANKING ALGO FOUND NO NEW EDGES')
            return
        if use_refresh:
            infr.refresh.clear()
        for _ in infr._inner_priority_gen(use_refresh):
            yield _

    def incon_recovery_gen(infr):
        """
        Subiterator for recovery mode of the mainm algorithm

        Iterates until the graph is consistent

        Note:
            inconsistency recovery is implicitly handled by the main algorithm,
            so other phases do not need to call this explicitly. This exists
            for the case where the only mode we wish to run is inconsistency
            recovery.
        """
        maybe_error_edges = list(infr.maybe_error_edges())
        if len(maybe_error_edges) == 0:
            raise StopIteration()
        infr.print('============================', color='white')
        infr.print('--- INCON RECOVER LOOP ---', color='white')
        infr.queue.clear()
        infr.add_candidate_edges(maybe_error_edges)
        for _ in infr._inner_priority_gen(use_refresh=False):
            yield _

    def pos_redun_gen(infr):
        """
        Subiterator for phase2 of the main algorithm.

        Searches for decisions that would commplete positive redundancy

        Doctest:
            >>> from wbia.algo.graph.mixin_loops import *
            >>> import wbia
            >>> infr = wbia.AnnotInference('PZ_MTEST', aids='all',
            >>>                             autoinit='staging', verbose=4)
            >>> #infr.load_published()
            >>> gen = infr.pos_redun_gen()
            >>> feedback = next(gen)
        """
        infr.print('===========================', color='white')
        infr.print('--- POSITIVE REDUN LOOP ---', color='white')
        # FIXME: should prioritize inconsistentices first
        count = -1

        def thread_gen():
            # This is probably not safe
            new_edges = infr.find_pos_redun_candidate_edges()
            for new_edges in buffered_add_candidate_edges(infr, 50, new_edges):
                yield new_edges

        def serial_gen():
            # use this if threading does bad things
            if True:
                new_edges = list(infr.find_pos_redun_candidate_edges())
                if len(new_edges) > 0:
                    infr.add_candidate_edges(new_edges)
                    yield new_edges
            else:
                for new_edges in ub.chunks(infr.find_pos_redun_candidate_edges(), 100):
                    if len(new_edges) > 0:
                        infr.add_candidate_edges(new_edges)
                        yield new_edges

        def filtered_gen():
            # Buffer one-vs-one scores in the background and present an edge to
            # the user ASAP.
            # if infr.test_mode:
            candgen = serial_gen()
            # else:
            #     candgen = thread_gen()

            include_filter_set = None
            if PRINCETON_KAIA_EDGE_LIST is not None:
                # print('[mixin_loops] FILTERING EDGES FOR KAIA')
                # Sanity check, make sure that one of the edges is in the tier 1 dataset
                include_filter_set = set(PRINCETON_KAIA_EDGE_LIST)

            for new_edges in candgen:
                if infr.ibs is not None:
                    ibs = infr.ibs
                    qual_edges = ibs.unflat_map(ibs.get_annot_quality_int, new_edges)
                    valid_edges = []
                    for (u, v), (q1, q2) in zip(new_edges, qual_edges):
                        # Skip edges involving qualities less than ok
                        if q1 is not None and q1 < ibs.const.QUAL.OK:
                            continue
                        if q2 is not None and q2 < ibs.const.QUAL.OK:
                            continue
                        if include_filter_set is not None:
                            if (
                                u not in include_filter_set
                                and v not in include_filter_set
                            ):
                                continue
                        valid_edges.append((u, v))
                    if len(valid_edges) > 0:
                        yield valid_edges
                else:
                    yield new_edges

        for count in it.count(0):
            infr.print('check pos-redun iter {}'.format(count))
            infr.queue.clear()

            found_any = False

            for new_edges in filtered_gen():
                found_any = True
                gen = infr._inner_priority_gen(use_refresh=False)
                for value in gen:
                    yield value

            # print('found_any = {!r}'.format(found_any))
            if not found_any:
                break

            infr.print('not pos-reduntant yet.', color='white')
        infr.print('pos-redundancy achieved in {} iterations'.format(count + 1))

    def neg_redun_gen(infr):
        """
        Subiterator for phase3 of the main algorithm.

        Searches for decisions that would commplete negative redundancy
        """
        infr.print('===========================', color='white')
        infr.print('--- NEGATIVE REDUN LOOP ---', color='white')

        infr.queue.clear()

        only_auto = infr.params['redun.neg.only_auto']

        # TODO: outer loop that re-iterates until negative redundancy is
        # accomplished.
        needs_neg_redun = infr.find_neg_redun_candidate_edges()
        chunksize = 500
        for new_edges in ub.chunks(needs_neg_redun, chunksize):
            infr.print('another neg redun chunk')
            # Add chunks in a little at a time for faster response time
            infr.add_candidate_edges(new_edges)
            gen = infr._inner_priority_gen(use_refresh=False, only_auto=only_auto)
            for value in gen:
                yield value

    def _inner_priority_gen(infr, use_refresh=False, only_auto=False):
        """
        Helper function that implements the general inner priority loop.

        Executes reviews until the queue is empty or needs refresh

        Args:
            user_refresh (bool): if True enables the refresh criteria.
                (set to True in Phase 1)

            only_auto (bool) if True, then the user wont be prompted with
                reviews unless the graph is inconsistent.
                (set to True in Phase 3)

        Notes:
            The caller is responsible for populating the priority queue.  This
            will iterate until the queue is empty or the refresh critieron is
            triggered.
        """
        if infr.refresh:
            infr.refresh.enabled = use_refresh
        infr.print('Start inner loop with {} items in the queue'.format(len(infr.queue)))
        for count in it.count(0):
            if infr.is_recovering():
                infr.print(
                    'Still recovering after %d iterations' % (count,),
                    3,
                    color='turquoise',
                )
            else:
                # Do not check for refresh if we are recovering
                if use_refresh and infr.refresh.check():
                    infr.print(
                        'Triggered refresh criteria after %d iterations' % (count,),
                        1,
                        color='yellow',
                    )
                    break

            # If the queue is empty break
            if len(infr.queue) == 0:
                infr.print(
                    'No more edges after %d iterations, need refresh' % (count,),
                    1,
                    color='yellow',
                )
                break

            # Try to automatically do the next review.
            edge, priority = infr.peek()
            infr.print('next_review. edge={}'.format(edge), 100)

            inconsistent = infr.is_recovering(edge)

            feedback = None
            if infr.params['autoreview.enabled'] and not inconsistent:
                # Try to autoreview if we aren't in an inconsistent state
                feedback = infr.try_auto_review(edge)

            if feedback is not None:
                # Add feedback from the automated method
                infr.add_feedback(edge, priority=priority, **feedback)
            else:
                # We can't automatically review, ask for help
                if only_auto and not inconsistent:
                    # We are in auto only mode, skip manual review
                    # unless there is an inconsistency
                    infr.skip(edge)
                else:
                    if infr.simulation_mode:
                        # Use oracle feedback
                        feedback = infr.request_oracle_review(edge)
                        infr.add_feedback(edge, priority=priority, **feedback)
                    else:
                        # Yield to the user if we need to pause
                        user_request = infr.emit_manual_review(edge, priority)
                        yield user_request

        if infr.metrics_list:
            infr._print_previous_loop_statistics(count)

    def init_refresh(infr):
        refresh_params = infr.subparams('refresh')
        infr.refresh = RefreshCriteria(**refresh_params)

    def start_id_review(infr, max_loops=None, use_refresh=None):
        assert infr._gen is None, 'algo already running'
        # Just exhaust the main generator
        infr._gen = infr.main_gen(max_loops=max_loops, use_refresh=use_refresh)
        # return infr._gen

    def main_loop(infr, max_loops=None, use_refresh=True):
        """ DEPRICATED

        use list(infr.main_gen) instead
        or assert not any(infr.main_gen())
        maybe this is fine.
        """
        infr.start_id_review(max_loops=max_loops, use_refresh=use_refresh)
        # To automatically run through the loop just exhaust the generator
        try:
            result = next(infr._gen)
            assert result is None, 'need user interaction. cannot auto loop'
        except StopIteration:
            pass
        infr._gen = None


class InfrReviewers(object):
    @profile
    def try_auto_review(infr, edge):
        review = {
            'user_id': 'algo:auto_clf',
            'confidence': const.CONFIDENCE.CODE.PRETTY_SURE,
            'evidence_decision': None,
            'meta_decision': NULL,
            'timestamp_s1': None,
            'timestamp_c1': None,
            'timestamp_c2': None,
            'tags': [],
        }
        if infr.is_recovering():
            # Do not autoreview if we are in an inconsistent state
            infr.print('Must manually review inconsistent edge', 3)
            return None
        # Determine if anything passes the match threshold
        primary_task = 'match_state'

        try:
            decision_probs = infr.task_probs[primary_task][edge]
        except KeyError:
            if infr.verifiers is None:
                return None
            if infr.verifiers.get(primary_task, None) is None:
                return None
            # Compute probs if they haven't been done yet
            infr.ensure_priority_scores([edge])
            try:
                decision_probs = infr.task_probs[primary_task][edge]
            except KeyError:
                return None

        primary_thresh = infr.task_thresh[primary_task]
        decision_flags = {
            k: decision_probs[k] > thresh for k, thresh in primary_thresh.items()
        }
        hasone = sum(decision_flags.values()) == 1
        auto_flag = False
        if hasone:
            try:
                # Check to see if it might be confounded by a photobomb
                pb_probs = infr.task_probs['photobomb_state'][edge]
                # pb_probs = infr.task_probs['photobomb_state'].loc[edge]
                # pb_probs = data['task_probs']['photobomb_state']
                pb_thresh = infr.task_thresh['photobomb_state']['pb']
                confounded = pb_probs['pb'] > pb_thresh
            except KeyError:
                print('Warning: confounding task probs not set (i.e. photobombs)')
                confounded = False
            if not confounded:
                # decision = decision_flags.argmax()
                evidence_decision = ut.argmax(decision_probs)
                review['evidence_decision'] = evidence_decision
                truth = infr.match_state_gt(edge)
                if review['evidence_decision'] != truth:
                    infr.print(
                        'AUTOMATIC ERROR edge={}, truth={}, decision={}, probs={}'.format(
                            edge, truth, review['evidence_decision'], decision_probs
                        ),
                        2,
                        color='darkred',
                    )
                auto_flag = True
        if auto_flag and infr.verbose > 1:
            infr.print('Automatic review success')

        if auto_flag:
            return review
        else:
            return None

    def request_oracle_review(infr, edge, **kw):
        truth = infr.match_state_gt(edge)
        feedback = infr.oracle.review(edge, truth, infr, **kw)
        return feedback

    def _make_review_tuple(infr, edge, priority=None):
        """ Makes tuple to be sent back to the user """
        edge_data = infr.get_nonvisual_edge_data(edge, on_missing='default')
        # Extra information
        edge_data['nid_edge'] = infr.pos_graph.node_labels(*edge)
        if infr.queue is None:
            edge_data['queue_len'] = 0
        else:
            edge_data['queue_len'] = len(infr.queue)
        edge_data['n_ccs'] = (
            len(infr.pos_graph.connected_to(edge[0])),
            len(infr.pos_graph.connected_to(edge[1])),
        )
        return (edge, priority, edge_data)

    def emit_manual_review(infr, edge, priority=None):
        """
        Emits a signal containing edges that need review. The callback should
        present them to a user, get feedback, and then call on_accpet.
        """
        infr.print('emit_manual_review', 100)
        # Emit a list of reviews that can be considered.
        # The first is the most important
        user_request = []
        user_request += [infr._make_review_tuple(edge, priority)]
        try:
            for edge_, priority in infr.peek_many(infr.params['manual.n_peek']):
                if edge == edge_:
                    continue
                user_request += [infr._make_review_tuple(edge_, priority)]
        except TypeError:
            pass

        # If registered, send the request via a callback.
        request_review = infr.callbacks.get('request_review', None)
        if request_review is not None:
            # Send these reviews to a user
            request_review(user_request)
        # Otherwise the current process must handle the request by return value
        return user_request

    def skip(infr, edge):
        infr.print('skipping edge={}'.format(edge), 100)
        try:
            del infr.queue[edge]
        except Exception:
            pass

    def accept(infr, feedback):
        """
        Called when user has completed feedback from qt or web
        """
        annot1_state = feedback.pop('annot1_state', None)
        annot2_state = feedback.pop('annot2_state', None)
        if annot1_state:
            infr.add_node_feedback(**annot1_state)
        if annot2_state:
            infr.add_node_feedback(**annot2_state)
        infr.add_feedback(**feedback)

        if infr.params['manual.autosave']:
            infr.write_wbia_staging_feedback()

    def continue_review(infr):
        infr.print('continue_review', 10)
        if infr._gen is None:
            return None
        try:
            user_request = next(infr._gen)
        except StopIteration:
            review_finished = infr.callbacks.get('review_finished', None)
            if review_finished is not None:
                review_finished()
            infr._gen = None
            user_request = None
        return user_request

    def qt_edge_reviewer(infr, edge=None):
        import wbia.guitool as gt

        gt.ensure_qapp()
        from wbia.viz import viz_graph2

        infr.manual_wgt = viz_graph2.AnnotPairDialog(
            edge=edge, infr=infr, standalone=False, cfgdict=infr.verifier_params
        )
        if edge is not None:
            # infr.emit_manual_review(edge, priority=None)
            infr.manual_wgt.seek(0)
            # infr.manual_wgt.show()
        return infr.manual_wgt

    def qt_review_loop(infr):
        r"""
        TODO: The loop parts should be a non-mixin class

        Qt review loop entry point

        CommandLine:
            python -m wbia.algo.graph.mixin_loops qt_review_loop --show

        Example:
            >>> # SCRIPT
            >>> import utool as ut
            >>> import wbia
            >>> ibs = wbia.opendb('PZ_MTEST')
            >>> infr = wbia.AnnotInference(ibs, 'all', autoinit=True)
            >>> infr.ensure_mst()
            >>> # Add dummy priorities to each edge
            >>> infr.set_edge_attrs('prob_match', ut.dzip(infr.edges(), [1]))
            >>> infr.prioritize('prob_match', infr.edges(), reset=True)
            >>> infr.params['redun.enabled'] = False
            >>> win = infr.qt_review_loop()
            >>> import wbia.guitool as gt
            >>> gt.qtapp_loop(qwin=win, freq=10)
        """
        infr.qt_edge_reviewer()
        # infr.continue_review()
        return infr.manual_wgt


if False:
    # Testing generating using threads
    from threading import Thread

    _sentinel = object()

    class _background_consumer(Thread):
        """
        Will fill the queue with content of the source in a separate thread.

        Ignore:
            >>> from wbia.algo.graph.mixin_loops import *
            >>> import wbia
            >>> infr = wbia.AnnotInference('PZ_MTEST', aids='all',
            >>>                             autoinit='staging', verbose=4)
            >>> infr.load_published()
            >>> gen = infr.find_pos_redun_candidate_edges()
            >>> parbuf = buffered_add_candidate_edges(infr, 3, gen)
            >>> next(parbuf)

        """

        def __init__(self, infr, queue, source):
            Thread.__init__(self)

            self.infr = infr

            self._queue = queue
            self._source = source

        def run(self):
            # for edges in ub.chunks(self._source, 5):
            #     print('edges = {!r}'.format(edges))
            #     # print('put item = {!r}'.format(item))
            #     # probably not thread safe
            #     infr = self.infr
            #     infr.add_candidate_edges(edges)
            #     for item in edges:
            #         self._queue.put(item)
            for _, item in enumerate(self._source):
                # import threading
                # import multiprocessing
                # print('multiproc = ' + str(multiprocessing.current_process()))
                # print('thread = ' + str(threading.current_thread()))
                # print('_ = {!r}'.format(_))
                # print('item = {!r}'.format(item))
                # print('put item = {!r}'.format(item))
                # probably not thread safe
                infr = self.infr
                infr.add_candidate_edges([item])
                self._queue.put(item, block=True)

            # Signal the consumer we are done.
            self._queue.put(_sentinel)

    class buffered_add_candidate_edges(object):
        """
        Buffers content of an iterator polling the contents of the given
        iterator in a separate thread.
        When the consumer is faster than many producers, this kind of
        concurrency and buffering makes sense.

        The size parameter is the number of elements to buffer.

        The source must be threadsafe.
        """

        def __init__(self, infr, size, source):
            if six.PY2:
                from Queue import Queue
            else:
                from queue import Queue
            self._queue = Queue(size)

            self._poller = _background_consumer(infr, self._queue, source)
            self._poller.daemon = True
            self._poller.start()

        def __iter__(self):
            return self

        def __next__(self):
            item = self._queue.get(True)
            if item is _sentinel:
                raise StopIteration()
            return item

        next = __next__


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.algo.graph.mixin_loops
        python -m wbia.algo.graph.mixin_loops --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
