# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from wbia.algo.graph import demo
import utool as ut
from wbia.algo.graph.state import POSTV, NEGTV, INCMP, UNREV  # NOQA
from wbia.algo.graph.state import SAME, DIFF, NULL  # NOQA

print, rrr, profile = ut.inject2(__name__)


def do_infr_test(ccs, edges, new_edges):
    """
    Creates a graph with `ccs` + `edges` and then adds `new_edges`
    """
    # import networkx as nx
    import wbia.plottool as pt

    infr = demo.make_demo_infr(ccs, edges)

    if ut.show_was_requested():
        pt.qtensure()

    # Preshow
    fnum = 1
    if ut.show_was_requested():
        infr.set_node_attrs('shape', 'circle')
        infr.show(
            pnum=(2, 1, 1),
            fnum=fnum,
            show_unreviewed_edges=True,
            show_reviewed_cuts=True,
            splines='spline',
            show_inferred_diff=True,
            groupby='name_label',
            show_labels=True,
            pickable=True,
        )
        pt.set_title('pre-review')
        pt.gca().set_aspect('equal')
        infr.set_node_attrs('pin', 'true')
        # fig1 = pt.gcf()
        # fig1.canvas.mpl_connect('pick_event', ut.partial(on_pick, infr=infr))

    infr1 = infr
    infr2 = infr.copy()
    for new_edge in new_edges:
        aid1, aid2, data = new_edge
        evidence_decision = data['evidence_decision']
        infr2.add_feedback((aid1, aid2), evidence_decision)
    infr2.relabel_using_reviews(rectify=False)
    infr2.apply_nondynamic_update()

    # Postshow
    if ut.show_was_requested():
        infr2.show(
            pnum=(2, 1, 2),
            fnum=fnum,
            show_unreviewed_edges=True,
            show_inferred_diff=True,
            show_labels=True,
        )
        pt.gca().set_aspect('equal')
        pt.set_title('post-review')
        # fig2 = pt.gcf()
        # if fig2 is not fig1:
        #     fig2.canvas.mpl_connect('pick_event', ut.partial(on_pick, infr=infr2))

    class Checker(object):
        """
        Asserts pre and post test properties of the graph
        """

        def __init__(self, infr1, infr2):
            self._errors = []
            self.infr1 = infr1
            self.infr2 = infr2

        def __call__(self, infr, u, v, key, val, msg):
            data = infr.get_nonvisual_edge_data((u, v))
            if data is None:
                assert infr.graph.has_edge(u, v), 'uv=%r, %r does not exist' % (u, v)
            got = data.get(key)
            if got != val:
                msg1 = 'key=%s %r!=%r, ' % (key, got, val)
                errmsg = ''.join(
                    [
                        msg1,
                        msg,
                        '\nedge=',
                        ut.repr2((u, v)),
                        '\n',
                        infr.repr_edge_data(data),
                    ]
                )
                self._errors.append(errmsg)

        def custom_precheck(self, func):
            try:
                func(self.infr1)
            except AssertionError as ex:
                self._errors.append(str(ex))

        def after(self, errors=[]):
            """
            Delays error reporting until after visualization

            prints errors, then shows you the graph, then
            finally if any errors were discovered they are raised
            """

            errors = errors + self._errors
            if errors:
                ut.cprint('PRINTING %d FAILURE' % (len(errors)), 'red')
                for msg in errors:
                    print(msg)
                ut.cprint('HAD %d FAILURE' % (len(errors)), 'red')
            if ut.show_was_requested():
                pt.all_figures_tile(percent_w=0.5)
                ut.show_if_requested()
            if errors:
                raise AssertionError('There were errors')

    check = Checker(infr1, infr2)
    return infr1, infr2, check


def case_negative_infr():
    """
    CommandLine:
        python -m wbia.algo.graph.tests.dyn_cases case_negative_infr --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.graph.tests.dyn_cases import *  # NOQA
        >>> case_negative_infr()
    """
    # Initial positive reviews
    ccs = [[1, 2, 3], [9]]
    # Add in initial reviews
    edges = [
        (9, 7, {'evidence_decision': NEGTV}),
        (1, 7, {'inferred_state': None}),
        (1, 9, {'inferred_state': None}),
    ]
    # Add in scored, but unreviewed edges
    new_edges = [(3, 9, {'evidence_decision': NEGTV})]
    infr1, infr2, check = do_infr_test(ccs, edges, new_edges)

    check(
        infr2,
        1,
        7,
        'inferred_state',
        None,
        'negative review of an edge should not jump more than one component',
    )

    check(
        infr2,
        1,
        9,
        'inferred_state',
        'diff',
        'negative review of an edge should cut within one jump',
    )

    check.after()


def case_match_infr():
    """
    CommandLine:
        python -m wbia.algo.graph.tests.dyn_cases case_match_infr --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.graph.tests.dyn_cases import *  # NOQA

        >>> case_match_infr()
    """
    # Initial positive reviews
    ccs = [[2], [7], [9, 10]]
    # Add in initial reviews
    edges = [
        (9, 8, {'evidence_decision': NEGTV}),
        (7, 2, {}),
    ]
    # Add in scored, but unreviewed edges
    edges += [
        (2, 8, {'inferred_state': None}),
        (2, 9, {'inferred_state': None}),
    ]
    new_edges = [(2, 10, {'evidence_decision': POSTV})]
    infr1, infr2, check = do_infr_test(ccs, edges, new_edges)

    # Checks out of cc inferences
    check(infr2, 2, 9, 'inferred_state', 'same', 'should infer a match')
    check(infr2, 2, 8, 'inferred_state', 'diff', 'should infer a negative')
    check(infr1, 2, 7, 'inferred_state', None, 'discon should have inference')

    check(infr2, 2, 7, 'inferred_state', None, 'discon should have inference')
    check.after()


def case_inconsistent():
    """
    CommandLine:
        python -m wbia.algo.graph.tests.dyn_cases case_inconsistent --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.graph.tests.dyn_cases import *  # NOQA
        >>> case_inconsistent()
    """
    ccs = [[1, 2], [3, 4, 5]]  # [6, 7]]
    edges = [
        (2, 3, {'evidence_decision': NEGTV}),
    ]
    edges += [
        (4, 1, {'inferred_state': None}),
        # (2, 7, {'inferred_state': None}),
    ]
    new_edges = [(1, 5, {'evidence_decision': POSTV})]
    infr1, infr2, check = do_infr_test(ccs, edges, new_edges)
    # Make sure the previously inferred edge is no longer inferred
    check(infr1, 4, 1, 'inferred_state', 'diff', 'should initially be an inferred diff')
    check(
        infr2,
        4,
        1,
        'inferred_state',
        'inconsistent_internal',
        'should not be inferred after incon',
    )
    check(infr2, 4, 3, 'maybe_error', True, 'need to have a maybe split')
    check.after()


def case_redo_incon():
    """
    CommandLine:
        python -m wbia.algo.graph.tests.dyn_cases case_redo_incon --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.graph.tests.dyn_cases import *  # NOQA
        >>> case_redo_incon()
    """
    ccs = [[1, 2], [3, 4]]  # [6, 7]]
    edges = [
        (2, 3, {'evidence_decision': NEGTV}),
        (1, 4, {'evidence_decision': NEGTV}),
    ]
    edges += []
    new_edges = [(2, 3, {'evidence_decision': POSTV})]
    infr1, infr2, check = do_infr_test(ccs, edges, new_edges)

    maybe_splits = infr2.get_edge_attrs('maybe_error', default=None)
    print('maybe_splits = %r' % (maybe_splits,))
    if not any(maybe_splits.values()):
        ut.cprint('FAILURE', 'red')
        print('At least one edge should be marked as a split')

    check.after()


def case_override_inference():
    """
    CommandLine:
        python -m wbia.algo.graph.tests.dyn_cases case_override_inference --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.graph.tests.dyn_cases import *  # NOQA
        >>> case_override_inference()
    """
    ccs = [[1, 2, 3, 4, 5]]
    edges = [
        (1, 3, {'inferred_state': 'same'}),
        (1, 4, {'inferred_state': 'same'}),
        # (1, 5, {'inferred_state': 'same'}),
        (1, 2, {'inferred_state': 'same', 'num_reviews': 2}),
        (2, 3, {'inferred_state': 'same', 'num_reviews': 2}),
        (2, 4, {'inferred_state': 'same'}),
        (2, 5, {'inferred_state': 'same'}),
        (3, 4, {'inferred_state': 'same', 'num_reviews': 100}),
        (4, 5, {'inferred_state': 'same', 'num_reviews': 0.01}),
    ]
    edges += []
    new_edges = [
        (1, 5, {'evidence_decision': NEGTV}),
        (5, 2, {'evidence_decision': NEGTV}),
    ]
    infr1, infr2, check = do_infr_test(ccs, edges, new_edges)
    # Make sure that the inferred edges are no longer inferred when an
    # inconsistent case is introduced
    check(infr2, 1, 4, 'maybe_error', None, 'should not split inferred edge')
    check(infr2, 4, 5, 'maybe_error', True, 'split me')
    check(
        infr2,
        5,
        2,
        'inferred_state',
        'inconsistent_internal',
        'inference should be overriden',
    )
    check.after()


def case_undo_match():
    """
    CommandLine:
        python -m wbia.algo.graph.tests.dyn_cases case_undo_match --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.graph.tests.dyn_cases import *  # NOQA
        >>> case_undo_match()
    """
    ccs = [[1, 2]]
    edges = []
    new_edges = [(1, 2, {'evidence_decision': NEGTV})]
    infr1, infr2, check = do_infr_test(ccs, edges, new_edges)

    check(infr2, 1, 2, 'inferred_state', 'diff', 'should have cut edge')
    check.after()


def case_undo_negative():
    """
    CommandLine:
        python -m wbia.algo.graph.tests.dyn_cases case_undo_negative --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.graph.tests.dyn_cases import *  # NOQA
        >>> case_undo_negative()
    """
    ccs = [[1], [2]]
    edges = [
        (1, 2, {'evidence_decision': NEGTV}),
    ]
    new_edges = [(1, 2, {'evidence_decision': POSTV})]
    infr1, infr2, check = do_infr_test(ccs, edges, new_edges)
    check(infr2, 1, 2, 'inferred_state', 'same', 'should have matched edge')
    check.after()


def case_incon_removes_inference():
    """
    CommandLine:
        python -m wbia.algo.graph.tests.dyn_cases case_incon_removes_inference --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.graph.tests.dyn_cases import *  # NOQA
        >>> case_incon_removes_inference()
    """
    ccs = [[1, 2, 3], [4, 5, 6]]
    edges = [
        (3, 4, {'evidence_decision': NEGTV}),
        (1, 5, {'evidence_decision': NEGTV}),
        (2, 5, {}),
        (1, 6, {}),
    ]
    new_edges = [(3, 4, {'evidence_decision': POSTV})]
    infr1, infr2, check = do_infr_test(ccs, edges, new_edges)

    check(infr1, 2, 5, 'inferred_state', 'diff', 'should be preinferred')
    check(
        infr2,
        2,
        5,
        'inferred_state',
        'inconsistent_internal',
        'should be uninferred on incon',
    )
    check.after()


def case_inferable_notcomp1():
    """
    make sure notcomparable edges can be inferred

    CommandLine:
        python -m wbia.algo.graph.tests.dyn_cases case_inferable_notcomp1 --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.graph.tests.dyn_cases import *  # NOQA
        >>> case_inferable_notcomp1()
    """
    ccs = [[1, 2], [3, 4]]
    edges = [
        (2, 3, {'evidence_decision': NEGTV}),
    ]
    new_edges = [(1, 4, {'evidence_decision': INCMP})]
    infr1, infr2, check = do_infr_test(ccs, edges, new_edges)
    check(infr2, 1, 4, 'inferred_state', 'diff', 'should be inferred')
    check.after()


def case_inferable_update_notcomp():
    """
    make sure inference updates for nocomparable edges

    CommandLine:
        python -m wbia.algo.graph.tests.dyn_cases case_inferable_update_notcomp --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.graph.tests.dyn_cases import *  # NOQA
        >>> case_inferable_update_notcomp()
    """
    ccs = [[1, 2], [3, 4]]
    edges = [
        (2, 3, {'evidence_decision': NEGTV}),
        (1, 4, {'evidence_decision': INCMP}),
    ]
    new_edges = [(2, 3, {'evidence_decision': POSTV})]
    infr1, infr2, check = do_infr_test(ccs, edges, new_edges)
    check(infr1, 1, 4, 'inferred_state', 'diff', 'should be inferred diff')
    check(infr2, 1, 4, 'inferred_state', 'same', 'should be inferred same')
    check.after()


def case_notcomp_remove_infr():
    """
    CommandLine:
        python -m wbia.algo.graph.tests.dyn_cases case_notcomp_remove_infr --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.graph.tests.dyn_cases import *  # NOQA
        >>> case_notcomp_remove_infr()
    """
    ccs = [[1, 2, 3], [4, 5, 6]]
    edges = [
        (1, 4, {'evidence_decision': POSTV}),
        # (1, 4, {'evidence_decision': INCMP}),
        (2, 5, {'evidence_decision': INCMP}),
        (3, 6, {'evidence_decision': INCMP}),
    ]
    new_edges = [(1, 4, {'evidence_decision': INCMP})]
    infr1, infr2, check = do_infr_test(ccs, edges, new_edges)
    check(infr2, 1, 4, 'inferred_state', INCMP, 'can not infer match here!')
    check(infr2, 2, 5, 'inferred_state', INCMP, 'can not infer match here!')
    check(infr2, 3, 6, 'inferred_state', INCMP, 'can not infer match here!')
    check.after()


def case_notcomp_remove_cuts():
    """
    CommandLine:
        python -m wbia.algo.graph.tests.dyn_cases case_notcomp_remove_cuts --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.graph.tests.dyn_cases import *  # NOQA
        >>> case_notcomp_remove_cuts()
    """
    ccs = [[1, 2, 3], [4, 5, 6]]
    edges = [
        (1, 4, {'evidence_decision': NEGTV}),
        # (1, 4, {'evidence_decision': INCMP}),
        (2, 5, {'evidence_decision': INCMP}),
        (3, 6, {'evidence_decision': INCMP}),
    ]
    new_edges = [(1, 4, {'evidence_decision': INCMP})]
    infr1, infr2, check = do_infr_test(ccs, edges, new_edges)
    check(infr1, 1, 4, 'inferred_state', 'diff', 'should infer diff!')
    check(infr1, 2, 5, 'inferred_state', 'diff', 'should infer diff!')
    check(infr1, 3, 6, 'inferred_state', 'diff', 'should infer diff!')
    check(infr2, 1, 4, 'evidence_decision', INCMP, 'can not infer cut here!')
    check(infr2, 2, 5, 'inferred_state', INCMP, 'can not infer cut here!')
    check(infr2, 3, 6, 'inferred_state', INCMP, 'can not infer cut here!')
    check.after()


def case_keep_in_cc_infr_post_negative():
    """
    CommandLine:
        python -m wbia.algo.graph.tests.dyn_cases case_keep_in_cc_infr_post_negative --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.graph.tests.dyn_cases import *  # NOQA
        >>> case_keep_in_cc_infr_post_negative()
    """
    ccs = [[1, 2, 3], [4]]
    edges = [(1, 3), (1, 4), (2, 4), (3, 4)]
    new_edges = [(4, 2, {'evidence_decision': NEGTV})]
    infr1, infr2, check = do_infr_test(ccs, edges, new_edges)
    check(infr1, 3, 4, 'inferred_state', None, 'should be no inference')
    check(infr1, 1, 3, 'inferred_state', 'same', 'should be inferred')
    check(infr2, 1, 3, 'inferred_state', 'same', 'should remain inferred')
    check(infr2, 3, 4, 'inferred_state', 'diff', 'should become inferred')
    check.after()


def case_keep_in_cc_infr_post_notcomp():
    """
    CommandLine:
        python -m wbia.algo.graph.tests.dyn_cases case_keep_in_cc_infr_post_notcomp --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.graph.tests.dyn_cases import *  # NOQA
        >>> case_keep_in_cc_infr_post_notcomp()
    """
    ccs = [[1, 2, 3], [4]]
    edges = [(1, 3), (1, 4), (2, 4), (3, 4)]
    new_edges = [(4, 2, {'evidence_decision': INCMP})]
    infr1, infr2, check = do_infr_test(ccs, edges, new_edges)
    check(infr1, 3, 4, 'inferred_state', None, 'should not be inferred')
    check(infr1, 1, 3, 'inferred_state', 'same', 'should be inferred')
    check(infr2, 1, 3, 'inferred_state', 'same', 'should remain inferred')
    check(infr2, 3, 4, 'inferred_state', None, 'should not become inferred')
    check.after()


def case_out_of_subgraph_modification():
    """
    CommandLine:
        python -m wbia.algo.graph.tests.dyn_cases case_out_of_subgraph_modification --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.graph.tests.dyn_cases import *  # NOQA
        >>> case_out_of_subgraph_modification()
    """
    # A case where a review between two ccs modifies state outside of
    # the subgraph of ccs
    ccs = [[1, 2], [3, 4], [5, 6]]
    edges = [(2, 6), (4, 5, {'evidence_decision': NEGTV})]
    new_edges = [(2, 3, {'evidence_decision': POSTV})]
    infr1, infr2, check = do_infr_test(ccs, edges, new_edges)
    check(infr1, 2, 6, 'inferred_state', None, 'should not be inferred')
    check(infr2, 2, 6, 'inferred_state', 'diff', 'should be inferred')
    check.after()


def case_flag_merge():
    """
    CommandLine:
        python -m wbia.algo.graph.tests.dyn_cases case_flag_merge --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.graph.tests.dyn_cases import *  # NOQA
        >>> case_flag_merge()
    """
    # A case where a review between two ccs modifies state outside of
    # the subgraph of ccs
    ccs = []
    edges = [
        (1, 2, {'evidence_decision': POSTV, 'num_reviews': 2}),
        (4, 1, {'evidence_decision': POSTV, 'num_reviews': 1}),
        (2, 4, {'evidence_decision': NEGTV, 'num_reviews': 1}),
    ]
    # Ensure that the negative edge comes back as potentially in error
    new_edges = [(1, 4, {'evidence_decision': POSTV})]
    infr1, infr2, check = do_infr_test(ccs, edges, new_edges)

    @check.custom_precheck
    def check_pre_state(infr):
        assert infr.nid_to_errors[1] == {(1, 4)}

    check(infr1, 2, 4, 'maybe_error', None, 'match edge should flag first None')
    check(infr1, 1, 4, 'maybe_error', True, 'match edge should flag first True')
    check(infr2, 2, 4, 'maybe_error', True, 'negative edge should flag second True')
    check(infr2, 1, 4, 'maybe_error', None, 'negative edge should flag second None')
    check.after()


def case_all_types():
    """
    CommandLine:
        python -m wbia.algo.graph.tests.dyn_cases case_all_types --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.algo.graph.tests.dyn_cases import *  # NOQA
        >>> case_all_types()
    """
    # A case where a review between two ccs modifies state outside of
    # the subgraph of ccs
    ccs = []
    # Define edges within components
    edges = [
        # Inconsistent component
        (11, 12, {'evidence_decision': POSTV}),
        (12, 13, {'evidence_decision': POSTV}),
        (11, 13, {'evidence_decision': NEGTV}),
        (11, 14, {'evidence_decision': POSTV}),
        (12, 14, {'evidence_decision': POSTV}),
        (13, 14, {}),
        (11, 15, {'evidence_decision': POSTV}),
        (12, 15, {'evidence_decision': INCMP}),
        # Positive component (with notcomp)
        (21, 22, {'evidence_decision': POSTV}),
        (22, 23, {'evidence_decision': POSTV}),
        (21, 23, {'evidence_decision': INCMP}),
        (21, 24, {'evidence_decision': POSTV}),
        (22, 24, {'evidence_decision': POSTV}),
        (23, 24, {}),
        # Positive component (with unreview)
        (31, 32, {'evidence_decision': POSTV}),
        (32, 33, {'evidence_decision': POSTV}),
        (31, 33, {'evidence_decision': POSTV}),
        (31, 34, {'evidence_decision': POSTV}),
        (32, 34, {'evidence_decision': POSTV}),
        (33, 34, {}),
        # Positive component
        (41, 42, {'evidence_decision': POSTV}),
        (42, 43, {'evidence_decision': POSTV}),
        (41, 43, {'evidence_decision': POSTV}),
        # Positive component (extra)
        (51, 52, {'evidence_decision': POSTV}),
        (52, 53, {'evidence_decision': POSTV}),
        (51, 53, {'evidence_decision': POSTV}),
        # Positive component (isolated)
        (61, 62, {'evidence_decision': POSTV}),
    ]
    # Define edges between components
    edges += [
        # 1 - 2
        (11, 21, {}),
        (12, 22, {}),
        # 1 - 3
        (11, 31, {}),
        (12, 32, {'evidence_decision': NEGTV}),
        (13, 33, {}),
        # 1 - 4
        (11, 41, {}),
        (12, 42, {'evidence_decision': INCMP}),
        (13, 43, {}),
        # 1 - 5
        (11, 51, {'evidence_decision': INCMP}),
        (12, 52, {'evidence_decision': NEGTV}),
        (13, 53, {}),
        # 2 - 3
        (21, 31, {'evidence_decision': INCMP}),
        (22, 32, {}),
        # 2 - 4
        (21, 41, {}),
        (22, 42, {}),
        # 2 - 5
        (21, 51, {'evidence_decision': INCMP}),
        (22, 52, {'evidence_decision': NEGTV}),
        # 3 - 4
        (31, 41, {'evidence_decision': NEGTV}),
        (32, 42, {}),
    ]
    # Ensure that the negative edge comes back as potentially in error
    # new_edges = [(2, 5, {'evidence_decision': POSTV})]
    new_edges = []
    infr1, infr2, check = do_infr_test(ccs, edges, new_edges)
    errors = []
    for u, v, d in infr2.graph.edges(data=True):
        state = d.get('inferred_state', '')
        if u < 20 or v < 20:
            if state is not None and 'inconsistent' not in state:
                print('u, v, state = %r, %r, %r' % (u, v, state))
                err = AssertionError('all of cc0 should be incon')
                print(err)
                errors.append(err)
        else:
            if state is not None and 'inconsistent' in state:
                print('u, v, state = %r, %r, %r' % (u, v, state))
                err = AssertionError('outside of cc0 should not be incon')
                print(err)
                errors.append(err)
    check(
        infr1,
        13,
        14,
        'inferred_state',
        'inconsistent_internal',
        'notcomp edge should be incon',
    )
    check(infr1, 21, 31, 'inferred_state', INCMP, 'notcomp edge should remain notcomp')
    check(infr1, 22, 32, 'inferred_state', None, 'notcomp edge should transfer knowledge')
    check(
        infr1,
        12,
        42,
        'inferred_state',
        'inconsistent_external',
        'inconsistency should override notcomp',
    )
    # check(infr1, 1, 4, 'maybe_error', True, 'match edge should flag first')
    # check(infr2, 2, 4, 'maybe_error', True, 'negative edge should flag second')
    # check(infr2, 1, 4, 'maybe_error', False, 'negative edge should flag second')
    check.after(errors)


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.algo.graph.tests.dyn_cases
        python -m wbia.algo.graph.tests.dyn_cases --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
