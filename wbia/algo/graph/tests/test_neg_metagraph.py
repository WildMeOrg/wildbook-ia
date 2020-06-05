# -*- coding: utf-8 -*-
"""
TODO: These tests are good and important to run.
Ensure they are run via run_tests even though they are not doctests.

Consider moving to pytest and using xdoctest (because regular doctest does not
accept the syntax of IBEIS doctests)
"""
from __future__ import absolute_import, division, print_function
import utool as ut

(print, rrr, profile) = ut.inject2(__name__)


def test_neg_metagraph_simple_add_remove():
    """
    Test that the negative metagraph tracks the number of negative edges
    between PCCs through non-label-changing operations
    """
    from wbia.algo.graph import demo
    from wbia.algo.graph.state import POSTV, NEGTV, INCMP, UNREV, UNKWN  # NOQA

    # Create a graph with 5-sized CCs, with 3-pos-redun, and no negative edges
    infr = demo.demodata_infr(
        num_pccs=2, pcc_size=5, pos_redun=3, ignore_pair=True, infer=True
    )
    cc_a, cc_b, cc_c, cc_d = infr.positive_components()
    a1, a2, a3, a4, a5 = cc_a
    b1, b2, b3, b4, b5 = cc_b

    nmg = infr.neg_metagraph

    # Check there are 4 meta-nodes and no edges
    assert nmg.number_of_edges() == 0
    assert nmg.number_of_nodes() == 4

    # Should add 1 edge to the negative metagraph
    u, v = a1, b1
    infr.add_feedback((u, v), NEGTV)
    nid1, nid2 = infr.node_labels(u, v)
    assert nmg.edges[nid1, nid2]['weight'] == 1
    assert nmg.number_of_edges() == 1
    assert nmg.number_of_nodes() == 4

    # Adding a second time should do nothing
    edge = a1, b1
    infr.add_feedback(edge, NEGTV)
    name_edge = infr.node_labels(*edge)
    assert nmg.edges[name_edge]['weight'] == 1
    assert nmg.number_of_edges() == 1
    assert nmg.number_of_nodes() == 4

    # But adding a second between different nodes will increase the weight
    edge = a1, b2
    infr.add_feedback(edge, NEGTV)
    name_edge = infr.node_labels(*edge)
    assert nmg.edges[name_edge]['weight'] == 2
    assert nmg.number_of_edges() == 1
    assert nmg.number_of_nodes() == 4

    infr.add_feedback((u, v), NEGTV)
    assert nmg.edges[name_edge]['weight'] == 2

    # Removing or relabeling the edge will decrease the weight
    infr.add_feedback((a1, b2), INCMP)
    assert nmg.edges[name_edge]['weight'] == 1

    # And removing all will remove the negative edge
    infr.add_feedback((a1, b1), INCMP)
    assert not nmg.has_edge(*name_edge)

    infr.assert_neg_metagraph()


def test_neg_metagraph_merge():
    """
    Test that the negative metagraph tracks the number of negative edges
    between PCCs through label-changing merge operations
    """
    from wbia.algo.graph import demo
    from wbia.algo.graph.state import POSTV, NEGTV, INCMP, UNREV, UNKWN  # NOQA

    # Create a graph with 4 CCs, with 3-pos-redun, and no negative edges
    infr = demo.demodata_infr(
        num_pccs=4, pcc_size=5, pos_redun=3, ignore_pair=True, infer=True
    )
    cc_a, cc_b, cc_c, cc_d = infr.positive_components()
    a1, a2, a3, a4, a5 = cc_a
    b1, b2, b3, b4, b5 = cc_b
    c1, c2, c3, c4, c5 = cc_c
    d1, d2, d3, d4, d5 = cc_d

    nmg = infr.neg_metagraph

    # Add three negative edges between a and b
    # one between (a, c), (b, d), (a, d), and (c, d)
    A, B, C, D = infr.node_labels(a1, b1, c1, d1)

    infr.add_feedback((a1, b1), NEGTV)
    infr.add_feedback((a2, b2), NEGTV)
    infr.add_feedback((a3, b3), NEGTV)
    infr.add_feedback((a4, c4), NEGTV)
    infr.add_feedback((b4, d4), NEGTV)
    infr.add_feedback((c1, d1), NEGTV)
    infr.add_feedback((a4, d4), NEGTV)

    assert nmg.edges[(A, B)]['weight'] == 3
    assert nmg.edges[(A, C)]['weight'] == 1
    assert (B, C) not in nmg.edges
    assert nmg.edges[(A, D)]['weight'] == 1
    assert nmg.edges[(B, D)]['weight'] == 1
    assert nmg.number_of_edges() == 5
    assert nmg.number_of_nodes() == 4

    # Now merge A and B
    infr.add_feedback((a1, b1), POSTV)
    AB = infr.node_label(a1)

    # The old meta-nodes should not be combined into AB
    assert infr.node_label(b1) == AB
    assert A != B
    assert A == AB or A not in nmg.nodes
    assert B == AB or B not in nmg.nodes

    # Should have combined weights from (A, D) and (B, D)
    # And (A, C) should be brought over as-is
    assert nmg.edges[(AB, D)]['weight'] == 2
    assert nmg.edges[(AB, C)]['weight'] == 1

    # should not have a self-loop weight weight 2
    # (it decreased because we changed a previously neg edge to pos)
    assert nmg.edges[(AB, AB)]['weight'] == 2
    assert len(list(nmg.selfloop_edges())) == 1

    # nothing should change between C and D
    assert nmg.edges[(C, D)]['weight'] == 1

    # Should decrease number of nodes and edges
    assert nmg.number_of_nodes() == 3
    assert nmg.number_of_edges() == 4

    infr.assert_neg_metagraph()

    # Additional merge
    infr.add_feedback((c2, d2), POSTV)
    CD = infr.node_label(c1)
    infr.assert_neg_metagraph()

    assert nmg.number_of_nodes() == 2
    assert nmg.number_of_edges() == 3

    assert nmg.edges[(CD, CD)]['weight'] == 1
    assert nmg.edges[(AB, CD)]['weight'] == 3
    assert nmg.edges[(AB, AB)]['weight'] == 2

    # Yet another merge
    infr.add_feedback((a1, c1), POSTV)
    ABCD = infr.node_label(c1)
    assert nmg.number_of_nodes() == 1
    assert nmg.number_of_edges() == 1
    nmg.edges[(ABCD, ABCD)]['weight'] = 6
    infr.assert_neg_metagraph()


def test_neg_metagraph_split_neg():
    """
    Test that the negative metagraph tracks the number of negative edges
    between PCCs through label-changing split operations
    """
    from wbia.algo.graph import demo
    from wbia.algo.graph.state import POSTV, NEGTV, INCMP, UNREV, UNKWN  # NOQA

    # Create a graph with 4 CCs, with 3-pos-redun, and no negative edges
    infr = demo.demodata_infr(
        num_pccs=4, pcc_size=5, pos_redun=3, ignore_pair=True, infer=True
    )
    nmg = infr.neg_metagraph
    assert nmg.number_of_nodes() != infr.neg_graph.number_of_nodes()
    assert nmg.number_of_edges() == 0
    # remove all positive edges
    for edge in list(infr.pos_graph.edges()):
        infr.add_feedback(edge, NEGTV)
    # metagraph should not be isomorphic to infr.neg_graph
    assert nmg.number_of_nodes() == infr.neg_graph.number_of_nodes()
    assert nmg.number_of_edges() > 0
    assert nmg.number_of_edges() == infr.neg_graph.number_of_edges()
    infr.assert_neg_metagraph()


def test_neg_metagraph_split_incomp():
    from wbia.algo.graph import demo
    from wbia.algo.graph.state import POSTV, NEGTV, INCMP, UNREV, UNKWN  # NOQA

    infr = demo.demodata_infr(
        num_pccs=4, pcc_size=5, pos_redun=3, ignore_pair=True, infer=True
    )
    nmg = infr.neg_metagraph
    assert nmg.number_of_nodes() < infr.neg_graph.number_of_nodes()
    assert nmg.number_of_edges() == 0
    # remove all positive edges
    for edge in list(infr.pos_graph.edges()):
        infr.add_feedback(edge, INCMP)
    # metagraph should not be isomorphic to infr.neg_graph
    assert nmg.number_of_nodes() == infr.neg_graph.number_of_nodes()
    assert nmg.number_of_edges() == 0
    infr.assert_neg_metagraph()


def test_neg_metagraph_split_and_merge():
    """
    Test that the negative metagraph tracks the number of negative edges
    between PCCs through label-changing split and merge operations
    """
    from wbia.algo.graph import demo
    from wbia.algo.graph.state import POSTV, NEGTV, INCMP, UNREV, UNKWN  # NOQA

    # Create a graph with 4 CCs, with 3-pos-redun, and no negative edges
    infr = demo.demodata_infr(
        num_pccs=4, pcc_size=5, pos_redun=3, ignore_pair=True, infer=True
    )
    cc_a, cc_b, cc_c, cc_d = infr.positive_components()
    a1, a2, a3, a4, a5 = cc_a
    b1, b2, b3, b4, b5 = cc_b
    c1, c2, c3, c4, c5 = cc_c
    d1, d2, d3, d4, d5 = cc_d

    nmg = infr.neg_metagraph

    # Add three negative edges between a and b
    # one between (a, c), (b, d), (a, d), and (c, d)
    A, B, C, D = infr.node_labels(a1, b1, c1, d1)

    infr.add_feedback((a1, b1), NEGTV)
    infr.add_feedback((a2, b2), NEGTV)
    infr.add_feedback((a3, b3), NEGTV)
    infr.add_feedback((a4, c4), NEGTV)
    infr.add_feedback((b4, d4), NEGTV)
    infr.add_feedback((c1, d1), NEGTV)
    infr.add_feedback((a4, d4), NEGTV)

    assert nmg.edges[(A, B)]['weight'] == 3
    assert nmg.edges[(A, C)]['weight'] == 1
    assert (B, C) not in nmg.edges
    assert nmg.edges[(A, D)]['weight'] == 1
    assert nmg.edges[(B, D)]['weight'] == 1
    assert nmg.number_of_edges() == 5
    assert nmg.number_of_nodes() == 4

    # merge A and B
    infr.add_feedback((a1, b1), POSTV)
    assert nmg.number_of_edges() == 4
    assert nmg.number_of_nodes() == 3
    AB = infr.node_label(a1)
    assert nmg.edges[(AB, AB)]['weight'] == 2

    # split A and B
    # the number of nodes should increase, but the edges should stay the
    # same because we added an incmp edge
    infr.add_feedback((a1, b1), INCMP)
    assert nmg.number_of_edges() == 5
    assert nmg.number_of_nodes() == 4
    assert nmg.edges[(A, B)]['weight'] == 2
    infr.assert_neg_metagraph()

    # remove all positive edges
    for edge in list(infr.pos_graph.edges()):
        infr.add_feedback(edge, INCMP)

    # metagraph should not be isomorphic to infr.neg_graph
    assert nmg.number_of_nodes() == infr.neg_graph.number_of_nodes()
    assert nmg.number_of_edges() == infr.neg_graph.number_of_edges()
    infr.assert_neg_metagraph()
