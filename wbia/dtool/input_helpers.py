# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import numpy as np
import six
import networkx as nx  # NOQA

(print, rrr, profile) = ut.inject2(__name__, '[depc_input_helpers]')


class BranchId(ut.HashComparable):
    def __init__(_id, accum_ids, k, parent_colx):
        _id.accum_ids = accum_ids
        # hack in multi-edge id
        _id.k = k
        _id.parent_colx = parent_colx

    def __hash__(_id):
        return hash(_id.accum_ids)

    def __getitem__(_id, index):
        return _id.accum_ids[index]

    def __repr__(_id):
        return '[' + ', '.join(_id.accum_ids) + ']'


class ExiNode(ut.HashComparable):
    """
    Expanded Input Node

    helps distinguish nodes and branch_ids
    """

    def __init__(node, node_id, branch_id):
        node.args = (node_id, branch_id)

    @property
    def node_id(node):
        return node.args[0]

    @property
    def branch_id(node):
        return node.args[1]

    def __hash__(node):
        return hash(node.args)

    def __getitem__(node, index):
        return node.args[index]

    def __repr__(node):
        tablename = node.args[0]
        id_ = '[' + ', '.join(node.args[1]) + ']'
        return tablename + id_

    __str__ = __repr__


def make_expanded_input_graph(graph, target):
    """
    Starting from the `target` property we trace all possible paths in the
    `graph` back to all sources.

    Args:
        graph (nx.DiMultiGraph): the dependency graph with a single source.
        target (str): a single target node in graph

    Notes:
        Each edge in the graph must have a `local_input_id` that defines the
        type of edge it is: (eg one-to-many, one-to-one, nwise/multi).

        # Step 1: Extracting the Relevant Subgraph
        We start by searching for all sources of the graph (we assume there is
        only one). Then we extract the subgraph defined by all edges between
        the sources and the target.  We augment this graph with a dummy super
        source `s` and super sink `t`. This allows us to associate an edge with
        the real source and sink.

        # Step 2: Trace all paths from `s` to `t`.
        Create a set of all paths from the source to the sink and accumulate
        the `local_input_id` of each edge along the path. This will uniquely
        identify each path. We use a hack to condense the accumualated ids in
        order to display them nicely.

        # Step 3: Create the new `exi_graph`
        Using the traced paths with ids we construct a new graph representing
        expanded inputs. The nodes in the original graph will be copied for each
        unique path that passes through the node. We identify these nodes using
        the accumulated ids built along the edges in our path set.  For each
        path starting from the target we add each node augmented with the
        accumulated ids on its output(?) edge. We also add the edges along
        these paths which results in the final `exi_graph`.

        # Step 4: Identify valid inputs candidates
        The purpose of this graph is to identify which inputs are needed
        to compute dependant properties. One valid set of inputs is all
        sources of the graph. However, sometimes it is preferable to specify
        a model that may have been trained from many inputs. Therefore any
        node with a one-to-many input edge may also be specified as an input.

        # Step 5: Identify root-most inputs
        The user will only specify one possible set of the inputs. We refer  to
        this set as the "root-most" inputs. This is a set of candiate nodes
        such that all paths from the sink to the super source are blocked.  We
        default to the set of inputs which results in the fewest dependency
        computations. However this is arbitary.

        The last step that is not represented here is to compute the order that
        the branches must be specified in when given to the depcache for a
        computation.

    Returns:
        nx.DiGraph: exi_graph: the expanded input graph

    Notes:
        All * nodes are defined to be distinct.
        TODO: To make a * node non-distinct it must be suffixed with an
        identifier.

    CommandLine:
        python -m dtool.input_helpers make_expanded_input_graph --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.dtool.input_helpers import *  # NOQA
        >>> from wbia.dtool.example_depcache2 import * # NOQA
        >>> depc = testdata_depc3()
        >>> table = depc['smk_match']
        >>> table = depc['vsone']
        >>> graph = table.depc.explicit_graph.copy()
        >>> target = table.tablename
        >>> exi_graph = make_expanded_input_graph(graph, target)
        >>> x = list(exi_graph.nodes())[0]
        >>> print('x = %r' % (x,))
        >>> # xdoctest: +REQUIRES(--show)
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> pt.show_nx(graph, fnum=1, pnum=(1, 2, 1))
        >>> pt.show_nx(exi_graph, fnum=1, pnum=(1, 2, 2))
        >>> ut.show_if_requested()
    """
    # FIXME: this does not work correctly when
    # The nesting of non-1-to-1 dependencies is greater than 2 (I think)
    # algorithm for finding inputs does not work.

    # FIXME: two vocabs have the same edge id, they should be the same in the
    # Expanded Input Graph as well. Their accum_id needs to be changed.

    def condense_accum_ids(rinput_path_id):
        # Hack to condense and consolidate graph sources
        prev = None
        compressed = []
        for item in rinput_path_id:
            if item == '1' and prev is not None:
                pass  # done append ones
            elif item != prev:
                compressed.append(item)
            prev = item
        # if len(compressed) > 1 and compressed[0] in ['1', '*']:
        if len(compressed) > 1 and compressed[0] == '1':
            compressed = compressed[1:]
        compressed = tuple(compressed)
        return compressed

    BIG_HACK = True
    # BIG_HACK = False

    def condense_accum_ids_stars(rinput_path_id):
        # Hack to condense and consolidate graph sources
        rcompressed = []
        has_star = False
        # Remove all but the final star (this is a really bad hack)
        for item in reversed(rinput_path_id):
            is_star = '*' in item
            if not (is_star and has_star):
                if not has_star:
                    rcompressed.append(item)
            has_star = has_star or is_star
        compressed = tuple(rcompressed[::-1])
        return compressed

    def accumulate_input_ids(edge_list):
        """
        python -m dtool.example_depcache2 testdata_depc4 --show
        """
        edge_data = ut.take_column(edge_list, 3)
        # We are accumulating local input ids
        toaccum_list_ = ut.dict_take_column(edge_data, 'local_input_id')
        if BIG_HACK and True:
            v_list = ut.take_column(edge_list, 1)
            # show the local_input_ids at the entire level
            pred_ids = [
                [x['local_input_id'] for x in list(graph.pred[node].values())[0].values()]
                if len(graph.pred[node])
                else []
                for node in v_list
            ]
            toaccum_list = [
                x + ':' + ';'.join(y) for x, y in zip(toaccum_list_, pred_ids)
            ]
        else:
            toaccum_list = toaccum_list_

        # Default dumb accumulation
        accum_ids_ = ut.cumsum(zip(toaccum_list), tuple())
        accum_ids = ut.lmap(condense_accum_ids, accum_ids_)
        if BIG_HACK:
            accum_ids = ut.lmap(condense_accum_ids_stars, accum_ids)
            accum_ids = [('t',) + x for x in accum_ids]
        ut.dict_set_column(edge_data, 'accum_id', accum_ids)
        return accum_ids

    sources = list(ut.nx_source_nodes(graph))
    print(sources)
    # assert len(sources) == 1, 'expected a unique source'
    source = sources[0]

    graph = graph.subgraph(ut.nx_all_nodes_between(graph, source, target)).copy()
    # Remove superfluous data
    ut.nx_delete_edge_attr(
        graph,
        [
            'edge_type',
            'isnwise',
            'nwise_idx',
            # 'parent_colx',
            'ismulti',
        ],
    )

    # Make all '*' edges have distinct local_input_id's.
    # TODO: allow non-distinct suffixes
    count = ord('a')
    for edge in graph.edges(keys=True, data=True):
        dat = edge[3]
        if dat['local_input_id'] == '*':
            dat['local_input_id'] = '*' + chr(count)
            dat['taillabel'] = '*' + chr(count)
            count += 1

    # Augment with dummy super source/sink nodes
    source_input = 'source_input'
    target_output = 'target_output'
    graph.add_edge(source_input, source, local_input_id='s', taillabel='1')
    graph.add_edge(target, target_output, local_input_id='t', taillabel='1')

    # Find all paths from the table to the source.
    paths_to_source = ut.all_multi_paths(graph, source_input, target_output, data=True)

    # Build expanded input graph
    # The inputs to this table can be derived from this graph.
    # The output is a new expanded input graph.
    exi_graph = nx.DiGraph()
    for path in paths_to_source:
        # Accumlate unique identifiers along the reversed path
        edge_list = ut.reverse_path_edges(path)
        accumulate_input_ids(edge_list)

        # A node's output(?) on this path determines its expanded branch id
        exi_nodes = [
            ExiNode(v, BranchId(d['accum_id'], k, d.get('parent_colx', -1)))
            for u, v, k, d in edge_list[:-1]
        ]
        exi_node_to_label = {
            node: node[0] + '[' + ','.join([str(x) for x in node[1]]) + ']'
            for node in exi_nodes
        }
        exi_graph.add_nodes_from(exi_nodes)
        nx.set_node_attributes(exi_graph, name='label', values=exi_node_to_label)

        # Undo any accumulation ordering and remove dummy nodes
        old_edges = ut.reverse_path_edges(edge_list[1:-1])
        new_edges = ut.reverse_path_edges(list(ut.itertwo(exi_nodes)))
        for new_edge, old_edge in zip(new_edges, old_edges):
            u2, v2 = new_edge[:2]
            d = old_edge[3]
            taillabel = d['taillabel']
            parent_colx = d.get('parent_colx', -1)
            if not exi_graph.has_edge(u2, v2):
                exi_graph.add_edge(u2, v2, taillabel=taillabel, parent_colx=parent_colx)

    sink_nodes = list(ut.nx_sink_nodes(exi_graph))
    source_nodes = list(ut.nx_source_nodes(exi_graph))
    assert len(sink_nodes) == 1, 'expected a unique sink'
    sink_node = sink_nodes[0]

    # First identify if a node is root_specifiable
    node_dict = ut.nx_node_dict(exi_graph)
    for node in exi_graph.nodes():
        root_specifiable = False
        # for edge in exi_graph.in_edges(node, keys=True):
        for edge in exi_graph.in_edges(node):
            # key = edge[-1]
            # assert key == 0, 'multi di graph is necessary'
            edata = exi_graph.get_edge_data(*edge)
            if edata.get('taillabel').startswith('*'):
                if node != sink_node:
                    root_specifiable = True
        if exi_graph.in_degree(node) == 0:
            root_specifiable = True
        node_dict[node]['root_specifiable'] = root_specifiable

    # Need to specify any combo of red nodes such that
    # 1) for each path from a (leaf) to the (root) there is exactly one red
    # node along that path.  This garentees that all inputs are gievn.
    path_list = ut.flatten(
        [
            nx.all_simple_paths(exi_graph, source_node, sink_node)
            for source_node in source_nodes
        ]
    )
    rootmost_nodes = set([])
    for path in path_list:
        flags = [node_dict[node]['root_specifiable'] for node in path]
        valid_nodes = ut.compress(path, flags)
        rootmost_nodes.add(valid_nodes[-1])
    # Rootmost nodes are the ones specifiable by default when computing the
    # normal property.
    for node in rootmost_nodes:
        node_dict[node]['rootmost'] = True

    # We actually need to hack away any root-most nodes that have another
    # rootmost node as the parent.  Otherwise, this would cause constraints in
    # what the user could specify as valid input combinations.
    # ie: specify a vocab and an index, but the index depends on the vocab.
    # this forces the user to specify the vocab that was the parent of the index
    # the user should either just specify the index and have the vocab inferred
    # or for now, we just dont allow this to happen.
    nx.get_node_attributes(exi_graph, 'rootmost')

    recolor_exi_graph(exi_graph, rootmost_nodes)
    return exi_graph


def recolor_exi_graph(exi_graph, rootmost_nodes):
    node_dict = ut.nx_node_dict(exi_graph)
    for node in exi_graph.nodes():
        if node_dict[node]['root_specifiable']:
            node_dict[node]['color'] = [1, 0.7, 0.6]
    for node in rootmost_nodes:
        node_dict[node]['color'] = [1, 0, 0]


# @ut.reloadable_class
class RootMostInput(ut.HashComparable):
    def __init__(rmi, node, sink, exi_graph):
        rmi.node = node
        rmi.sink = sink
        rmi.tablename = node[0]
        rmi.input_id = node[1]
        rmi.exi_graph = exi_graph

    def __getitem__(rmi, index):
        return rmi.node[index]

    def parent_level(rmi):
        """
        Returns rootmost inputs above this node

        Example:
            >>> from wbia.dtool.example_depcache2 import *  # NOQA
            >>> depc = testdata_depc4()
            >>> inputs = depc['smk_match'].rootmost_inputs
            >>> rmi = inputs.rmi_list[1]
            >>> assert len(rmi.parent_level()) == 2
        """

        def yield_if(G, child, edge):
            node_dict = ut.nx_node_dict(G)
            return node_dict[child].get('root_specifiable')

        def continue_if(G, child, edge):
            node_dict = ut.nx_node_dict(G)
            return not node_dict[child].get('root_specifiable')

        bfs_iter = ut.bfs_conditional(
            rmi.exi_graph,
            rmi.node,
            reverse=True,
            yield_if=yield_if,
            continue_if=continue_if,
            yield_source=False,
            yield_nodes=True,
        )
        bfs_iter = list(bfs_iter)
        parent_level = [RootMostInput(node, rmi.sink, rmi.exi_graph) for node in bfs_iter]
        return parent_level

    @property
    def ismulti(rmi):
        return any(['*' in x.split(':')[0] for x in rmi.input_id])

    def compute_order(rmi):
        """
        Returns order of computation from this input node to the sink
        """
        # graph, source, target = rmi.exi_graph, rmi.node, rmi.sink
        node_order_ = list(ut.nx_all_nodes_between(rmi.exi_graph, rmi.node, rmi.sink))
        node_rank = ut.nx_dag_node_rank(rmi.exi_graph.reverse(), node_order_)
        node_names = list(map(str, node_order_))
        # lexsort via names to break ties for consistent ordering
        sortx = ut.argsort(node_rank, node_names)[::-1]
        # sortx = ut.argsort(node_rank)[::-1]
        node_order = ut.take(node_order_, sortx)
        return node_order

    def __hash__(rmi):
        return hash(rmi.node)

    def __repr__(rmi):
        return str(rmi.node)
        # rmi.tablename + '[' + ', '.join(rmi.input_id) + ']'

    __str__ = __repr__


def sort_rmi_list(rmi_list):
    """
    CommandLine:
        python -m dtool.input_helpers sort_rmi_list

    Example:
        >>> from wbia.dtool.input_helpers import *  # NOQA
        >>> from wbia.dtool.example_depcache2 import *  # NOQA
        >>> depc =testdata_custom_annot_depc([
        ...    dict(tablename='Notch_Tips', parents=['annot']),
        ...    dict(tablename='chips', parents=['annot']),
        ...    dict(tablename='Cropped_Chips', parents=['chips', 'Notch_Tips']),
        ... ])
        >>> table = depc['Cropped_Chips']
        >>> inputs = exi_inputs = table.rootmost_inputs
        >>> compute_rmi_edges = exi_inputs.flat_compute_rmi_edges()
        >>> input_rmis = compute_rmi_edges[-1][0]
        >>> rmi_list = input_rmis[::-1]
        >>> rmi_list = sort_rmi_list(rmi_list)
        >>> assert rmi_list[0].node[0] == 'chips'
    """
    # Order the input rmis via declaration
    reverse_compute_branches = [rmi.compute_order()[::-1] for rmi in rmi_list]
    # print('rmi_list = %r' % (rmi_list,))
    # rmi = rmi_list[0]  # hack
    # reverse_compute_branches = [path[::-1] for path in nx.all_simple_paths(rmi.exi_graph, rmi.node, rmi.sink)]
    sort_keys = [
        tuple([r.branch_id.parent_colx for r in rs]) for rs in reverse_compute_branches
    ]
    sortx = ut.argsort(sort_keys)
    rmi_list = ut.take(rmi_list, sortx)
    return rmi_list


@ut.reloadable_class
class TableInput(ut.NiceRepr):
    """
    Specifies a set of inputs that can validly compute the output of a table in
    the dependency graph
    """

    def __init__(inputs, rmi_list, exi_graph, table, reorder=False):
        # The order of the RMI list defines the expect input order
        inputs.rmi_list = rmi_list
        inputs.exi_graph = exi_graph
        inputs.table = table
        # if reorder:
        inputs._order_rmi_list(reorder)

    def _order_rmi_list(inputs, reorder=False):
        """
        Attempts to put the required inputs in the correct order as specified
        by the order of declared dependencies the user specified during the
        depcache declaration (in the user defined decorators).
        for 1-to-1 properties this is just the root_ids.

        For vsone, it should be root1, root2
        For vsmany it should be root1, root2*

        Ok, here is the measure:
        Order is primarily determined by your parent input order as given in
        the table definition. If one parent expands in to multiple parents then
        the secondary ordering inherits from the parents. If the two paths
        merge, then there is no problem. There is only one parent.

        CommandLine:
            python -m dtool.input_helpers _order_rmi_list --show

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.dtool.example_depcache2 import *  # NOQA
            >>> depc = testdata_depc3()
            >>> exi_inputs1 = depc['vsone'].rootmost_inputs.total_expand()
            >>> assert exi_inputs1.rmi_list[0] != exi_inputs1.rmi_list[1]
            >>> print('exi_inputs1 = %r' % (exi_inputs1,))
            >>> exi_inputs2 = depc['neighbs'].rootmost_inputs.total_expand()
            >>> assert '*' not in str(exi_inputs2.rmi_list[0])
            >>> assert '*' in str(exi_inputs2.rmi_list[1])
            >>> print('exi_inputs2 = %r' % (exi_inputs2,))
            >>> exi_inputs3 = depc['meta_labeler'].rootmost_inputs.total_expand()
            >>> print('exi_inputs3 = %r' % (exi_inputs3,))
            >>> exi_inputs4 = depc['smk_match'].rootmost_inputs.total_expand()
            >>> print('exi_inputs4 = %r' % (exi_inputs4,))
            >>> # xdoctest: +REQUIRES(--show)
            >>> ut.quit_if_noshow()
            >>> import wbia.plottool as pt
            >>> from wbia.plottool.interactions import ExpandableInteraction
            >>> inter = ExpandableInteraction(nCols=2)
            >>> depc['vsone'].show_dep_subgraph(inter)
            >>> exi_inputs1.show_exi_graph(inter)
            >>> depc['neighbs'].show_dep_subgraph(inter)
            >>> exi_inputs2.show_exi_graph(inter)
            >>> depc['meta_labeler'].show_dep_subgraph(inter)
            >>> exi_inputs3.show_exi_graph(inter)
            >>> depc['smk_match'].show_dep_subgraph(inter)
            >>> exi_inputs4.show_exi_graph(inter)
            >>> inter.start()
            >>> #depc['viewpoint_classification'].show_input_graph()
            >>> ut.show_if_requested()
        """
        # hack for labels
        rmi_list = ut.unique(inputs.rmi_list)
        rootmost_exi_nodes = [rmi.node for rmi in rmi_list]

        # Ensure that nodes form a complete rootmost set
        # Remove over-complete nodes
        sink_nodes = list(ut.nx_sink_nodes(inputs.exi_graph))
        source_nodes = list(ut.nx_source_nodes(inputs.exi_graph))
        assert len(sink_nodes) == 1, 'can only have one sink node'
        sink_node = sink_nodes[0]
        path_list = ut.flatten(
            [
                nx.all_simple_paths(inputs.exi_graph, source_node, sink_node)
                for source_node in source_nodes
            ]
        )
        rootmost_nodes = set([])
        rootmost_candidates = set(rootmost_exi_nodes)
        rootmost_nodes = set([])
        for path in path_list:
            flags = [node in rootmost_candidates for node in path]
            if not any(flags):
                raise ValueError('Missing RMI on path=%r' % (path,))
            valid_nodes = ut.compress(path, flags)
            rootmost_nodes.add(valid_nodes[-1])

        if reorder:
            # This re-orders the parent input specs based on the declared order
            # input defined by the user. This ordering is represented by the
            # parent_colx property from the table.parents()
            if len(inputs.rmi_list) > 1:
                inputs.rmi_list = sort_rmi_list(inputs.rmi_list)
        else:
            flags = [x in rootmost_nodes for x in inputs.rmi_list]
            inputs.rmi_list = ut.compress(inputs.rmi_list, flags)

    def __nice__(inputs):
        return repr(inputs.rmi_list)

    def __len__(inputs):
        return len(inputs.rmi_list)

    def is_single_inputs(inputs):
        return len(inputs.rmi_list) == 0

    def expected_input_depth(inputs):
        """
        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.dtool.input_helpers import *  # NOQA
            >>> from wbia.dtool.example_depcache2 import *  # NOQA
            >>> depc = testdata_depc4()
            >>> inputs = depc['neighbs'].rootmost_inputs
            >>> index = 'indexer'
            >>> inputs = inputs.expand_input(index)
            >>> size = inputs.expected_input_depth()
            >>> print('size = %r' % (size,))
            >>> inputs = depc['feat'].rootmost_inputs
            >>> size = inputs.expected_input_depth()
            >>> print('size = %r' % (size,))
        """
        return [0 if not rmi.ismulti else 1 for rmi in inputs.rmi_list]

    def total_expand(inputs):
        source_nodes = list(ut.nx_source_nodes(inputs.exi_graph))
        sink = list(ut.nx_sink_nodes(inputs.exi_graph))[0]
        rmi_list = [RootMostInput(node, sink, inputs.exi_graph) for node in source_nodes]
        exi_graph = inputs.exi_graph
        table = inputs.table
        reorder = True
        new_inputs = TableInput(rmi_list, exi_graph, table, reorder=reorder)
        return new_inputs

    def expand_input(inputs, index, inplace=False):
        """
        Pushes the rootmost inputs all the way up to the sources of the graph

        CommandLine:
            python -m dtool.input_helpers expand_input

        Example:
            >>> # ENABLE_DOCTEST
            >>> from wbia.dtool.input_helpers import *  # NOQA
            >>> from wbia.dtool.example_depcache2 import *  # NOQA
            >>> depc = testdata_depc4()
            >>> inputs = depc['smk_match'].rootmost_inputs
            >>> inputs = depc['neighbs'].rootmost_inputs
            >>> print('(pre-expand)  inputs  = %r' % (inputs,))
            >>> index = 'indexer'
            >>> inputs2 = inputs.expand_input(index)
            >>> print('(post-expand) inputs2 = %r' % (inputs2,))
            >>> assert 'indexer' in str(inputs), 'missing indexer1'
            >>> assert 'indexer' not in str(inputs2), (
            >>>     '(2) unexpected indexer in %s' % (inputs2,))
        """
        if isinstance(index, six.string_types):
            index_list = ut.where([rmi.tablename == index for rmi in inputs.rmi_list])
            if len(index_list) == 0:
                index = 0
            else:
                index = index_list[0]

        rmi = inputs.rmi_list[index]
        parent_level = rmi.parent_level()
        if len(parent_level) == 0:
            # raise AssertionError('no parents to expand')
            new_rmi_list = inputs.rmi_list[:]
        else:
            new_rmi_list = ut.insert_values(inputs.rmi_list, index, parent_level, inplace)
            new_rmi_list = ut.unique(new_rmi_list)
        if inplace:
            inputs.rmi_list = new_rmi_list
            new_inputs = inputs
        else:
            new_inputs = TableInput(new_rmi_list, inputs.exi_graph, inputs.table)
        return new_inputs

    def exi_nodes(inputs):
        return [rmi.node for rmi in inputs.rmi_list]

    def flat_compute_order(inputs):
        """
        This is basically the scheduler

        TODO:
            We need to verify the correctness of this logic. It seems to
            not be deterministic between versions of python.

        CommandLine:
            python -m dtool.input_helpers flat_compute_order

        Example:
            >>> # xdoctest: +REQUIRES(--fixme)
            >>> from wbia.dtool.input_helpers import *  # NOQA
            >>> from wbia.dtool.example_depcache2 import *  # NOQA
            >>> depc = testdata_depc4()
            >>> inputs = depc['feat'].rootmost_inputs.total_expand()
            >>> flat_compute_order = inputs.flat_compute_order()
            >>> result = ut.repr2(flat_compute_order)
            ...
            >>> print(result)
            [chip[t, t:1, 1:1], probchip[t, t:1, 1:1], feat[t, t:1]]
        """
        # Compute the order in which all noes must be evaluated
        import networkx as nx  # NOQA

        ordered_compute_nodes = [rmi.compute_order() for rmi in inputs.rmi_list]
        flat_node_order_ = ut.unique(ut.flatten(ordered_compute_nodes))

        rgraph = inputs.exi_graph.reverse()
        toprank = ut.nx_topsort_rank(rgraph, flat_node_order_)
        sortx = ut.argsort(toprank)[::-1]
        flat_compute_order = ut.take(flat_node_order_, sortx)
        # Inputs are pre-computed.
        for rmi in inputs.rmi_list:
            try:
                flat_compute_order.remove(rmi.node)
            except ValueError as ex:
                ut.printex(ex, 'something is wrong', keys=['rmi.node'])
                raise
        return flat_compute_order

    def flat_compute_rmi_edges(inputs):
        """
        Defines order of computation that maps input_ids to target_ids.

        CommandLine:
            python -m dtool.input_helpers flat_compute_rmi_edges

        Returns:
            list: compute_edges
                Each item is a tuple of input/output RootMostInputs
                    ([parent_1, ..., parent_n], node_i)
                All parents should be known before you reach the i-th item in
                the list.
                Results of the the i-th item may be used in subsequent item
                computations.

        Example:
            >>> from wbia.dtool.input_helpers import *  # NOQA
            >>> from wbia.dtool.example_depcache2 import *  # NOQA
            >>> depc =testdata_custom_annot_depc([
            ...    dict(tablename='chips', parents=['annot']),
            ...    dict(tablename='Notch_Tips', parents=['annot']),
            ...    dict(tablename='Cropped_Chips', parents=['chips', 'Notch_Tips']),
            ... ])
            >>> table = depc['Cropped_Chips']
            >>> inputs = exi_inputs = table.rootmost_inputs.total_expand()
            >>> compute_rmi_edges = exi_inputs.flat_compute_rmi_edges()
            >>> input_rmis = compute_rmi_edges[-1][0]
            >>> result = ut.repr2(input_rmis)
            >>> print(result)
            [chips[t, t:1, 1:1], Notch_Tips[t, t:1, 1:1]]
        """
        sink = list(ut.nx_sink_nodes(inputs.exi_graph))[0]
        exi_graph = inputs.exi_graph
        compute_rmi_edges = []

        flat_compute_order = inputs.flat_compute_order()
        exi_graph = inputs.exi_graph
        for output_node in flat_compute_order:
            input_nodes = list(exi_graph.predecessors(output_node))
            # input_edges = [(node, output_node, node.branch_id.k) for node in input_nodes]
            input_edges = [(node, output_node) for node in input_nodes]

            # another sorting strategy. maybe this is correct.
            sortx = [exi_graph.get_edge_data(*e).get('parent_colx') for e in input_edges]
            sortx_ = np.argsort(sortx)
            input_nodes = ut.take(input_nodes, sortx_)

            input_rmis = [RootMostInput(node, sink, exi_graph) for node in input_nodes]

            # input_rmis = sort_rmi_list(input_rmis)

            output_rmis = RootMostInput(output_node, sink, exi_graph)
            edge = (input_rmis, output_rmis)
            compute_rmi_edges.append(edge)
        return compute_rmi_edges

    # def get_node_to_branch_ids(inputs):
    #     """
    #     Nodes may belong to several computation branches (paths)
    #     This returns a mapping from a node to each branch it belongs to
    #     """
    #     sources = ut.nx_source_nodes(inputs.exi_graph)
    #     sinks = ut.nx_sink_nodes(inputs.exi_graph)
    #     _node_branchid_pairs = [
    #         (s[1], node)
    #         for s, t in ut.product(sources, sinks)
    #         for node in ut.nx_all_nodes_between(inputs.exi_graph, s, t)
    #     ]
    #     branch_ids = ut.take_column(_node_branchid_pairs, 0)
    #     node_ids = ut.take_column(_node_branchid_pairs, 1)
    #     node_to_branchids_ = ut.group_items(branch_ids, node_ids)
    #     node_to_branchids = ut.map_dict_vals(tuple, node_to_branchids_)
    #     return node_to_branchids

    # def get_input_branch_ids(inputs):
    #     """ Return what branches the inputs are used in """
    #     # Get node to branch-id mapping
    #     node_to_branchids = inputs.get_node_to_branch_ids()
    #     # Map input nodes to branch-ids
    #     exi_nodes = inputs.exi_nodes()
    #     rootmost_exi_branches = ut.dict_take(node_to_branchids, exi_nodes)
    #     rootmost_tables = ut.take_column(exi_nodes, 0)
    #     input_compute_ids = list(zip(rootmost_tables, rootmost_exi_branches))
    #     return input_compute_ids

    def show_exi_graph(inputs, inter=None):
        """
        CommandLine:
            python -m dtool.input_helpers TableInput.show_exi_graph --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from wbia.dtool.input_helpers import *  # NOQA
            >>> from wbia.dtool.example_depcache2 import *  # NOQA
            >>> depc = testdata_depc3()
            >>> # table = depc['smk_match']
            >>> table = depc['neighbs']
            >>> inputs = table.rootmost_inputs
            >>> print('inputs = %r' % (inputs,))
            >>> import wbia.plottool as pt
            >>> from wbia.plottool.interactions import ExpandableInteraction
            >>> inter = ExpandableInteraction(nCols=1)
            >>> inputs.show_exi_graph(inter=inter)
            >>> # FIXME; Expanding inputs can overspecify inputs
            >>> #inputs = inputs.expand_input(2)
            >>> #print('inputs = %r' % (inputs,))
            >>> #inputs.show_exi_graph(inter=inter)
            >>> #inputs = inputs.expand_input(1)
            >>> #inputs = inputs.expand_input(3)
            >>> #inputs = inputs.expand_input(2)
            >>> #inputs = inputs.expand_input(2)
            >>> #inputs = inputs.expand_input(1)
            >>> #print('inputs = %r' % (inputs,))
            >>> #inputs.show_exi_graph(inter=inter)
            >>> inter.start()
            >>> ut.show_if_requested()
        """
        import wbia.plottool as pt
        from wbia.plottool.interactions import ExpandableInteraction

        autostart = inter is None
        if inter is None:
            inter = ExpandableInteraction()
        tablename = inputs.table.tablename

        exi_graph = inputs.exi_graph.copy()
        recolor_exi_graph(exi_graph, inputs.exi_nodes())

        # Add numbering to indicate the input order
        node_dict = ut.nx_node_dict(exi_graph)
        for count, rmi in enumerate(inputs.rmi_list, start=0):
            if rmi.ismulti:
                node_dict[rmi.node]['label'] += ' #%d*' % (count,)
            else:
                node_dict[rmi.node]['label'] += ' #%d' % (count,)

        plot_kw = {'fontname': 'Ubuntu'}
        # inter.append_plot(
        #    ut.partial(pt.show_nx, G, title='Dependency Subgraph (%s)' % (tablename), **plot_kw))
        inter.append_plot(
            ut.partial(
                pt.show_nx,
                exi_graph,
                title='Expanded Input (%s)' % (tablename,),
                **plot_kw,
            )
        )
        if autostart:
            inter.start()
        return inter


def get_rootmost_inputs(exi_graph, table):
    r"""
    CommandLine:
        python -m dtool.input_helpers get_rootmost_inputs --show

    Args:
        exi_graph (nx.Graph): made from make_expanded_input_graph(graph, target)
        table (dtool.Table):

    CommandLine:
        python -m dtool.input_helpers get_rootmost_inputs

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.dtool.input_helpers import *  # NOQA
        >>> from wbia.dtool.example_depcache2 import *  # NOQA
        >>> depc = testdata_depc3()
        >>> tablename = 'smk_match'
        >>> table = depc[tablename]
        >>> exi_graph = table.expanded_input_graph
        >>> inputs_ = get_rootmost_inputs(exi_graph, table)
        >>> print('inputs_ = %r' % (inputs_,))
        >>> inputs = inputs_.expand_input(1)
        >>> rmi = inputs.rmi_list[0]
        >>> result = ('inputs = %s' % (inputs,)) + '\n'
        >>> result += ('compute_edges = %s' % (ut.repr2(inputs.flat_compute_rmi_edges(), nl=1)))
        >>> print(result)
    """
    # Take out the shallowest (wrt target) rootmost nodes
    # attrs = nx.get_node_attributes(exi_graph, 'rootmost')
    attrs = ut.nx_get_default_node_attributes(exi_graph, 'rootmost', False)
    rootmost_exi_nodes = [node for node, v in attrs.items() if v]
    sink = list(ut.nx_sink_nodes(exi_graph))[0]
    rmi_list = [RootMostInput(node, sink, exi_graph) for node in rootmost_exi_nodes]
    inputs = TableInput(rmi_list, exi_graph, table, reorder=True)
    # x = inmputs.parent_level()[0].parent_level()[0]  # NOQA
    return inputs


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m dtool.input_helpers
        python -m dtool.input_helpers --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
