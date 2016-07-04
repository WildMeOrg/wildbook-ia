# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import six
import networkx as nx  # NOQA
(print, rrr, profile) = ut.inject2(__name__, '[depc_input_helpers]')


class BranchId(ut.HashComparable):
    def __init__(_id, *args):
        _id.args = args

    def __hash__(_id):
        return hash(_id.args)

    def __getitem__(_id, index):
        return _id.args[index]

    def __repr__(_id):
        return '[' + ', '.join(_id.args) + ']'


class ExiNode(ut.HashComparable):
    """ helps distinguish nodes and tuples """
    def __init__(node, *args):
        node.args = args

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

    CommandLine:
        python -m dtool.input_helpers make_expanded_input_graph --show

    Example:
        >>> from dtool.input_helpers import *  # NOQA
        >>> from dtool.example_depcache2 import * # NOQA
        >>> depc = testdata_depc3()
        >>> table = depc['smk_match']
        >>> graph = table.depc.explicit_graph.copy()
        >>> target = table.tablename
        >>> exi_graph = make_expanded_input_graph(graph, target)
        >>> x = list(exi_graph.nodes())[0]
        >>> print('x = %r' % (x,))
    """
    # FIXME: this does not work correctly when
    # The nesting of non-1-to-1 dependencies is greater than 2 (I think)
    # algorithm for finding inputs does not work.

    # FIXME: two vocabs have the same edge id, they should be the same in the
    # Expanded Input Graph as well. Their accum_id needs to be changed.

    NODE_TYPE = ExiNode
    BRANCH_TYPE = BranchId
    #BRANCH_TYPE = lambda *args: args
    #NODE_TYPE = lambda *args: args

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
        #if len(compressed) > 1 and compressed[0] in ['1', '*']:
        if len(compressed) > 1 and compressed[0] == '1':
            compressed = compressed[1:]
        compressed = tuple(compressed)
        return compressed

    BIG_HACK = True
    #BIG_HACK = False
    SMALL_HACK = True
    #SMALL_HACK = BIG_HACK or 1

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
        #uv_list = ut.take_column(edge_list, [0, 1])
        #v_list = ut.take_column(edge_list, [1])[:-1]
        #if target == 'vsone':
        #    import utool
        #    utool.embed()

        edge_data = ut.take_column(edge_list, 3)
        # We are accumulating local input ids
        toaccum_list_ = ut.dict_take_column(edge_data, 'local_input_id')
        if BIG_HACK and True:
            v_list = ut.take_column(edge_list, 1)
            pred_ids = ([
                [x['local_input_id'] for x in list(graph.pred[node].values())[0].values()]
                if len(graph.pred[node]) else []
                for node in v_list
            ])
            toaccum_list = [x + ':' + ';'.join(y) for x, y in zip(toaccum_list_, pred_ids)]
        #elif BIG_HACK:
        #    next_input_id = toaccum_list_[1:]
        #    next_input_id[-1] = '1'
        #    toaccum_list = next_input_id + ['1']
        #    #toaccum_list[0] = 't'
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
    assert len(sources) == 1
    source = sources[0]

    graph = graph.subgraph(ut.nx_all_nodes_between(graph, source, target))
    # Remove superfluous data
    ut.nx_delete_edge_attr(graph, ['edge_type', 'isnwise', 'nwise_idx',
                                   'parent_colx', 'ismulti'])

    # Hack multi edges stars to uniquely identify stars
    if SMALL_HACK:
        count = ord('a')
        for edge in graph.edges(keys=True, data=True):
            dat = edge[3]
            if dat['local_input_id'] == '*':
                dat['local_input_id'] = '*' + chr(count)
                dat['taillabel'] = '*' + chr(count)
                count += 1

    # Append dummy input/output nodes
    source_input = 'source_input'
    target_output = 'target_output'
    graph.add_edge(source_input, source, local_input_id='s', taillabel='1')
    graph.add_edge(target, target_output, local_input_id='t', taillabel='1')

    # Find all paths from the table to the source.
    paths_to_source   = ut.all_multi_paths(graph, source_input,
                                           target_output, data=True)

    accumulate_order = ut.reverse_path_edges
    #accumulate_order = ut.identity

    # Build expanded input graph
    # The inputs to this table can be derived from this graph.
    # The output is a new expanded input graph.
    exi_graph = graph.__class__()
    for path in paths_to_source:

        # Accumlate unique identifiers along the reversed(?) path
        edge_list = accumulate_order(path)
        accumulate_input_ids(edge_list)

        # A node's output(?) on this path determines its expanded id
        #exi_nodes = [tuple(v, d['accum_id']) for u, v, k, d in edge_list[:-1]]
        #exi_nodes = [NODE_TYPE(v, BRANCH_TYPE(*d['accum_id'])) for u, v, k, d in edge_list[:-1]]
        exi_nodes = [NODE_TYPE(v, BRANCH_TYPE(*d['accum_id'])) for u, v, k, d in edge_list[:-1]]
        # A node's input(?) on this path determines its expanded id
        #exi_nodes = [(u, d['accum_id']) for u, v, k, d in edge_list[1:]]

        exi_node_to_label = {
            # remove hacked in * ids
            #node: node[0] + '[' + ','.join([str(x)[0] for x in node[1]]) + ']'
            node: node[0] + '[' + ','.join([str(x) for x in node[1]]) + ']'
            for node in exi_nodes
        }
        exi_graph.add_nodes_from(exi_nodes)
        nx.set_node_attributes(exi_graph, 'label', exi_node_to_label)

        # Undo any accumulation ordering and remove dummy nodes
        old_edges = accumulate_order(edge_list[1:-1])
        new_edges = accumulate_order(list(ut.itertwo(exi_nodes)))
        for new_edge, old_edge in zip(new_edges, old_edges):
            u2, v2 = new_edge[:2]
            d = old_edge[3]
            taillabel = d['taillabel']
            if not exi_graph.has_edge(u2, v2):
                exi_graph.add_edge(u2, v2, taillabel=taillabel)

    sink_nodes = list(ut.nx_sink_nodes(exi_graph))
    source_nodes = list(ut.nx_source_nodes(exi_graph))
    try:
        assert len(sink_nodes) == 1, 'can only have one sink node'
    except AssertionError as ex:
        ut.printex(ex, iswarning=0)
        raise
        #if ut.SUPER_STRICT:
        #    raise
    sink_node = sink_nodes[0]

    # Color Rootmost inputs

    # First identify if a node is root_specifiable
    for node in exi_graph.nodes():
        root_specifiable = False
        for edge in exi_graph.in_edges(node, keys=True):
            edata = exi_graph.get_edge_data(*edge)
            if edata.get('taillabel').startswith('*'):
                if node != sink_node:
                    root_specifiable = True
        if exi_graph.in_degree(node) == 0:
            root_specifiable = True
        if root_specifiable:
            #exi_graph.node[node]['color'] = [1, .7, .6]
            exi_graph.node[node]['root_specifiable'] = True
        else:
            exi_graph.node[node]['root_specifiable'] = False

    # Need to specify any combo of red nodes such that
    # 1) for each path from a (leaf) to the (root) there is exactly one
    # red node along that path.
    # This garentees that all inputs are gievn.
    #path_list = [nx.shortest_path(exi_graph, source_node, sink_node)
    #             for source_node in source_nodes]
    path_list = ut.flatten([
        nx.all_simple_paths(exi_graph, source_node, sink_node)
        for source_node in source_nodes])
    rootmost_nodes = set([])
    for path in path_list:
        flags = [exi_graph.node[node]['root_specifiable'] for node in path]
        valid_nodes = ut.compress(path, flags)
        rootmost_nodes.add(valid_nodes[-1])
        #print('valid_nodes = %r' % (valid_nodes,))
    #print('rootmost_nodes = %r' % (rootmost_nodes,))
    # Rootmost nodes are the ones specifiable by default when computing
    # the normal property.
    for node in rootmost_nodes:
        #exi_graph.node[node]['color'] = [1, 0, 0]
        exi_graph.node[node]['rootmost'] = True

    recolor_exi_graph(exi_graph, rootmost_nodes)
    return exi_graph


def recolor_exi_graph(exi_graph, rootmost_nodes):
    for node in exi_graph.nodes():
        if exi_graph.node[node]['root_specifiable']:
            exi_graph.node[node]['color'] = [1, .7, .6]
    for node in rootmost_nodes:
        exi_graph.node[node]['color'] = [1, 0, 0]


#@ut.reloadable_class
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
        """
        def yield_condition(G, child, edge):
            return G.node[child].get('root_specifiable')
        def continue_condition(G, child, edge):
            return not G.node[child].get('root_specifiable')
        bfs_iter = ut.bfs_conditional(
            rmi.exi_graph, rmi.node, reverse=True, yield_nodes=True,
            yield_condition=yield_condition,
            continue_condition=continue_condition)
        parent_level = [RootMostInput(node, rmi.sink, rmi.exi_graph) for node in bfs_iter]
        return parent_level

    @property
    def ismulti(rmi):
        return any(['*' in x.split(':')[0] for x in rmi.input_id])

    def compute_order(rmi):
        """
        Returns order of computation from this input node to the sink
        """
        node_order = list(ut.nx_all_nodes_between(rmi.exi_graph, rmi.node, rmi.sink))
        node_rank = ut.nx_dag_node_rank(rmi.exi_graph.reverse(), node_order)
        sortx = ut.argsort(node_rank)[::-1]
        node_order = ut.take(node_order, sortx)
        return node_order

    def __hash__(rmi):
        return hash(rmi.node)

    def __repr__(rmi):
        return str(rmi.node)
        #rmi.tablename + '[' + ', '.join(rmi.input_id) + ']'

    __str__ = __repr__


@ut.reloadable_class
class TableInput(ut.NiceRepr):
    def __init__(inputs, rmi_list, exi_graph, table, reorder=False):
        # The order of the RMI list defines the expect input order
        inputs.rmi_list = rmi_list
        inputs.exi_graph = exi_graph
        inputs.table = table
        #if reorder:
        inputs._order_rmi_list(reorder)

    def _order_rmi_list(inputs, reorder=False):
        # hack for labels
        rmi_list = ut.unique(inputs.rmi_list)
        rootmost_exi_nodes = [rmi.node for rmi in rmi_list]

        # Ensure that nodes form a complete rootmost set
        # Remove over-complete nodes
        sink_nodes = list(ut.nx_sink_nodes(inputs.exi_graph))
        source_nodes = list(ut.nx_source_nodes(inputs.exi_graph))
        assert len(sink_nodes) == 1, 'can only have one sink node'
        sink_node = sink_nodes[0]
        path_list = ut.flatten([
            nx.all_simple_paths(inputs.exi_graph, source_node, sink_node)
            for source_node in source_nodes])
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
            rootmost_exi_nodes = list(rootmost_nodes)
            rootmost_depc_nodes = [node[0] for node in rootmost_exi_nodes]
            ranks = ut.nx_dag_node_rank(inputs.table.depc.graph, rootmost_depc_nodes)
            # make tiebreaker attribute
            ranks_breaker = ut.nx_dag_node_rank(inputs.exi_graph.reverse(), rootmost_exi_nodes)
            sortx = ut.argsort(list(zip(ranks, [-x for x in ranks_breaker])))
            #sortx = ut.argsort(ranks)
            inputs.rmi_list = ut.take(rmi_list, sortx)
        else:
            flags = [x in rootmost_nodes for x in inputs.rmi_list]
            inputs.rmi_list = ut.compress(inputs.rmi_list, flags)
            pass

    def __nice__(inputs):
        return ' ' + repr(inputs.rmi_list)

    def __len__(inputs):
        return len(inputs.rmi_list)

    def is_single_inputs(inputs):
        return len(inputs.rmi_list) == 0

    def expected_input_depth(inputs):
        """
        Example:
            >>> # DISABLE_DOCTEST
            >>> from dtool.input_helpers import *  # NOQA
            >>> from dtool.example_depcache2 import *  # NOQA
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
        new_inputs = TableInput(rmi_list, inputs.exi_graph, inputs.table, reorder=True)
        return new_inputs

        #pass

    def expand_input(inputs, index, inplace=False):
        """
        Example:
            >>> # ENABLE_DOCTEST
            >>> from dtool.input_helpers import *  # NOQA
            >>> from dtool.example_depcache2 import *  # NOQA
            >>> depc = testdata_depc4()
            >>> inputs = depc['smk_match'].rootmost_inputs
            >>> inputs = depc['neighbs'].rootmost_inputs
            >>> print('inputs = %r' % (inputs,))
            >>> index = 'indexer'
            >>> inputs = inputs.expand_input(index)
            >>> print('inputs = %r' % (inputs,))
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
            #raise AssertionError('no parents to expand')
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
        # Compute the order in which all noes must be evaluated
        import networkx as nx  # NOQA
        ordered_compute_nodes =  [rmi.compute_order() for rmi in inputs.rmi_list]
        flat_node_order_ = ut.unique(ut.flatten(ordered_compute_nodes))
        rgraph = inputs.exi_graph.reverse()
        toprank = ut.nx_topsort_rank(rgraph, flat_node_order_)
        sortx = ut.argsort(toprank)[::-1]
        #dagrank = ut.nx_dag_node_rank(rgraph, flat_node_order_)
        #sortx = ut.argsort(dagrank)[::-1]
        flat_compute_order = ut.take(flat_node_order_, sortx)
        # Inputs are pre-computed.
        for rmi in inputs.rmi_list:
            flat_compute_order.remove(rmi.node)
        return flat_compute_order

    def flat_compute_edges(inputs):
        """
        Returns:
            list: compute_edges
                Each item is a tuple in the form
                    ([parent_1, ..., parent_n], node_i)
                All parents should be known before you reach the i-th item in the list.
                Results of the the i-th item may be used in subsequent item computations.

        """
        flat_compute_order = inputs.flat_compute_order()
        compute_edges = [(list(inputs.exi_graph.predecessors(node)), node) for node in flat_compute_order]
        return compute_edges

    def flat_compute_rmi_edges(inputs):
        sink = list(ut.nx_sink_nodes(inputs.exi_graph))[0]

        compute_rmi_edges = [
            ([RootMostInput(node, sink, inputs.exi_graph) for node in nodes],
             RootMostInput(output_node, sink, inputs.exi_graph) )
            for nodes, output_node in inputs.flat_compute_edges()
        ]
        return compute_rmi_edges

    def flat_input_order(inputs):
        flat_input_order = [rmi.node for rmi in inputs.rmi_list]
        return flat_input_order

    def get_node_to_branch_ids(inputs):
        """
        Nodes may belong to several computation branches (paths)
        This returns a mapping from a node to each branch it belongs to
        """
        sources = ut.nx_source_nodes(inputs.exi_graph)
        sinks = ut.nx_sink_nodes(inputs.exi_graph)
        _node_branchid_pairs = [
            (s[1], node)
            for s, t in ut.product(sources, sinks)
            for node in ut.nx_all_nodes_between(inputs.exi_graph, s, t)
        ]
        branch_ids = ut.take_column(_node_branchid_pairs, 0)
        node_ids = ut.take_column(_node_branchid_pairs, 1)
        node_to_branchids_ = ut.group_items(branch_ids, node_ids)
        node_to_branchids = ut.map_dict_vals(tuple, node_to_branchids_)
        return node_to_branchids

    def get_input_branch_ids(inputs):
        """ Return what branches the inputs are used in """
        # Get node to branch-id mapping
        node_to_branchids = inputs.get_node_to_branch_ids()
        # Map input nodes to branch-ids
        exi_nodes = inputs.exi_nodes()
        rootmost_exi_branches = ut.dict_take(node_to_branchids, exi_nodes)
        rootmost_tables = ut.take_column(exi_nodes, 0)
        input_compute_ids = list(zip(rootmost_tables, rootmost_exi_branches))
        return input_compute_ids

    def get_compute_order(inputs):
        """
        Defines order of computation that maps input_ids to target_ids.

        Returns:
            list: compute_order
                Each item in the list is a tuple of the form:
                    (tablename_i, [branchid_1, ..., branchid_n])
                The order of the list indicates which table is computed first
                The table property must be computed for each branch specified
                by branch_id. before moving to the next item in compute_order
        """
        #node_to_branchids = inputs.get_node_to_branch_ids()

        #flat_compute_order = inputs.flat_compute_order()
        #branch_ids = ut.dict_take(node_to_branchids, flat_compute_order)
        #tablenames = ut.take_column(flat_compute_order, 0)
        #depend_compute_ids = list(zip(tablenames, branch_ids))

        # remove inputs that should be given
        #depend_compute_ids2 = depend_compute_ids[:]
        #input_compute_ids = inputs.get_input_branch_ids()
        #for w in input_compute_ids:
        #    depend_compute_ids2.remove(w)

        #compute_order = depend_compute_ids2
        return inputs.flat_compute_edges()

    def show_exi_graph(inputs, inter=None):
        """
        CommandLine:
            python -m dtool.input_helpers TableInput.show_exi_graph --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from dtool.input_helpers import *  # NOQA
            >>> from dtool.example_depcache2 import *  # NOQA
            >>> depc = testdata_depc3()
            >>> import plottool as pt
            >>> table = depc['smk_match']
            >>> inputs = table.rootmost_inputs
            >>> print('inputs = %r' % (inputs,))
            >>> from plottool.interactions import ExpandableInteraction
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
            >>> print(depc['smk_match'].compute_order)
            >>> ut.show_if_requested()
        """
        import plottool as pt
        from plottool.interactions import ExpandableInteraction
        autostart = inter is None
        if inter is None:
            inter = ExpandableInteraction()
        tablename = inputs.table.tablename

        exi_graph = inputs.exi_graph.copy()
        recolor_exi_graph(exi_graph, inputs.exi_nodes())
        for count, rmi in enumerate(inputs.rmi_list, start=0):
            if rmi.ismulti:
                exi_graph.node[rmi.node]['label'] += ' #%d*' % (count,)
            else:
                exi_graph.node[rmi.node]['label'] += ' #%d' % (count,)

        plot_kw = {'fontname': 'Ubuntu'}
        #inter.append_plot(
        #    ut.partial(pt.show_nx, G, title='Dependency Subgraph (%s)' % (tablename), **plot_kw))
        inter.append_plot(
            ut.partial(pt.show_nx, exi_graph, title='Expanded Input (%s)' % (tablename,), **plot_kw))
        if autostart:
            inter.start()
        return inter


def get_rootmost_inputs(exi_graph, table):
    """
    CommandLine:
        python -m dtool.input_helpers get_rootmost_inputs --show

    Example:
        >>> from dtool.input_helpers import *  # NOQA
        >>> from dtool.example_depcache2 import *  # NOQA
        >>> import plottool as pt
        >>> pt.ensure_pylab_qt4()
        >>> depc = testdata_depc3()
        >>> tablename = 'smk_match'
        >>> table = depc[tablename]
        >>> exi_graph = table.expanded_input_graph
        >>> inputs_ = get_rootmost_inputs(exi_graph, table)
        >>> inputs = inputs_.expand_input(1)
        >>> rmi = inputs.rmi_list[0]
        >>> result = ('inputs = %s' % (inputs,)) + '\n'
        >>> result += ('compute_order = %s' % (ut.repr2(inputs.get_compute_order(), nl=1)))
        >>> print(result)
    """
    # Take out the shallowest (wrt target) rootmost nodes
    attrs = ut.nx_get_default_node_attributes(exi_graph, 'rootmost', False)
    rootmost_exi_nodes = [node for node, v in attrs.items() if v]
    sink = list(ut.nx_sink_nodes(exi_graph))[0]
    rmi_list = [RootMostInput(node, sink, exi_graph) for node in rootmost_exi_nodes]
    inputs = TableInput(rmi_list, exi_graph, table, reorder=True)
    #x = inmputs.parent_level()[0].parent_level()[0]  # NOQA
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
