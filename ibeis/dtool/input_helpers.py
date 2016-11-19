# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import six
import networkx as nx  # NOQA
(print, rrr, profile) = ut.inject2(__name__, '[depc_input_helpers]')


class BranchId(ut.HashComparable):
    def __init__(_id, accum_ids, k):
        _id.accum_ids = accum_ids
        # hack in multi-edge id
        _id.k = k

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
            # show the local_input_ids at the entire level
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
        # node_id_list = [v for u, v, k, d in edge_list[:-1]]
        # branch_id_list = [BranchId(d['accum_id'], k) for u, v, k, d in edge_list[:-1]]
        # exi_nodes = [NODE_TYPE(v, BRANCH_TYPE(*d['accum_id'])) for u, v, k, d in edge_list[:-1]]
        exi_nodes = [ExiNode(v, BranchId(d['accum_id'], k)) for u, v, k, d in edge_list[:-1]]
        # exi_nodes = [ExiNode(v, bid) for v, bid in zip(node_id_list,
        #                                                branch_id_list)]
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
        # TODO: testme to make sure I still work
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
        """
        Attempts to put the required inputs in a reasonable order
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
            >>> from dtool.example_depcache2 import *  # NOQA
            >>> depc = testdata_depc3()
            >>> exi_inputs1 = depc['vsone'].rootmost_inputs.total_expand()
            >>> print('exi_inputs1 = %r' % (exi_inputs1,))
            >>> exi_inputs2 = depc['neighbs'].rootmost_inputs.total_expand()
            >>> print('exi_inputs2 = %r' % (exi_inputs2,))
            >>> exi_inputs3 = depc['meta_labeler'].rootmost_inputs.total_expand()
            >>> print('exi_inputs3 = %r' % (exi_inputs3,))
            >>> ut.quit_if_noshow()
            >>> import plottool as pt
            >>> from plottool.interactions import ExpandableInteraction
            >>> inter = ExpandableInteraction(nCols=2)
            >>> depc['vsone'].show_dep_subgraph(inter)
            >>> exi_inputs1.show_exi_graph(inter)
            >>> depc['neighbs'].show_dep_subgraph(inter)
            >>> exi_inputs2.show_exi_graph(inter)
            >>> depc['meta_labeler'].show_dep_subgraph(inter)
            >>> exi_inputs3.show_exi_graph(inter)
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
            # FIXME: This should re-order based in the parent input specs
            # from the table.parents()
            if 1:
                import utool
                with utool.embed_on_exception_context:
                    # HACK: This only works with cases that exist so far.  Not
                    # sure what the general solution is. Too hungry to think
                    # about that.
                    # if sink_node.args[0] == 'vsone':
                    #     # hack to make vsone work
                    #     inputs.rmi_list = rmi_list
                    if len(inputs.rmi_list) > 1:
                        # I forgot what this hack fixes. maybe indexer?
                        rmi_list = inputs.rmi_list

                        # The second to last item in the computer order is the parent node
                        # FIXME; need to recursively get compute order for any
                        # set of inputs that resolve to the same node.
                        # if False:
                        #     current_table = [inputs.table for _ in range(len(rmi_list))]

                        #     reverse_compute_branches = [rmi.compute_order()[::-1] for rmi in rmi_list]
                        #     rcb = reverse_compute_branches

                        #     def recursive_group(rcb_):
                        #         print('---')
                        #         print('rcb_ = %r' % (rcb_,))
                        #         import utool as ut
                        #         current = ut.take_column(rcb_, slice(0, 2))
                        #         nexts = [c[1] if len(c) > 1 else c[0] for c in current]
                        #         nexts_ = [(p.args[0], p.args[1].k + 1) for p in nexts]
                        #         bases = ut.unique(ut.take_column(current, 0))
                        #         print('bases = %r' % (bases,))
                        #         assert len(bases) == 1
                        #         base = bases[0]

                        #         # order this leven
                        #         base_order = [(n, d.get('nwise_idx', 1))
                        #                       for n, d in inputs.table.depc[base.args[0]].parents(data=True)]

                        #         #
                        #         unique_, groupxs = ut.group_indices(nexts_)
                        #         level_sortx = ut.list_alignment(base_order, unique_)
                        #         unique_ = ut.take(unique_, level_sortx)
                        #         groupxs = ut.take(groupxs, level_sortx)
                        #         print('groupxs = %r' % (groupxs,))

                        #         suborders = []

                        #         for xs in groupxs:
                        #             if len(xs) == 1:
                        #                 suborder = [0]
                        #             else:
                        #                 next_rcbs = ut.take(rcb_, xs)
                        #                 next_rcbs = [n if len(n) == 1 else n[1:] for n in next_rcbs]
                        #                 print('next_rcbs = %r' % (next_rcbs,))
                        #                 suborder = recursive_group(next_rcbs)
                        #             suborders.append(suborder)
                        #         level_orders = [count + x for count, sub in enumerate(suborders) for x in sub]
                        #         return level_orders
                        #     ut.fix_embed_globals()
                        #     x = recursive_group(rcb)

                        #     compute_branches = [rmi.compute_order() for rmi in rmi_list]
                        #     pointers = [len(b) - 1 for b in compute_branches]
                        #     # move pointers up until every branch has a unique node
                        #     current_nodes = ut.ziptake(compute_branches, pointers)
                        #     dups = ut.find_duplicate_items(current_nodes)

                        #     # while dups:
                        #     max_iters = max(pointers)
                        #     for _ in range(max_iters):
                        #         for idx in ut.flatten(dups.values()):
                        #             if pointers[idx] > 0:
                        #                 pointers[idx] -= 1
                        #                 compute_branches[idx][pointers[idx]].args[0]
                        #                 current_table[idx].parents()
                        #         current_nodes = ut.ziptake(compute_branches, pointers)
                        #         dups = ut.find_duplicate_items(current_nodes)
                        #         if not dups:
                        #             break

                        #     upstream_rmis = current_nodes

                        if False:
                            parent_rmis = [rmi.compute_order()[-2]
                                           for rmi in rmi_list]
                            # parent_nodes = [parent_rmi.args[0] for parent_rmi in parent_rmis]

                            rmi_order = [(parent_rmi.args[0], parent_rmi.args[1].k + 1) for parent_rmi in parent_rmis]

                            # Target Order
                            target_order = [(n, d.get('nwise_idx', 1)) for n, d in inputs.table.parents(data=True)]

                            order_lookup = ut.make_index_lookup(target_order)

                            sortx = ut.take(order_lookup, rmi_order)
                            rmi_list = ut.take(rmi_list, sortx)
                            inputs.rmi_list = rmi_list

                        parent_nodes = [rmi.compute_order()[-2].args[0]
                                        for rmi in inputs.rmi_list]
                        order_lookup = ut.make_index_lookup(inputs.table.parents())
                        sortx = ut.take(order_lookup, parent_nodes)
                        inputs.rmi_list = ut.take(rmi_list, sortx)

                        # We need to map these parent rmis to the input order
                        # specified by the parent. This requires knowledge of
                        # local_input_ids / nwise_idx / parent_colx
                        # The k attribute of branch_id has been hacked in for
                        # this it may only work with vsone edge at the same
                        # level though...
                        # parent_rmis = [rmi.compute_order()[-2]
                        #                for rmi in rmi_list]
                        # parent_nodes = [parent_rmi.args[0] for parent_rmi in parent_rmis]
                        # print('parent_nodes = %r' % (parent_nodes,))
                        # order_rank1 = ut.unique_inverse(parent_nodes)[1]
                        # print('order_rank1 = %r' % (order_rank1,))
                        # order_rank2 = [parent_rmi.args[1].k for parent_rmi in parent_rmis]
                        # print('order_rank2 = %r' % (order_rank2,))
                        # inputs.rmi_list = ut.sortedby2(rmi_list, order_rank1, order_rank2)

                        # inputs.table.parents(data=True)
            else:
                # HACK: this fails
                rootmost_exi_nodes = list(rootmost_nodes)
                rootmost_depc_nodes = [node[0] for node in rootmost_exi_nodes]
                ranks = ut.nx_dag_node_rank(inputs.table.depc.graph,
                                            rootmost_depc_nodes)
                # make tiebreaker attribute
                ranks_breaker = ut.nx_dag_node_rank(inputs.exi_graph.reverse(),
                                                    rootmost_exi_nodes)
                sortx = ut.argsort(list(zip(ranks, [-x for x in ranks_breaker])))
                #sortx = ut.argsort(ranks)
                inputs.rmi_list = ut.take(rmi_list, sortx)
        else:
            flags = [x in rootmost_nodes for x in inputs.rmi_list]
            inputs.rmi_list = ut.compress(inputs.rmi_list, flags)
            pass

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
        rmi_list = [RootMostInput(node, sink, inputs.exi_graph)
                    for node in source_nodes]
        exi_graph = inputs.exi_graph
        table = inputs.table
        reorder = True
        new_inputs = TableInput(rmi_list, exi_graph, table,
                                reorder=reorder)
        return new_inputs

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
            index_list = ut.where([rmi.tablename == index
                                   for rmi in inputs.rmi_list])
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
            new_rmi_list = ut.insert_values(inputs.rmi_list, index,
                                            parent_level, inplace)
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
        flat_compute_order = ut.take(flat_node_order_, sortx)
        # Inputs are pre-computed.
        for rmi in inputs.rmi_list:
            try:
                flat_compute_order.remove(rmi.node)
            except ValueError as ex:
                ut.printex(ex, 'something is wrong', keys=['rmi.node'])
                raise
        return flat_compute_order

    def flat_compute_edges(inputs):
        """
        Defines order of computation that maps input_ids to target_ids.

        Returns:
            list: compute_edges
                Each item is a tuple in the form
                    ([parent_1, ..., parent_n], node_i)
                All parents should be known before you reach the i-th item in
                the list.
                Results of the the i-th item may be used in subsequent item
                computations.
        """
        flat_compute_order = inputs.flat_compute_order()
        compute_edges = []
        exi_graph = inputs.exi_graph
        for output_node in flat_compute_order:
            input_nodes = list(exi_graph.predecessors(output_node))
            edge = (input_nodes, output_node)
            compute_edges.append(edge)
        return compute_edges

    def flat_compute_rmi_edges(inputs):
        """ Wraps flat compute edges in RMI structure """
        sink = list(ut.nx_sink_nodes(inputs.exi_graph))[0]
        exi_graph = inputs.exi_graph
        compute_rmi_edges = []
        for input_nodes, output_node in inputs.flat_compute_edges():
            input_rmis = [RootMostInput(node, sink, exi_graph)
                          for node in input_nodes]
            output_rmis = RootMostInput(output_node, sink, exi_graph)
            edge = (input_rmis, output_rmis)
            compute_rmi_edges.append(edge)
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
            >>> #print(depc['smk_match'].rootmost_inputs.compute_order)
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
        >>> from dtool.input_helpers import *  # NOQA
        >>> from dtool.example_depcache2 import *  # NOQA
        >>> import plottool as pt
        >>> pt.ensure_pylab_qt4()
        >>> depc = testdata_depc3()
        >>> tablename = 'smk_match'
        >>> table = depc[tablename]
        >>> exi_graph = table.expanded_input_graph
        >>> inputs_ = get_rootmost_inputs(exi_graph, table)
        >>> print('inputs_ = %r' % (inputs_,))
        >>> inputs = inputs_.expand_input(1)
        >>> rmi = inputs.rmi_list[0]
        >>> result = ('inputs = %s' % (inputs,)) + '\n'
        >>> result += ('compute_order = %s' % (ut.repr2(inputs.flat_compute_edges(), nl=1)))
        >>> print(result)
    """
    # Take out the shallowest (wrt target) rootmost nodes
    # attrs = nx.get_node_attributes(exi_graph, 'rootmost')
    attrs = ut.nx_get_default_node_attributes(exi_graph, 'rootmost', False)
    rootmost_exi_nodes = [node for node, v in attrs.items() if v]
    sink = list(ut.nx_sink_nodes(exi_graph))[0]
    rmi_list = [RootMostInput(node, sink, exi_graph)
                for node in rootmost_exi_nodes]
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
