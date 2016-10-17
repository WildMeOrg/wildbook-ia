
    def mst_review(infr):
        """
        Adds implicit reviews to connect all ndoes with the same name label
        """
        if infr.verbose:
            print('[infr] ensure_mst')
        import networkx as nx
        # Find clusters by labels
        node2_label = nx.get_node_attrs(infr.graph, 'name_label')
        label2_nodes = ut.group_items(node2_label.keys(), node2_label.values())

        aug_graph = infr.graph.copy().to_undirected()

        # remove cut edges
        edge_to_iscut = nx.get_edge_attrs(aug_graph, 'is_cut')
        cut_edges = [edge for edge, flag in edge_to_iscut.items() if flag]
        aug_graph.remove_edges_from(cut_edges)

        # Enumerate chains inside labels
        unflat_edges = [list(ut.itertwo(nodes)) for nodes in label2_nodes.values()]
        node_pairs = [tup for tup in ut.iflatten(unflat_edges) if tup[0] != tup[1]]

        # Remove candidate MST edges that exist in the original graph
        orig_edges = list(aug_graph.edges())
        candidate_mst_edges = [edge for edge in node_pairs
                               if not aug_graph.has_edge(*edge)]

        aug_graph.add_edges_from(candidate_mst_edges)
        # Weight edges in aug_graph such that existing edges are chosen
        # to be part of the MST first before suplementary edges.

        def get_edge_mst_weights(edge):
            state = aug_graph.get_edge_data(*edge).get('reviewed_state',
                                                       'unreviewed')
            is_mst = aug_graph.get_edge_data(*edge).get('_mst_edge', False)
            normscore = aug_graph.get_edge_data(*edge).get('normscore', 0)

            if state == 'match':
                # favor reviewed edges
                weight = .01
            else:
                # faveor states with high scores
                weight = 1 + (1 - normscore)
            if is_mst:
                # try to not use mst edges
                weight += 3.0
            return weight

        rng = np.random.RandomState(42)

        nx.set_edge_attributes(aug_graph, 'weight',
                               {edge: get_edge_mst_weights(edge)
                                for edge in orig_edges})
        nx.set_edge_attributes(aug_graph, 'weight',
                               {edge: 10.0 + rng.randint(1, 100)
                                for edge in candidate_mst_edges})
        new_mst_edges = []
        for cc_sub_graph in nx.connected_component_subgraphs(aug_graph):
            mst_sub_graph = nx.minimum_spanning_tree(cc_sub_graph)
            for edge in mst_sub_graph.edges():
                data = aug_graph.get_edge_data(*edge)
                state = data.get('reviewed_state', 'unreviewed')
                # Append only if this edge needs a review flag
                if state != 'match':
                    new_mst_edges.append(edge)

        if infr.verbose:
            print('[infr] reviewing %d MST edges' % (len(new_mst_edges)))

        # Apply data / add edges if needed
        graph = infr.graph
        for edge in new_mst_edges:
            redge = edge[::-1]
            # Only add if this edge is not in the original graph
            if graph.has_edge(*edge):
                nx.set_edge_attrs(graph, 'reviewed_state', {edge: 'match'})
                infr.add_feedback(edge[0], edge[1], 'match')
            elif graph.has_edge(*redge):
                nx.set_edge_attrs(graph, 'reviewed_state', {redge: 'match'})
                infr.add_feedback(edge[0], edge[1], 'match')
            else:
                graph.add_edge(*edge, attr_dict={
                    '_mst_edge': True, 'reviewed_state': 'match'})
                infr.add_feedback(edge[0], edge[1], 'match')



    def augment_name_nodes(infr):
        raise NotImplementedError('do not use')
        # If we want to represent name nodes in the graph
        name_graph = infr.graph.copy()
        #infr.qreq_.dnid_list
        #infr.qreq_.daid_list
        daids = infr.qreq_.daids
        dnids = infr.qreq_.get_qreq_annot_nids(daids)
        unique_dnids = ut.unique(dnids)
        dname_nodes = [('nid', nid) for nid in unique_dnids]
        name_graph.add_nodes_from(dname_nodes)
        nx.set_node_attributes(name_graph, 'nid', _dz(dname_nodes, unique_dnids))

        node_to_nid = nx.get_node_attrs(name_graph, 'nid')
        nid_to_node = ut.invert_dict(node_to_nid)

        dannot_nodes = ut.take(infr.aid_to_node, daids)
        dname_nodes = ut.take(nid_to_node, dnids)
        name_graph.add_edges_from(zip(dannot_nodes, dname_nodes))

        #graph = infr.graph
        graph = name_graph
        nx.set_node_attrs(name_graph, 'name_label', node_to_nid)
        infr.initialize_visual_node_attrs(graph)
        nx.set_node_attrs(graph, 'shape', _dz(dname_nodes, ['circle']))
        infr.update_visual_attrs(graph=name_graph, show_cuts=False)
        namenode_to_label = {
            node: 'nid=%r' % (nid,)
            for node, nid in node_to_nid.items()
        }
        nx.set_node_attributes(name_graph, 'label', namenode_to_label)
        pt.show_nx(graph, layout='custom', as_directed=False, modify_ax=False,
                   use_image=False, verbose=0)
        pt.zoom_factory()
        pt.pan_factory(pt.gca())
        #dannot_nodes = ut.take(infr.aid_to_node, dnids)
        pass

    #def remove_cuts(infr):
    #    """
    #    Undo all cuts HACK
    #    """
    #    if infr.verbose:
    #        print('[infr] apply_cuts')
    #    graph = infr.graph
    #    infr.ensure_mst()
    #    ut.nx_delete_edge_attr(graph, 'is_cut')
