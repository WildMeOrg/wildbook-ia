

def dyn():
    # ORIIGNALLY PART OF DYNAMIC APPLY
        # this older code attempts to update state based on the single edge
        # change. It doesn't quite work. It may be worth it to try and fix it
        # for efficiency.
        if False:
            # Most things we modify will belong to this subgraph
            # HOWEVER, we may also modify edges leaving the subgraph as well
            subgraph2 = infr.graph.subgraph(relevant_nodes)

            if state == 'match':
                cc1 = infr.get_annot_cc(n1)
                cc2 = cc1
                ccs = [cc1]
            else:
                cc1 = infr.get_annot_cc(n1)
                cc2 = infr.get_annot_cc(n2)
                ccs = [cc1, cc2]

            # Check for consistency
            # First remove previous split cases and inferred states
            nx.set_edge_attributes(infr.graph, 'maybe_error',
                                   _dz(subgraph2.edges(), [False]))
            nx.set_edge_attributes(infr.graph, 'is_cut',
                                   _dz(subgraph2.edges(), [False]))
            inconsistent = False
            for cc in ccs:
                cc_inconsistent = False
                _subgraph = infr.graph.subgraph(cc)
                for u, v, d in _subgraph.edges(data=True):
                    _state = d.get('reviewed_state', 'unreviewed')
                    if _state == 'nomatch':
                        cc_inconsistent = True
                        break
                if cc_inconsistent:
                    edge_to_state = nx.get_edge_attributes(_subgraph,
                                                           'reviewed_state')
                    # only pass in reviewed edges in the subgraph
                    keep_edges = [e for e, s in edge_to_state.items()
                                  if s != 'unreviewed']
                    split_subgraph = nx.Graph([e + (_subgraph.get_edge_data(*e),)
                                               for e in keep_edges])
                    # _subgraph.remove_edges_from(keep_edges)
                    error_edges = infr._find_possible_error_edges(split_subgraph)
                    nx.set_edge_attributes(infr.graph, 'maybe_error',
                                           _dz(error_edges, [True]))
                inconsistent |= cc_inconsistent

            # First remove all inferences in the subgraph so we can locally
            # reconstruct them based on the new information
            if inconsistent:
                nx.set_edge_attributes(infr.graph, 'inferred_state',
                                       _dz(subgraph2.edges(), [None]))

            # Update the match edges within each compoment?
            # If there are two compoments, update the inferred states
            # between them.
            if len(ccs) == 2:
                for u, v in infr.edges_between(*ccs):
                    pass

            # Update any inferred states
            if state == 'match':
                # Infer any unreviewed edge within a compoment as reviewed
                if not inconsistent:
                    inferred_state = {}
                    for u, v, d in subgraph2.edges(data=True):
                        _state = d.get('reviewed_state', 'unreviewed')
                        if _state in ['unreviewed', 'notcomp']:
                            inferred_state[(u, v)] = 'same'
                    nx.set_edge_attributes(infr.graph, 'inferred_state',
                                           inferred_state)
                # Remove all cut states from a matched compoment
                nx.set_edge_attributes(infr.graph, 'is_cut',
                                       _dz(subgraph2.edges(), [False]))

            # A -X- B: Grab any other no-match edges out of A and B
            # if any of those compoments connected to those non matching nodes
            # would match either A or B, those edges should be implicitly cut.
            for cc in ccs:
                is_cut = {}
                inferred_state = {}
                nomatch_nodes = set(ut.flatten(infr.get_nomatch_ccs(cc)))
                for u, v in infr.edges_between(cc, nomatch_nodes):
                    is_cut[(u, v)] = True
                    if not inconsistent:
                        d = infr.graph.get_edge_data(u, v)
                        if d.get('reviewed_state', 'unreviewed') in {'notcomp',
                                                                     'unreviewed'}:
                            inferred_state[(u, v)] = 'diff'
                nx.set_edge_attributes(infr.graph, 'is_cut', is_cut)
                if not inconsistent:
                    # print('inferred_state = %s' % (ut.repr4(inferred_state),))
                    nx.set_edge_attributes(infr.graph, 'inferred_state',
                                           inferred_state)

            # SLEDGE HAMMER METHOD
            # infr.apply_review_inference()
