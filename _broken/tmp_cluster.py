def flow():
    """
    http://pmneila.github.io/PyMaxflow/maxflow.html#maxflow-fastmin

    pip install PyMaxFlow
    pip install pystruct
    pip install hdbscan
    """
    # Toy problem representing attempting to discover names via annotation
    # scores

    import pystruct  # NOQA
    import pystruct.models  # NOQA
    import networkx as netx  # NOQA

    import vtool_ibeis as vt
    num_annots = 10
    num_names = num_annots
    hidden_nids = np.random.randint(0, num_names, num_annots)
    unique_nids, groupxs = vt.group_indices(hidden_nids)

    toy_params = {
        True: {'mu': 1.0, 'sigma': 2.2},
        False: {'mu': 7.0, 'sigma': .9}
    }

    if True:
        import plottool_ibeis as pt
        xdata = np.linspace(0, 100, 1000)
        tp_pdf = vt.gauss_func1d(xdata, **toy_params[True])
        fp_pdf = vt.gauss_func1d(xdata, **toy_params[False])
        pt.plot_probabilities([tp_pdf, fp_pdf], ['TP', 'TF'], xdata=xdata)

    def metric(aidx1, aidx2, hidden_nids=hidden_nids, toy_params=toy_params):
        if aidx1 == aidx2:
            return 0
        rng = np.random.RandomState(int(aidx1 + aidx2))
        same = hidden_nids[int(aidx1)] == hidden_nids[int(aidx2)]
        mu, sigma = ut.dict_take(toy_params[same], ['mu', 'sigma'])
        return np.clip(rng.normal(mu, sigma), 0, np.inf)

    pairwise_aidxs = list(ut.iprod(range(num_annots), range(num_annots)))
    pairwise_labels = np.array([hidden_nids[a1] == hidden_nids[a2] for a1, a2 in pairwise_aidxs])
    pairwise_scores = np.array([metric(*zz) for zz in pairwise_aidxs])
    pairwise_scores_mat = pairwise_scores.reshape(num_annots, num_annots)
    if num_annots <= 10:
        print(ut.repr2(pairwise_scores_mat, precision=1))

    #aids = list(range(num_annots))
    #g = netx.DiGraph()
    #g.add_nodes_from(aids)
    #g.add_edges_from([(tup[0], tup[1], {'weight': score}) for tup, score in zip(pairwise_aidxs, pairwise_scores) if tup[0] != tup[1]])
    #netx.draw_graphviz(g)
    #pr = netx.pagerank(g)

    X = pairwise_scores
    Y = pairwise_labels

    encoder = vt.ScoreNormalizer()
    encoder.fit(X, Y)
    encoder.visualize()

    # meanshift clustering
    import sklearn
    bandwidth = sklearn.cluster.estimate_bandwidth(X[:, None])  # , quantile=quantile, n_samples=500)
    assert bandwidth != 0, ('bandwidth is 0. Cannot cluster')
    # bandwidth is with respect to the RBF used in clustering
    #ms = sklearn.cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True)
    ms = sklearn.cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=False)
    ms.fit(X[:, None])
    label_arr = ms.labels_
    unique_labels = np.unique(label_arr)
    max_label = max(0, unique_labels.max())
    num_orphans = (label_arr == -1).sum()
    label_arr[label_arr == -1] = np.arange(max_label + 1, max_label + 1 + num_orphans)

    X_data = np.arange(num_annots)[:, None].astype(np.int64)

    #graph = pystruct.models.GraphCRF(
    #    n_states=None,
    #    n_features=None,
    #    inference_method='lp',
    #    class_weight=None,
    #    directed=False,
    #)

    import scipy
    import scipy.cluster
    import scipy.cluster.hierarchy

    thresh = 2.0
    labels = scipy.cluster.hierarchy.fclusterdata(X_data, thresh, metric=metric)
    unique_lbls, lblgroupxs = vt.group_indices(labels)
    print(groupxs)
    print(lblgroupxs)
    print('groupdiff = %r' % (ut.compare_groupings(groupxs, lblgroupxs),))
    print('common groups = %r' % (ut.find_grouping_consistencies(groupxs, lblgroupxs),))
    #X_data, seconds_thresh, criterion='distance')

    #help(hdbscan.HDBSCAN)

    import hdbscan
    alg = hdbscan.HDBSCAN(metric=metric, min_cluster_size=1, p=1, gen_min_span_tree=1, min_samples=2)
    labels = alg.fit_predict(X_data)
    labels[labels == -1] = np.arange(np.sum(labels == -1)) + labels.max() + 1
    unique_lbls, lblgroupxs = vt.group_indices(labels)
    print(groupxs)
    print(lblgroupxs)
    print('groupdiff = %r' % (ut.compare_groupings(groupxs, lblgroupxs),))
    print('common groups = %r' % (ut.find_grouping_consistencies(groupxs, lblgroupxs),))

    #import ddbscan
    #help(ddbscan.DDBSCAN)
    #alg = ddbscan.DDBSCAN(2, 2)

    #D = np.zeros((len(aids), len(aids) + 1))
    #D.T[-1] = np.arange(len(aids))

    ## Can alpha-expansion be used when the pairwise potentials are not in a grid?

    #hidden_ut.group_items(aids, hidden_nids)
    if False:
        import maxflow
        #from maxflow import fastmin
        # Create a graph with integer capacities.
        g = maxflow.Graph[int](2, 2)
        # Add two (non-terminal) nodes. Get the index to the first one.
        nodes = g.add_nodes(2)
        # Create two edges (forwards and backwards) with the given capacities.
        # The indices of the nodes are always consecutive.
        g.add_edge(nodes[0], nodes[1], 1, 2)
        # Set the capacities of the terminal edges...
        # ...for the first node.
        g.add_tedge(nodes[0], 2, 5)
        # ...for the second node.
        g.add_tedge(nodes[1], 9, 4)
        g = maxflow.Graph[float](2, 2)
        g.maxflow()
        g.get_nx_graph()
        g.get_segment(nodes[0])
