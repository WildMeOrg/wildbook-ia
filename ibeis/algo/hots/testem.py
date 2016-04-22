import numpy as np
import utool as ut


def draw_em_graph(P, Pn, PL, gam, num_labels):
    num_labels = PL.shape[1]
    lset_nodes = list(range(1, num_labels + 1))
    uset_nodes = ut.chr_range(len(Pn), base='a')
    nodes = lset_nodes + uset_nodes
    weight_matrix = P  # NOQA
    graph = ut.nx_from_matrix(P, nodes=nodes)
    import plottool as pt
    import networkx as nx
    nx.set_node_attributes(graph, 'groupid', {node: 'lset' for node in lset_nodes})
    nx.set_node_attributes(graph, 'color', {node: pt.RED for node in lset_nodes})
    nx.set_node_attributes(graph, 'groupid', {node: 'uset' for node in uset_nodes})
    pt.show_nx(graph, fontsize=10, prog='neato', layoutkw={'splines': 'spline', 'prog': 'neato', 'sep': 2.0}, verbose=0)
    pt.interactions.zoom_factory()


def test_em():
    """
    CommandLine:
        python -m ibeis.algo.hots.testem test_em --show
        python -m ibeis.algo.hots.testem test_em --show --no-cnn

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.algo.hots.testem import *  # NOQA
        >>> P, Pn, PL, gam, num_labels = test_em()
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> pt.qt4ensure()
        >>> draw_em_graph(P, Pn, PL, gam, num_labels)
        >>> ut.show_if_requested()
    """
    print('EM')

    # Matrix if unary probabilites, The probability that each node takes on a
    # given label, independent of its edges.

    test_case = [
        {'name': 1, 'view': 'L'},
        {'name': 1, 'view': 'L'},
        {'name': 2, 'view': 'L'},
        {'name': 2, 'view': 'R'},
        {'name': 2, 'view': 'B'},
        {'name': 3, 'view': 'L'},
        #{'name': 3, 'view': 'L'},
        #{'name': 4, 'view': 'L'},
    ]

    def make_test_similarity(test_case):
        #toy_params = {
        #    True:  {'mu': 0.9, 'sigma': .1},
        #    False: {'mu': 0.1, 'sigma': .4}
        #}
        tau = np.pi * 2
        view_to_ori = {
            'F': -1 * tau / 4,
            'L':  0 * tau / 4,
            'B':  1 * tau / 4,
            'R':  2 * tau / 4,
        }
        import vtool as vt

        nid_list = np.array(ut.dict_take_column(test_case, 'name'))
        yaw_list = np.array(ut.dict_take(view_to_ori, ut.dict_take_column(test_case, 'view')))

        rng = np.random.RandomState(0)
        pmat = []
        for idx in range(len(test_case)):
            nid = nid_list[idx]
            yaw = yaw_list[idx]
            p_same = nid == nid_list
            p_comp = 1 - vt.ori_distance(yaw_list, yaw) / np.pi
            # estimate noisy measurements
            p_same_m = np.clip(p_same + rng.normal(0, .5, size=len(p_same)), 0, .9)
            p_comp_m = np.clip(p_comp + rng.normal(0, .5, size=len(p_comp)), 0, .9)
            #
            p_same_and_comp = p_same_m * p_comp_m
            pmat.append(p_same_and_comp)
        #
        P = np.array(pmat)
        P[np.diag_indices(len(P))] = 0
        P = P + P.T / 2
        P = np.clip(P, .01, .99)
        print(ut.hz_str(' P = ', ut.array_repr2(P, precision=2, max_line_width=140)))
        return P

    Pn = make_test_similarity(test_case)

    if False:
        Pn = np.array(np.matrix(
            b"""
            .0 .7 .3 .2 .4 .5;
            .7 .0 .4 .4 .3 .5;
            .3 .4 .0 .6 .1 .5;
            .2 .4 .6 .0 .2 .3;
            .4 .3 .1 .2 .0 .8;
            .5 .5 .5 .3 .8 .0
            """))

        PL = np.array(np.matrix(
            b"""
            .7 .5 .5;
            .8 .4 .3;
            .5 .7 .3;
            .5 .8 .4;
            .3 .2 .8;
            .5 .5 .8
            """))
    num_nodes = Pn.shape[0]

    for num_labels in range(1, 6):
        #Pn = np.array(np.matrix(
        #    b"""
        #    .0 .7 .3 .2 .4 .5;
        #    .7 .0 .4 .4 .3 .5;
        #    .3 .4 .0 .6 .1 .5;
        #    .2 .4 .6 .0 .2 .3;
        #    .4 .3 .1 .2 .0 .8;
        #    .5 .5 .5 .3 .8 .0
        #    """))

        # Uniform distribution over labels
        PL = np.ones((num_nodes, num_labels)) / num_labels
        # Give nodes preferences
        PL[np.diag_indices(num_labels)] *= 1.01
        PL /= np.linalg.norm(PL, axis=0)
        # PL[0, :] = .01 / (num_labels - 1)
        # PL[0, 0] = .99

        # Number of nodes
        num_nodes = Pn.shape[0]
        # Number of classes
        num_labels = PL.shape[1]
        #num_labels = num_nodes
        #if 0 or num_labels != 3:
        #    PL = np.ones((num_nodes, num_labels)) / num_labels
        #    # PL[0, :] = .01 / (num_labels - 1)
        #    # PL[0, 0] = .99
        d = num_labels + num_nodes

        # Stack everything into a single matrix
        zero_part = np.zeros((num_labels, num_nodes + num_labels))
        prob_part = np.hstack([PL, Pn])
        #print(ut.hz_str(' prob_part = ', ut.array_repr2(prob_part[:, :], precision=2)))
        P = np.vstack([zero_part, prob_part])

        # Gamma will hold a probability distribution over the nodes
        # The labeled nodes must match themselves.
        # The unlabeld nodes are initialized with a uniform distribution.
        gam = np.hstack([np.eye(num_labels), np.ones((num_labels, num_nodes)) / num_labels])

        print('Initialize')
        print('num_labels = %r' % (num_labels,))
        #print(ut.hz_str(' gamma = ', ut.array_repr2(gam[:, num_labels:], max_line_width=140, precision=2)))

        delta_i = np.zeros(num_labels)
        def dErr(i, gam, P, delta_i=delta_i):
            # exepcted liklihood is cross entropy error
            delta_i[:] = 0
            # Compute the gradient of the cross entropy error
            for j in range(d):
                if i != j:
                    delta_i += gam[:, j] * np.log(P[i, j] / (1 - P[i, j]))
            # compute the projected gradient
            delta_i_hat = delta_i - delta_i.sum() / num_labels
            return delta_i_hat

        # Maximies the expected liklihood of gamma
        learn_rate = 0.05
        num_iters = 1000
        dGam = np.zeros(gam.shape)
        #for j in ut.ProgIter(range(num_iters), label='EM'):
        for j in range(num_iters):
            # Compute error gradient
            for i in range(num_labels, d):
                dGam[:, i] = dErr(i, gam, P)
            # Make a step in the gradient direction
            gam = gam + learn_rate * dGam
            # Normalize
            gam = np.clip(gam, 0, 1)
            for i in range(num_labels, d):
                gam[:, i] = gam[:, i] / np.sum(gam[:, i])
        print(ut.hz_str(' gamma = ', ut.array_repr2(gam[:, num_labels:], max_line_width=140, precision=2)))
        print('Finished')
    return P, Pn, PL, gam, num_labels


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.algo.hots.testem
        python -m ibeis.algo.hots.testem --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()

r"""

I've gone over both the pdf and the code and I would like to clarify a few things.

In the pdf the measure of uncertainty u_{ij} seems to not be used anywhere
else, is this true or am I missing it?

In the code there is a matrix PL and Pn.
I believe Pn is represented by the edge between the numbered nodes and the
lettered nodes, while PL (corresponds to gamma and) represents the connections
between the lettered nodes.  Is it correct that PL is proportional to the
probability that node i takes label j, and Pn is the probability that node i
matches node j?

In the animal ID problem there is not always a notion of groundtruth likelihood.
Nor, is the groundtruth fixed.  We need to find and correct errors in the
groundtruth.

For instance node 1 and node 2 might actually need to have the same label (node
1 may be a right side zebra and node 2 may be a left side zebra. Identity
equivalence is a transitive property. We need to be able to infer labeling
through chains of high likelihood connections even though the end points might
have a low probability edge between them.)

Using the language of image segmentation, I believe that without the "unary"
terms on each node (the probability of them taking on a given label) the
solution converges to a uniform distribution regardless of the "pairwise" terms
between the lettered nodes.


In summary:
    * How does this framework address uncertainty in the pairwise probabilities?

    * Does this framework work in the scenario where there is no groundtruth
      and the number of labels is unknown?


I'm having trouble understanding how this EM algorithm addresses uncertainty
and the case where there is no label information.

First, I do not see how uncertaint plays a role in the formulation. In the
attached pdf it is defined, but never used.

Second, I don't see how this works in the case where there are no "groundtruth"
exemplars and the number of classes is unknown.  Consider the case where we have
6 unlabeled annotations and we compute the pairwise similarity between each of
them.  We do not know how many individuals (classes) there are in this set.
There is a minimum of 1 and a maximum of 6.  We cannot compute the (unary)
probability that a pariticular annotation takes on a particular class because
the class is just an abstract notion used to define a partition of this graph.








If we have a unlabeled set of annotations and we compute the pairwise similarity between each of them,
how does this

I don't see how the uncertainty measure plays a role in the

I don't see how to get away from the "unary"
terms in this formulation

In fact in the example you gave, if you simply look at the probabilities on
each edge from the labeled nodes to the unlabeled nodes, you can simply take
the edge with the highest weight to a labeled node and arive at the same solution.
"""
