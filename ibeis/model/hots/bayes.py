# -*- coding: utf-8 -*-
"""

1) Ambiguity / num names
2) independence of annotations
3) continuous
4) exponential case
5) speicifc examples of our prob
6) human in loop

Arc reversal
http://www.cs.toronto.edu/~cebly/Papers/simulation.pdf

TODO:
    Need to find faster more mature libraries
    http://dlib.net/bayes.html
    http://www.cs.waikato.ac.nz/ml/weka/
    http://www.cs.waikato.ac.nz/~remco/weka.bn.pdf
    https://code.google.com/p/pebl-project/
    https://github.com/abhik/pebl
    http://www.cs.ubc.ca/~murphyk/Software/bnsoft.html

    Demo case where we think we know the labels of others.  Only one unknown
    name. Need to classify it as one of the other known names.

References:
    https://en.wikipedia.org/wiki/Bayesian_network
    https://class.coursera.org/pgm-003/lecture/17
    http://www.cs.ubc.ca/~murphyk/Bayes/bnintro.html
    http://www3.cs.stonybrook.edu/~sael/teaching/cse537/Slides/chapter14d_BP.pdf
    http://www.cse.unsw.edu.au/~cs9417ml/Bayes/Pages/PearlPropagation.html
    https://github.com/pgmpy/pgmpy.git
    http://pgmpy.readthedocs.org/en/latest/
    http://nipy.bic.berkeley.edu:5000/download/11
    http://pgmpy.readthedocs.org/en/latest/wiki.html#add-feature-to-accept-and-output-state-names-for-models
    http://www.csse.monash.edu.au/bai/book/BAI_Chapter2.pdf


Clustering with CRF:
    http://srl.informatik.uni-freiburg.de/publicationsdir/tipaldiIROS09.pdf
    http://www.dis.uniroma1.it/~dottoratoii/media/students/documents/thesis_tipaldi.pdf
    An Unsupervised Conditional Random Fields Approach for Clustering Gene Expression Time Series
    http://bioinformatics.oxfordjournals.org/content/24/21/2467.full


CRFs:
    http://homepages.inf.ed.ac.uk/csutton/publications/crftutv2.pdf

AlphaBeta Swap:
    https://github.com/amueller/gco_python
    https://github.com/pmneila/PyMaxflow
    http://www.cs.cornell.edu/rdz/papers/bvz-iccv99.pdf

    http://arxiv.org/pdf/1411.6340.pdf  Iteratively Reweighted Graph Cut for Multi-label MRFs with Non-convex Priors

Fusion Moves:
    http://www.robots.ox.ac.uk/~vilem/fusion.pdf
    http://hci.iwr.uni-heidelberg.de/publications/mip/techrep/beier_15_fusion.pdf

Consensus Clustering

Explaining Away

Course Notes:
    Tie breaking for MAP assignment.
    https://class.coursera.org/pgm-003/lecture/60
    * random perdibiation

    Correspondence Problem is discussed in
    https://class.coursera.org/pgm-003/lecture/68

    Sparse Pattern Factors

    Collective Inference:
    Plate Models / Aggragator CPD is used to define dependencies.


"""
from __future__ import absolute_import, division, print_function, unicode_literals
import six  # NOQA
import utool as ut
import numpy as np
from six.moves import zip
import pgmpy
import pgmpy.inference
import pgmpy.factors
import pgmpy.models
from ibeis.model.hots import pgm_ext
print, rrr, profile = ut.inject2(__name__, '[bayes]')


def test_model(num_annots, num_names, score_evidence=[], name_evidence=[],
               other_evidence={}, show_prior=False, noquery=False, **kwargs):
    verbose = ut.VERBOSE

    model = make_name_model(num_annots, num_names, verbose=verbose, **kwargs)

    if verbose:
        model.print_priors(ignore_ttypes=['match'])
        #ut.colorprint('\n --- Priors ---', 'darkblue')
        #for ttype, cpds in model.ttype2_cpds.items():
        #    if ttype != 'match':
        #        for fs_ in ut.ichunks(cpds, 4):
        #            ut.colorprint(ut.hz_str([f._cpdstr('psql') for f in fs_]), 'purple')

    model, evidence, soft_evidence = update_model_evidence(model, name_evidence, score_evidence, other_evidence)

    if verbose:
        ut.colorprint('\n --- Soft Evidence ---', 'white')
        for ttype, cpds in model.ttype2_cpds.items():
            if ttype != 'match':
                for fs_ in ut.ichunks(cpds, 4):
                    ut.colorprint(ut.hz_str([f._cpdstr('psql') for f in fs_]),
                                  'green')

    if verbose:
        ut.colorprint('\n --- Inference ---', 'red')

    if (len(evidence) > 0 or len(soft_evidence) > 0) and not noquery:
        interest_ttypes = ['name']
        model_inference = pgmpy.inference.VariableElimination(model)
        #model_inference = pgmpy.inference.BeliefPropagation(model)
        evidence = model_inference._ensure_internal_evidence(evidence, model)
        query_results = try_query(model, model_inference, evidence, interest_ttypes, verbose=verbose)
    else:
        query_results = {
            'factor_list': [],
            'joint_factor': None,
            'marginalized_joints': {},
        }

    factor_list = query_results['factor_list']
    marginalized_joints = query_results['marginalized_joints']

    if verbose:
        if verbose:
            print('+--------')
        semtypes = [model.var2_cpd[f.variables[0]].ttype
                    for f in factor_list]
        for type_, factors in ut.group_items(factor_list, semtypes).items():
            print('Result Factors (%r)' % (type_,))
            factors = ut.sortedby(factors, [f.variables[0] for f in factors])
            for fs_ in ut.ichunks(factors, 4):
                ut.colorprint(ut.hz_str([f._str('phi', 'psql') for f in fs_]),
                              'yellow')
        print('Joint Factors')
        for ttype, marginal in marginalized_joints.items():
            print('Marginal Joint %s Factors' % (ttype,))
            ut.colorprint(marginal._str('phi', 'psql', sort=-1, maxrows=4), 'white')
        print('L_____\n')

    show_model(model, evidence, '', factor_list, marginalized_joints, soft_evidence,  show_prior=show_prior)
    return (model,)
    # print_ascii_graph(model)


def coin_example():
    """

    CommandLine:
        python -m ibeis.model.hots.bayes --exec-coin_example

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.bayes import *  # NOQA
        >>> coin_example()
    """
    def toss_pmf(side, coin):
        toss_lookup = {
            'fair': {'heads': .5, 'tails': .5},
            #'bias': {'heads': .6, 'tails': .4},
            'bias': {'heads': .9, 'tails': 1},
        }
        return toss_lookup[coin][side]
    coin_cpd_t = pgm_ext.TemplateCPD(
        'coin', ['fair', 'bias'], varpref='C')
    toss_cpd_t = pgm_ext.TemplateCPD(
        'toss', ['heads', 'tails'], varpref='T',
        evidence_ttypes=[coin_cpd_t], pmf_func=toss_pmf)
    coin_cpd = coin_cpd_t.new_cpd(0)
    toss1_cpd = toss_cpd_t.new_cpd(parents=[coin_cpd, 1])
    toss2_cpd = toss_cpd_t.new_cpd(parents=[coin_cpd, 2])
    model = pgm_ext.define_model([coin_cpd, toss2_cpd, toss1_cpd])
    model.print_templates()
    model.print_priors()

    model_inference = pgmpy.inference.VariableElimination(model)

    print('Observe nothing')
    factor_list1 = model_inference.query(['T02'], {}).values()
    pgm_ext.print_factors(model, factor_list1)

    print('Observe that toss 1 was heads')
    evidence = model_inference._ensure_internal_evidence({'T01': 'heads'}, model)
    factor_list2 = model_inference.query(['T02'], evidence).values()
    pgm_ext.print_factors(model, factor_list2)

    phi1 = factor_list1[0]
    phi2 = factor_list2[0]
    assert phi2['heads'] > phi1['heads']
    print('Slightly more likely that you will see heads in the second coin toss')

    print('Observe nothing')
    factor_list1 = model_inference.query(['T02'], {}).values()
    pgm_ext.print_factors(model, factor_list1)

    print('Observe that toss 1 was tails')
    evidence = model_inference._ensure_internal_evidence({'T01': 'tails'}, model)
    factor_list2 = model_inference.query(['T02'], evidence).values()
    pgm_ext.print_factors(model, factor_list2)

    return model


def make_name_model(num_annots, num_names=None, verbose=True, mode=1):
    """
    Defines the general name model

    CommandLine:
        python -m ibeis.model.hots.bayes --exec-make_name_model --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.bayes import *  # NOQA
        >>> #defaults = dict(num_annots=5, num_names=3, verbose=True, mode=1)
        >>> defaults = dict(num_annots=2, num_names=2, verbose=True, mode=2)
        >>> kw = ut.argparse_funckw(make_name_model, defaults)
        >>> model = make_name_model(**kw)
        >>> ut.quit_if_noshow()
        >>> show_model(model, {}, '', [], {}, {},  show_prior=True)
        >>> ut.show_if_requested()
    """
    #annots = ut.chr_range(num_annots, base='a')
    mode = ut.get_argval('--mode', default=mode)
    annots = ut.chr_range(num_annots, base=ut.get_argval('--base', default='a'))
    # The indexes of match CPDs will not change if another annotation is added
    # It actually just needs to be a an index in rows of columns
    upper_diag_idxs = ut.upper_diagonalized_idxs(num_annots)
    if num_names is None:
        num_names = num_annots

    # -- Define CPD Templates
    def match_pmf(match_type, n1, n2):
        if n1 == n2:
            val = 1.0 if match_type == 'same' else 0.0
            #val = .999 if match_type == 'same' else 0.001
        elif n1 != n2:
            #val = 0.01 if match_type == 'same' else .99
            val = 0.0 if match_type == 'same' else 1.0
        return val

    def score_pmf(score_type, match_type):
        score_lookup = {
            'same': {'low': .1, 'high': .4, 'veryhigh': .5},
            'diff': {'low': .9, 'high': .09, 'veryhigh': .01}
            #'same': {'low': .1, 'high': .9},
            #'diff': {'low': .9, 'high': .1}
        }
        val = score_lookup[match_type][score_type]
        return val

    def score_pmf3(score_type, match_type, isdup='False'):
        score_lookup = {
            'False': {
                'same': {'low': .1, 'high': .5, 'veryhigh': .4},
                'diff': {'low': .9, 'high': .09, 'veryhigh': .01}
            },
            'True': {
                'same': {'low': .01, 'high': .2, 'veryhigh': .79},
                'diff': {'low': .4, 'high': .4, 'veryhigh': .2}
            }
        }
        val = score_lookup[isdup][match_type][score_type]
        return val

    def score_pmf2(score_type, n1, n2):
        score_lookup = {
            True: {'low': .1, 'high': .4, 'veryhigh': .5},
            False: {'low': .9, 'high': .09, 'veryhigh': .01}
            #'same': {'low': .1, 'high': .9},
            #'diff': {'low': .9, 'high': .1}
        }
        #if n1 == n2:
        #    val = .1 if score_type == 'low' else .9
        #else:
        #    val = .9 if score_type == 'low' else .1
        val = score_lookup[n1 == n2][score_type]
        return val

    def dup_pmf(True, match_type):
        lookup = {
            'same': {'True': 0.5, 'False': 0.5},
            'diff': {'True': 0.0, 'False': 1.0}
        }
        return lookup[match_type][True]

    special_basis_pool = ['fred', 'sue', 'paul']

    name_cpd_t = pgm_ext.TemplateCPD(
        'name', ('n', num_names), varpref='N',
        special_basis_pool=special_basis_pool)

    if mode == 1:
        match_cpd_t = pgm_ext.TemplateCPD(
            'match', ['diff', 'same'], varpref='M',
            evidence_ttypes=[name_cpd_t, name_cpd_t], pmf_func=match_pmf)
        score_cpd_t = pgm_ext.TemplateCPD(
            'score', ['low', 'high', 'veryhigh'],
            #'score', ['low', 'high'],
            varpref='S',
            evidence_ttypes=[match_cpd_t], pmf_func=score_pmf)

    elif mode == 2:
        name_cpd_t = pgm_ext.TemplateCPD(
            'name', ('n', num_names), varpref='N',
            special_basis_pool=special_basis_pool)
        score_cpd_t = pgm_ext.TemplateCPD(
            'score', ['low', 'high', 'veryhigh'],
            #'score', ['low', 'high'],
            varpref='S',
            evidence_ttypes=[name_cpd_t, name_cpd_t],
            pmf_func=score_pmf2)
    elif mode == 3 or mode == 4:
        match_cpd_t = pgm_ext.TemplateCPD(
            'match', ['diff', 'same'], varpref='M',
            evidence_ttypes=[name_cpd_t, name_cpd_t], pmf_func=match_pmf)
        if mode == 3:
            dup_cpd_t = pgm_ext.TemplateCPD(
                'dup', ['False', 'True'], varpref='D',
            )
        else:
            dup_cpd_t = pgm_ext.TemplateCPD(
                'dup', ['False', 'True'], varpref='D',
                evidence_ttypes=[match_cpd_t], pmf_func=dup_pmf
            )
        score_cpd_t = pgm_ext.TemplateCPD(
            'score', ['low', 'high', 'veryhigh'], varpref='S',
            evidence_ttypes=[match_cpd_t, dup_cpd_t], pmf_func=score_pmf3)

    # Instanciate templates

    if mode == 1:
        name_cpds = [name_cpd_t.new_cpd(parents=aid) for aid in annots]
        namepair_cpds = ut.list_unflat_take(name_cpds, upper_diag_idxs)
        match_cpds = [match_cpd_t.new_cpd(parents=cpds)
                      for cpds in namepair_cpds]
        score_cpds = [score_cpd_t.new_cpd(parents=cpds)
                      for cpds in zip(match_cpds)]
        cpd_list = name_cpds + score_cpds + match_cpds
    elif mode == 2:
        name_cpds = [name_cpd_t.new_cpd(parents=aid) for aid in annots]
        namepair_cpds = ut.list_unflat_take(name_cpds, upper_diag_idxs)
        score_cpds = [score_cpd_t.new_cpd(parents=cpds)
                      for cpds in namepair_cpds]
        cpd_list = name_cpds + score_cpds
    elif mode == 3 or mode == 4:
        name_cpds = [name_cpd_t.new_cpd(parents=aid) for aid in annots]
        namepair_cpds = ut.list_unflat_take(name_cpds, upper_diag_idxs)
        match_cpds = [match_cpd_t.new_cpd(parents=cpds)
                      for cpds in namepair_cpds]
        if mode == 3:
            dup_cpds = [dup_cpd_t.new_cpd(parents=''.join(map(str, aids))) for aids
                        in ut.list_unflat_take(annots, upper_diag_idxs)]
        else:
            dup_cpds = [dup_cpd_t.new_cpd(parents=[mcpds]) for mcpds
                        in match_cpds]
        score_cpds = [score_cpd_t.new_cpd(parents=([mcpds] + [dcpd]))
                      for mcpds, dcpd in zip(match_cpds, dup_cpds)]
        cpd_list = name_cpds + score_cpds + match_cpds + dup_cpds

    print('upper_diag_idxs = %r' % (upper_diag_idxs,))
    print('score_cpds = %r' % (ut.list_getattr(score_cpds, 'variable'),))
    # import sys
    # sys.exit(1)

    # Make Model
    model = pgm_ext.define_model(cpd_list)
    model.num_names = num_names

    if verbose:
        model.print_templates()
        #ut.colorprint('\n --- CPD Templates ---', 'blue')
        #for temp_cpd in templates:
        #    ut.colorprint(temp_cpd._cpdstr('psql'), 'turquoise')
    #print_ascii_graph(model)
    return model


def update_model_evidence(model, name_evidence, score_evidence, other_evidence):
    r"""

    CommandLine:
        python -m ibeis.model.hots.bayes --exec-update_model_evidence

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.bayes import *  # NOQA
        >>> verbose = True
        >>> other_evidence = {}
        >>> name_evidence = [0, 0, 1, 1, None]
        >>> score_evidence = ['high', 'low', 'low', 'low', 'low', 'high']
        >>> model = make_name_model(num_annots=5, num_names=3, verbose=True, mode=1)
        >>> model, evidence, soft_evidence = update_model_evidence(model, name_evidence, score_evidence, other_evidence)
    """
    name_cpds = model.ttype2_cpds['name']
    score_cpds = model.ttype2_cpds['score']

    evidence = {}
    evidence.update(other_evidence)
    soft_evidence = {}

    def apply_hard_soft_evidence(cpd_list, evidence_list):
        for cpd, ev in zip(cpd_list, evidence_list):
            if isinstance(ev, int):
                # hard internal evidence
                evidence[cpd.variable] = ev
            if isinstance(ev, six.string_types):
                # hard external evidence
                evidence[cpd.variable] = cpd._internal_varindex(
                    cpd.variable, ev)
            if isinstance(ev, dict):
                # soft external evidence
                # HACK THAT MODIFIES CPD IN PLACE
                def rectify_evidence_val(_v, card=cpd.variable_card):
                    # rectify hacky string structures
                    return 2 / (card + 1) if _v == '+eps' else _v
                ev_ = ut.map_dict_vals(rectify_evidence_val, ev)
                fill = (1 - sum(ev_.values())) / (cpd.variable_card - len(ev_))
                assert fill >= 0
                row_labels = list(ut.iprod(*cpd.statenames))

                for i, lbl in enumerate(row_labels):
                    if lbl in ev_:
                        # external case1
                        cpd.values[i] = ev_[lbl]
                    elif len(lbl) == 1 and lbl[0] in ev_:
                        # external case2
                        cpd.values[i] = ev_[lbl[0]]
                    elif i in ev_:
                        # internal case
                        cpd.values[i] = ev_[i]
                    else:
                        cpd.values[i] = fill
                soft_evidence[cpd.variable] = True

    apply_hard_soft_evidence(name_cpds, name_evidence)
    apply_hard_soft_evidence(score_cpds, score_evidence)
    return model, evidence, soft_evidence


def try_query(model, model_inference, evidence, interest_ttypes=[], verbose=True):
    r"""
    CommandLine:
        python -m ibeis.model.hots.bayes --exec-try_query --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.model.hots.bayes import *  # NOQA
        >>> verbose = True
        >>> other_evidence = {}
        >>> name_evidence = [0, 0, 1, 1, None]
        >>> score_evidence = ['high', 'low', 'low', 'low', 'low', 'high']
        >>> model = make_name_model(num_annots=5, num_names=3, verbose=True, mode=1)
        >>> model, evidence, soft_evidence = update_model_evidence(model, name_evidence, score_evidence, other_evidence)
        >>> interest_ttypes = ['name']
        >>> model_inference = pgmpy.inference.BeliefPropagation(model)
        >>> evidence = model_inference._ensure_internal_evidence(evidence, model)
        >>> query_results = try_query(model, model_inference, evidence, interest_ttypes, verbose)
        >>> result = ('query_results = %s' % (str(query_results),))
        >>> factor_list = query_results['factor_list']
        >>> marginalized_joints = query_results['marginalized_joints']
        >>> ut.quit_if_noshow()
        >>> show_model(model, evidence, '', factor_list, marginalized_joints, soft_evidence, show_prior=True)
        >>> ut.show_if_requested()

    Ignore:
        query_vars = ut.setdiff_ordered(model.nodes(), list(evidence.keys()))
        probs = model_inference.query(query_vars, evidence)
        map_assignment = model_inference.map_query(query_vars, evidence)
    """
    query_vars = ut.setdiff_ordered(model.nodes(), list(evidence.keys()))
    if verbose:
        evidence_str = ', '.join(model.pretty_evidence(evidence))
        print('P(' + ', '.join(query_vars) + ' | ' + evidence_str + ') = ')

    # Compute all marginals
    probs = model_inference.query(query_vars, evidence)
    #probs = model_inference.query(query_vars, evidence)

    #ut.embed()
    #probs = model_inference.map_query(query_vars, evidence)
    factor_list = probs.values()

    # Compute MAP joints
    # (probably an invalid thing to do)
    joint_factor = pgmpy.factors.factor_product(*factor_list)

    # Compute Marginalized MAP joints
    marginalized_joints = {}
    for ttype in interest_ttypes:
        other_vars = [v for v in joint_factor.scope()
                      if model.var2_cpd[v].ttype != ttype]
        marginal = joint_factor.marginalize(other_vars, inplace=False)
        marginalized_joints[ttype] = marginal

    query_results = {
        'factor_list': factor_list,
        'joint_factor': joint_factor,
        'marginalized_joints': marginalized_joints,
    }
    return query_results


def draw_tree_model(model):
    import plottool as pt
    import networkx as netx
    fnum = pt.ensure_fnum(None)
    fig = pt.figure(fnum=fnum, doclf=True)  # NOQA
    ax = pt.gca()
    #name_nodes = sorted(ut.list_getattr(model.ttype2_cpds['name'], 'variable'))
    netx_graph = model.to_markov_model()
    #pos = netx.pygraphviz_layout(netx_graph)
    #pos = netx.graphviz_layout(netx_graph)
    #pos = get_hacked_pos(netx_graph, name_nodes, prog='neato')
    pos = netx.pydot_layout(netx_graph)
    drawkw = dict(pos=pos, ax=ax, with_labels=True, node_size=1000)
    netx.draw(netx_graph, **drawkw)
    pt.set_figtitle('Markov Model')

    fnum = pt.ensure_fnum(None)
    fig = pt.figure(fnum=fnum, doclf=True)  # NOQA
    ax = pt.gca()
    netx_graph = model.to_junction_tree()
    #netx_graph = model.to_markov_model()
    #pos = netx.pygraphviz_layout(netx_graph)
    #pos = netx.graphviz_layout(netx_graph)
    pos = netx.pydot_layout(netx_graph)
    drawkw = dict(pos=pos, ax=ax, with_labels=True, node_size=1000)
    netx.draw(netx_graph, **drawkw)
    pt.set_figtitle('Junction/Clique Tree / Cluster Graph')


def get_hacked_pos2(netx_graph, name_nodes=None, prog='dot'):
    import pygraphviz
    import networkx as netx
    # Add "invisible" edges to induce an ordering
    # Hack for layout (ordering of top level nodes)
    if hasattr(netx_graph, 'ttype2_cpds'):
        name_nodes = sorted(ut.list_getattr(netx_graph.ttype2_cpds['name'], 'variable'))
    if name_nodes is not None:
        #netx.set_node_attributes(netx_graph, 'label', {n: {'label': n} for n in all_nodes})
        invis_edges = list(ut.itertwo(name_nodes))
        netx_graph2 = netx_graph.copy()
        netx_graph2.add_edges_from(invis_edges)
        A = netx.to_agraph(netx_graph2)
        A.add_subgraph(name_nodes, rank='same')
    else:
        netx_graph2 = netx_graph
        A = netx.to_agraph(netx_graph2)
    args = ''
    G = netx_graph
    A.layout(prog=prog, args=args)
    #A.draw('example.png', prog='dot')
    node_pos = {}
    for n in G:
        node_ = pygraphviz.Node(A, n)
        try:
            xx, yy = node_.attr["pos"].split(',')
            node_pos[n] = (float(xx), float(yy))
        except:
            print("no position for node", n)
            node_pos[n] = (0.0, 0.0)
    return node_pos


def get_hacked_pos(netx_graph, name_nodes=None, prog='dot'):
    import pygraphviz
    import networkx as netx
    # Add "invisible" edges to induce an ordering
    # Hack for layout (ordering of top level nodes)
    netx_graph2 = netx_graph.copy()
    if getattr(netx_graph, 'ttype2_cpds', None) is not None:
        grouped_nodes = []
        for ttype in netx_graph.ttype2_cpds.keys():
            # if ttype not in ['match', 'name']:
            #     continue
            ttype_cpds = netx_graph.ttype2_cpds[ttype]
            # use defined ordering
            ttype_nodes = ut.list_getattr(ttype_cpds, 'variable')
            # ttype_nodes = sorted(ttype_nodes)
            invis_edges = list(ut.itertwo(ttype_nodes))
            netx_graph2.add_edges_from(invis_edges)
            grouped_nodes.append(ttype_nodes)

        A = netx.to_agraph(netx_graph2)
        for nodes in grouped_nodes:
            A.add_subgraph(nodes, rank='same')
    else:
        A = netx.to_agraph(netx_graph2)

    #if name_nodes is not None:
    #    #netx.set_node_attributes(netx_graph, 'label', {n: {'label': n} for n in all_nodes})
    #    invis_edges = list(ut.itertwo(name_nodes))
    #    netx_graph2.add_edges_from(invis_edges)
    #    A.add_subgraph(name_nodes, rank='same')
    #else:
    #    A = netx.to_agraph(netx_graph2)
    args = ''
    G = netx_graph
    A.layout(prog=prog, args=args)
    #A.draw('example.png', prog='dot')
    node_pos = {}
    for n in G:
        node_ = pygraphviz.Node(A, n)
        try:
            xx, yy = node_.attr["pos"].split(',')
            node_pos[n] = (float(xx), float(yy))
        except:
            print("no position for node", n)
            node_pos[n] = (0.0, 0.0)
    return node_pos


def show_model(model, evidence=None, suff='', factor_list=None,
               marginalized_joints=None, soft_evidence={}, show_prior=False, ):
    """
    References:
        http://stackoverflow.com/questions/22207802/pygraphviz-networkx-set-node-level-or-layer

    Ignore:
        pkg-config --libs-only-L libcgraph
        sudo apt-get  install libgraphviz-dev -y
        sudo apt-get  install libgraphviz4 -y

        # sudo apt-get install pkg-config
        sudo apt-get install libgraphviz-dev
        # pip install git+git://github.com/pygraphviz/pygraphviz.git
        pip install pygraphviz
        python -c "import pygraphviz; print(pygraphviz.__file__)"
    """
    import plottool as pt
    import networkx as netx
    import matplotlib as mpl
    fnum = pt.ensure_fnum(None)
    fig = pt.figure(fnum=fnum, doclf=True)  # NOQA
    ax = pt.gca()
    var2_post = {f.variables[0]: f for f in factor_list}

    netx_graph = (model)
    #netx_graph.graph.setdefault('graph', {})['size'] = '"10,5"'
    #netx_graph.graph.setdefault('graph', {})['rankdir'] = 'LR'

    pos = get_hacked_pos(netx_graph)
    #netx.pygraphviz_layout(netx_graph)
    #pos = netx.pydot_layout(netx_graph, prog='dot')
    #pos = netx.graphviz_layout(netx_graph)

    drawkw = dict(pos=pos, ax=ax, with_labels=True, node_size=2000)
    if evidence is not None:
        node_colors = [
            # (pt.TRUE_BLUE
            (pt.WHITE
             if node not in soft_evidence else
             pt.LIGHT_PINK)
            if node not in evidence
            else pt.FALSE_RED
            for node in netx_graph.nodes()]

        for node in netx_graph.nodes():
            cpd = model.var2_cpd[node]
            if cpd.ttype == 'score':
                pass
        drawkw['node_color'] = node_colors
        #ut.embed()
    netx.draw(netx_graph, **drawkw)

    if True:
        textprops = {
            'family': 'monospace',
            'horizontalalignment': 'left',
            #'horizontalalignment': 'center',
            'size': 12,
            #'size': 8,
        }

        textkw = dict(
            xycoords='data', boxcoords='offset points', pad=0.25,
            frameon=True, arrowprops=dict(arrowstyle='->'),
            #bboxprops=dict(fc=node_attr['fillcolor']),
        )

        netx_nodes = model.nodes(data=True)
        node_key_list = ut.get_list_column(netx_nodes, 0)
        pos_list = ut.dict_take(pos, node_key_list)

        artist_list = []
        offset_box_list = []
        for pos_, node in zip(pos_list, netx_nodes):
            x, y = pos_
            variable = node[0]

            cpd = model.var2_cpd[variable]

            prior_marg = (cpd if cpd.evidence is None else
                          cpd.marginalize(cpd.evidence, inplace=False))

            prior_text = None
            text = None
            if variable in evidence:
                text = cpd.variable_statenames[evidence[variable]]
            elif variable in var2_post:
                post_marg = var2_post[variable]
                text = pgm_ext.make_factor_text(post_marg, 'post_marginal')
                prior_text = pgm_ext.make_factor_text(prior_marg, 'prior_marginal')
            else:
                if len(evidence) == 0 and len(soft_evidence) == 0:
                    prior_text = pgm_ext.make_factor_text(prior_marg, 'prior_marginal')

            if text is not None:
                offset_box = mpl.offsetbox.TextArea(text, textprops)
                artist = mpl.offsetbox.AnnotationBbox(
                    # offset_box, (x + 5, y), xybox=(20., 5.),
                    offset_box, (x, y + 5), xybox=(0., 20.),
                    #box_alignment=(0, 0),
                    box_alignment=(.5, 0),
                    **textkw)
                offset_box_list.append(offset_box)
                artist_list.append(artist)

            if show_prior and prior_text is not None:
                offset_box2 = mpl.offsetbox.TextArea(prior_text, textprops)
                artist2 = mpl.offsetbox.AnnotationBbox(
                    # offset_box2, (x - 5, y), xybox=(-20., -15.),
                    offset_box2, (x, y - 5), xybox=(-15., -20.),
                    #box_alignment=(1, 1),
                    box_alignment=(.5, 1),
                    **textkw)
                offset_box_list.append(offset_box2)
                artist_list.append(artist2)

        for artist in artist_list:
            ax.add_artist(artist)

        xmin, ymin = np.array(pos_list).min(axis=0)
        xmax, ymax = np.array(pos_list).max(axis=0)
        num_annots = len(model.ttype2_cpds['name'])
        if num_annots > 4:
            ax.set_xlim((xmin - 40, xmax + 40))
            ax.set_ylim((ymin - 50, ymax + 50))
            fig.set_size_inches(30, 7)
        else:
            ax.set_xlim((xmin - 42, xmax + 42))
            ax.set_ylim((ymin - 50, ymax + 50))
            fig.set_size_inches(23, 7)
        fig = pt.gcf()

        title = 'num_names=%r, num_annots=%r' % (model.num_names, num_annots,)
        max_marginal_list = []
        for name, marginal in marginalized_joints.items():
            # ut.embed()
            states = list(ut.iprod(*marginal.statenames))
            vals = marginal.values.ravel()
            x = vals.argmax()
            max_marginal_list += ['P(' + ', '.join(states[x]) + ') = ' + str(vals[x])]
            # title += str(marginal)
        title += '\n' + '\n'.join(max_marginal_list)
        pt.set_figtitle(title, size=14)
        #pt.set_xlabel()

        def hack_fix_centeralign():
            if textprops['horizontalalignment'] == 'center':
                print('Fixing centeralign')
                fig = pt.gcf()
                fig.canvas.draw()

                # Superhack for centered text. Fix bug in
                # /usr/local/lib/python2.7/dist-packages/matplotlib/offsetbox.py
                # /usr/local/lib/python2.7/dist-packages/matplotlib/text.py
                for offset_box in offset_box_list:
                    offset_box.set_offset
                    z = offset_box._text.get_window_extent()
                    (z.x1 - z.x0) / 2
                    offset_box._text
                    T = offset_box._text.get_transform()
                    A = mpl.transforms.Affine2D()
                    A.clear()
                    A.translate((z.x1 - z.x0) / 2, 0)
                    offset_box._text.set_transform(T + A)
        hack_fix_centeralign()
    #fpath = ('name_model_' + suff + '.png')
    #pt.plt.savefig(fpath)
    #return fpath


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

    import vtool as vt
    num_annots = 10
    num_names = num_annots
    hidden_nids = np.random.randint(0, num_names, num_annots)
    unique_nids, groupxs = vt.group_indices(hidden_nids)

    toy_params = {
        True: {'mu': 1.0, 'sigma': 2.2},
        False: {'mu': 7.0, 'sigma': .9}
    }

    if True:
        import plottool as pt
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
    assert bandwidth != 0, ('[enc] bandwidth is 0. Cannot cluster')
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


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.model.hots.bayes
        python -m ibeis.model.hots.bayes --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
