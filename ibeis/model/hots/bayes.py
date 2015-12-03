# -*- coding: utf-8 -*-
"""

References:
    https://class.coursera.org/pgm-003/lecture/17
    http://www.cs.ubc.ca/~murphyk/Bayes/bnintro.html
    http://www3.cs.stonybrook.edu/~sael/teaching/cse537/Slides/chapter14d_BP.pdf
    http://www.cse.unsw.edu.au/~cs9417ml/Bayes/Pages/PearlPropagation.html
    https://github.com/pgmpy/pgmpy.git
    http://pgmpy.readthedocs.org/en/latest/
    http://nipy.bic.berkeley.edu:5000/download/11
    http://pgmpy.readthedocs.org/en/latest/wiki.html#add-feature-to-accept-and-output-state-names-for-models

"""
from __future__ import absolute_import, division, print_function, unicode_literals
import six  # NOQA
import utool as ut
import numpy as np
from six.moves import zip, map
import pgmpy
import pgmpy.inference
import pgmpy.factors
import pgmpy.models
print, rrr, profile = ut.inject2(__name__, '[bayes]')


def make_bayes_notebook():
    r"""
    CommandLine:
        python -m ibeis.model.hots.bayes --exec-make_bayes_notebook

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.bayes import *  # NOQA
        >>> result = make_bayes_notebook()
        >>> print(result)
    """
    from ibeis.templates import generate_notebook
    initialize = ut.codeblock(
        r'''
        # STARTBLOCK
        from ibeis.model.hots.bayes import *  # NOQA
        # Matplotlib stuff
        import matplotlib as mpl
        %matplotlib inline
        %load_ext autoreload
        %autoreload
        # ENDBLOCK
        '''
    )
    cell_list_def = [
        initialize,
        show_model_templates,
        demo_name_annot_complexity,
        #demo_model_idependencies1,
        #demo_model_idependencies2,
        demo_single_add,
        demo_single_add_soft,
    ]
    def format_cell(cell):
        if ut.is_funclike(cell):
            header = '# ' + ut.to_title_caps(ut.get_funcname(cell))
            code = (header, ut.get_func_sourcecode(cell, stripdef=True, stripret=True))
        else:
            code = cell
        return generate_notebook.format_cells(code)

    cell_list = ut.flatten([format_cell(cell) for cell in cell_list_def])
    nbstr = generate_notebook.make_notebook(cell_list)
    print('nbstr = %s' % (nbstr,))
    ut.writeto('bayes.ipynb', nbstr)


def show_model_templates():
    r"""
    CommandLine:
        python -m ibeis.model.hots.bayes --exec-show_model_templates

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.bayes import *  # NOQA
        >>> result = show_model_templates()
        >>> print(result)
    """
    make_name_model(2, 2, verbose=True)


def demo_single_add():
    """
    This demo shows how a name is assigned to a new annotation.

    CommandLine:
        python -m ibeis.model.hots.bayes --exec-demo_single_add

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.bayes import *  # NOQA
        >>> result = demo_single_add()
        >>> print(result)
    """
    # Initially there are only two annotations that have a strong match
    test_model(num_annots=2, num_names=5, score_evidence=['high'], name_evidence=[0])
    # Adding a new annotation does not change the original probabilites
    test_model(num_annots=3, num_names=5, score_evidence=['high'], name_evidence=[0])
    # Adding evidence that Na matches Nc does not influence the probability
    # that Na matches Nb. However the probability that Nb matches Nc goes up.
    test_model(num_annots=3, num_names=5, score_evidence=['high', 'high'], name_evidence=[0])
    # However, once Nb is scored against Nb that does increase the likelihood
    # that all 3 are fred goes up significantly.
    test_model(num_annots=3, num_names=5, score_evidence=['high', 'high', 'high'], name_evidence=[0])


def demo_single_add_soft():
    """
    This is the same as demo_single_add, but soft labels are used.

    CommandLine:
        python -m ibeis.model.hots.bayes --exec-demo_single_add_soft

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.bayes import *  # NOQA
        >>> result = demo_single_add_soft()
        >>> print(result)
    """
    # Initially there are only two annotations that have a strong match
    #test_model(num_annots=2, num_names=5, score_evidence=['high'], name_evidence=[{0: .9}])
    # Adding a new annotation does not change the original probabilites
    #test_model(num_annots=3, num_names=5, score_evidence=['high'], name_evidence=[{0: .9}])
    # Adding evidence that Na matches Nc does not influence the probability
    # that Na matches Nb
    test_model(num_annots=3, num_names=5, score_evidence=['high', 'high'], name_evidence=[{0: .9}])
    # However, once Nb is scored against Nb that does increase the likelihood
    # that all 3 are fred goes up significantly.
    test_model(num_annots=3, num_names=5, score_evidence=['high', 'high', 'high'], name_evidence=[{0: .9}])


def demo_name_annot_complexity():
    """
    This demo is meant to show the structure of the graph as more annotations
    and names are added.

    CommandLine:
        python -m ibeis.model.hots.bayes --exec-demo_name_annot_complexity --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.bayes import *  # NOQA
        >>> demo_name_annot_complexity()
        >>> ut.show_if_requested()
    """
    # Initially there are 2 annots and 4 names
    test_model(num_annots=2, num_names=4, score_evidence=[], name_evidence=[])
    # Adding a name causes the probability of the other names to go down
    test_model(num_annots=2, num_names=5, score_evidence=[], name_evidence=[])
    # Adding an annotation wihtout matches does not effect probabilities of
    # names
    test_model(num_annots=3, num_names=5, score_evidence=[], name_evidence=[])
    test_model(num_annots=4, num_names=10, score_evidence=[], name_evidence=[])


def demo_model_idependencies1():
    """
    Independences of the 2 annot 2 name model
    """
    model = test_model(num_annots=2, num_names=2, score_evidence=[], name_evidence=[])[0]
    # This model has the following independenceis
    idens = model.get_independencies()
    # Might not be valid, try and collapse S and M
    xs = list(map(str, idens.independencies))
    import re
    xs = [re.sub(', M..', '', x) for x in xs]
    xs = [re.sub('M..,?', '', x) for x in xs]
    xs = [x for x in xs if not x.startswith('( _')]
    xs = [x for x in xs if not x.endswith('| )')]
    print('\n'.join(sorted(list(set(xs)))))


def demo_model_idependencies2():
    """
    Independences of the 3 annot 3 name model
    """
    model = test_model(num_annots=3, num_names=3, score_evidence=[], name_evidence=[])[0]
    # This model has the following independenceis
    idens = model.get_independencies()
    print(idens)

# Might not be valid, try and collapse S and M
#xs = list(map(str, idens.independencies))
#import re
#xs = [re.sub(', M..', '', x) for x in xs]
#xs = [re.sub('M..,?', '', x) for x in xs]
#xs = [x for x in xs if not x.startswith('( _')]
#xs = [x for x in xs if not x.endswith('| )')]
#print('\n'.join(sorted(list(set(xs)))))


def bayesnet_cases():
    r"""
    CommandLine:
        python -m ibeis.model.hots.bayes --exec-bayesnet_cases

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.bayes import *  # NOQA
        >>> bayesnet_cases()
    """
    from functools import partial
    import itertools
    count = partial(six.next, itertools.count(1))

    test_model(count(), num_annots=2, num_names=4, high_idx=[],
               name_evidence=[])  # init
    test_model(count(), num_annots=2, num_names=4, high_idx=[0],
               name_evidence=['n0'])  # Start with 4 names.
    test_model(count(), num_annots=2, num_names=5, high_idx=[0],
               name_evidence=['n0'])  # Add a name, Causes probability of match to go down
    test_model(count(), num_annots=3, num_names=5, high_idx=[0],
               name_evidence=['n0'])  # Add Annotation
    test_model(count(), num_annots=3, num_names=5, high_idx=[0, 2],
               name_evidence=['n0'])

    test_model(count(), num_annots=3, num_names=5, high_idx=[0, 2],
               name_evidence=['n0', {'n0': .9}])
    test_model(count(), num_annots=3, num_names=5, high_idx=[0],
               name_evidence=['n0', {'n0': .9}])
    test_model(count(), num_annots=3, num_names=5, high_idx=[0],
               name_evidence=[{'n0': .99}, {'n0': .9}])
    test_model(count(), num_annots=3, num_names=10, high_idx=[0],
               name_evidence=[{'n0': .99}, {'n0': .9}])
    test_model(count(), num_annots=3, num_names=10, high_idx=[0],
                       name_evidence=[{'n0': .99}, {'n0': .2, 'n0': .7}])

    #fpath = test_model(count(), (3, 10), high_idx=[0, 1],
    #                   name_evidence=[{'n0': .99}, {'n0': .2, 'n0': .7}])
    #fpath = test_model(count(), (3, 10), high_idx=[0, 1],
    #                   name_evidence=[{'n0': .99}, {'n0': .2, 'n0': .7}, {'n0': .32}])
    test_model(count(), num_annots=3, num_names=10, high_idx=[0, 1, 2],
                       name_evidence=[{'n0': .99}, {'n0': .2, 'n0': .7}, {'n0': .32}])
    # Fix indexing to move in diagonal order as opposed to row order
    test_model(count(), num_annots=4, num_names=10, high_idx=[0, 1, 2],
                       name_evidence=[{'n0': .99}, {'n0': .2, 'n0': .7}, {'n0': .32}])
    #fpath = test_model(count(), (4, 10),
    #                   high_idx=[0, 1, 3], low_idx=[2], name_evidence=[{'n0': .99}, {'n0': .2, 'n0': .7}, {'n0': .32}])

    #fpath = test_model(count(), (4, 10))

    #ut.startfile(fpath)


def test_model(num_annots, num_names, score_evidence=[], name_evidence=[]):
    verbose = ut.VERBOSE

    model = make_name_model(num_annots, num_names, verbose=verbose)

    if verbose:
        ut.colorprint('\n --- Inference ---', 'red')

    name_cpds = model.ttype2_cpds['name']
    score_cpds = model.ttype2_cpds['score']

    evidence = {}
    soft_evidence = {}

    # Set ni to always be Fred
    #N0 = name_cpds[0]

    def apply_hard_soft_evidence(cpd_list, evidence_list):
        for cpd, ev in zip(cpd_list, evidence_list):
            if isinstance(ev, int):
                # hard internal evidence
                evidence[cpd.variable] = ev
            if isinstance(ev, six.string_types):
                # hard external evidence
                evidence[cpd.variable] = cpd.statename_to_index(cpd.variable, ev)
            if isinstance(ev, dict):
                # soft external evidence
                # HACK THAT MODIFIES CPD IN PLACE
                fill = (1 - sum(ev.values())) / (len(cpd.values) - len(ev))
                assert fill >= 0
                row_labels = list(ut.iprod(*cpd.statenames))

                for i, lbl in enumerate(row_labels):
                    if lbl in ev:
                        # external case1
                        cpd.values[i] = ev[lbl]
                    elif len(lbl) == 1 and lbl[0] in ev:
                        # external case2
                        cpd.values[i] = ev[lbl[0]]
                    elif i in ev:
                        # internal case
                        cpd.values[i] = ev[i]
                    else:
                        cpd.values[i] = fill
                soft_evidence[cpd.variable] = True

    apply_hard_soft_evidence(name_cpds, name_evidence)
    apply_hard_soft_evidence(score_cpds, score_evidence)

    #for Sij, sev in zip(score_cpds, score_evidence):
    #    if isinstance(sev, six.string_types):
    #        evidence[Sij.variable] = Sij.statename_to_index(Sij.variable, sev)

    #for idx in high_idx:
    #    evidence[score_cpds[idx].variable] = 1
    #for idx in low_idx:
    #    evidence[score_cpds[idx].variable] = 0

    if len(evidence) > 0:
        model_inference = pgmpy.inference.BeliefPropagation(model)
        #model_inference = pgmpy.inference.VariableElimination(model)
        factor_list = try_query(model, model_inference, evidence, verbose=verbose)
    else:
        factor_list = []

    show_model(model, evidence, '', factor_list, soft_evidence)
    return (model,)
    # print_ascii_graph(model)
    #return evidence


def try_query(model, model_inference, evidence, verbose=True):
    if verbose:
        print('+--------')
    query_vars = ut.setdiff_ordered(model.nodes(), list(evidence.keys()))
    evidence_str = ', '.join(model.pretty_evidence(evidence))
    if verbose:
        print('P(' + ', '.join(query_vars) + ' | ' + evidence_str + ') = ')
    probs = model_inference.query(query_vars, evidence)
    factor_list = probs.values()
    if verbose:
        joint_factor = pgmpy.factors.factor_product(*factor_list)
        # print(joint_factor.get_independencies())
        # print(model.local_independencies([Ni.variable]))
        #print('Result Factors')
        factor = joint_factor  # NOQA
        semtypes = [model.var2_cpd[f.variables[0]].ttype for f in factor_list]
        for type_, factors in ut.group_items(factor_list, semtypes).items():
            print('Result Factors (%r)' % (type_,))
            factors = ut.sortedby(factors, [f.variables[0] for f in factors])
            for fs_ in ut.ichunks(factors, 4):
                ut.colorprint(ut.hz_str([f._str('phi', 'psql') for f in fs_]), 'yellow')
        #print('Joint Factors')
        #ut.colorprint(joint_factor._str('phi', 'psql', sort=True), 'white')
        #name_vars = [v for v in joint_factor.scope() if model.var2_cpd[v].ttype == 'name']
        #print('Marginal Factors')
        #marginal = joint_factor.marginalize(name_vars, inplace=False)
        #ut.colorprint(marginal._str('phi', 'psql', sort=-1, maxrows=4), 'white')
        print('L_____\n')
    return factor_list


def make_name_model(num_annots, num_names=None, verbose=True):
    #annots = ut.chr_range(num_annots, base='a')
    annots = ut.chr_range(num_annots, base=ut.get_argval('--base', default='a'))
    if num_names is None:
        num_names = num_annots

    # -- Define CPD Templates
    def match_pmf(match_type, n1, n2):
        if n1 == n2:
            val = 1.0 if match_type == 'same' else 0.0
        elif n1 != n2:
            val = 0.0 if match_type == 'same' else 1.0
        return val

    def score_pmf(score_type, match_type):
        if match_type == 'same':
            val = .1 if score_type == 'low' else .9
        elif match_type == 'diff':
            val = .9 if score_type == 'low' else .1
        return val

    special_basis_pool = ['fred', 'sue', 'paul']

    name_cpd = TemplateCPD('name', ('n', num_names), varpref='N',
                           special_basis_pool=special_basis_pool)

    match_cpd = TemplateCPD('match', ['diff', 'same'], varpref='M',
                            evidence_ttypes=[name_cpd, name_cpd],
                            pmf_func=match_pmf)

    score_cpd = TemplateCPD('score', ['low', 'high'], varpref='S',
                            evidence_ttypes=[match_cpd],
                            pmf_func=score_pmf)

    PRINT_TEMPLATES = verbose
    if PRINT_TEMPLATES:
        ut.colorprint('\n --- CPD Templates ---', 'blue')
        ut.colorprint(name_cpd._cpdstr('psql'), 'turquoise')
        ut.colorprint(match_cpd._cpdstr('psql'), 'turquoise')
        ut.colorprint(score_cpd._cpdstr('psql'), 'turquoise')

    # -- Build CPDS
    name_cpds = [name_cpd.new_cpd(_id=aid) for aid in annots]

    # This way of enumeration helps during testing
    # The indexes of match cpds will not change if another annotation is added
    diag_idxs = list(ut.diagonalized_iter(len(name_cpds)))
    upper_diag_idxs = [(r, c) for r, c in diag_idxs if r < c]

    match_cpds = [
        match_cpd.new_cpd(evidence_cpds=cpds)
        #for cpds in list(ut.iter_window(name_cpds, 2, wrap=len(name_cpds) > 2))
        for cpds in ut.list_unflat_take(name_cpds, upper_diag_idxs)
        #for cpds in list(ut.upper_diag_self_prodx(name_cpds))
    ]

    score_cpds = [
        score_cpd.new_cpd(evidence_cpds=cpds)
        for cpds in zip(match_cpds)
    ]

    if False:
        print(ut.list_getattr(name_cpds, 'variable'))
        print(ut.list_getattr(match_cpds, 'variable'))
        print(ut.list_getattr(score_cpds, 'variable'))
        print('num_names = %r' % (num_names,))
        print('len(annots) = %r' % (len(annots),))
        print('len(name_cpds) = %r' % (len(name_cpds),))
        print('len(match_cpds) = %r' % (len(match_cpds),))
        print('len(score_cpds) = %r' % (len(score_cpds),))

    # ----
    # Make Model
    cpd_list = name_cpds + score_cpds + match_cpds
    input_graph = ut.flatten([
        [(evar, cpd.variable) for evar in cpd.evidence]
        for cpd in cpd_list if cpd.evidence is not None
    ])
    model = pgmpy.models.BayesianModel(input_graph)
    model.add_cpds(*cpd_list)
    model.var2_cpd = {cpd.variable: cpd for cpd in model.cpds}
    model.ttype2_cpds = ut.groupby_attr(model.cpds, 'ttype')

    def pretty_evidence(model, evidence):
        return [evar + '=' + str(model.var2_cpd[evar].variable_statenames[val])
                for evar, val in evidence.items()]
    ut.inject_func_as_method(model, pretty_evidence)
    model.num_names = num_names
    #print_ascii_graph(model)
    return model


class TemplateCPD(object):
    """
    Factory for templated cpds
    """
    def __init__(self, ttype, basis, varpref, evidence_ttypes=None,
                 pmf_func=None, special_basis_pool=None):
        if isinstance(basis, tuple):
            state_pref, state_card = basis
            stop = state_card
            basis = []
            num_special = 0
            if special_basis_pool is not None:
                start = stop - len(special_basis_pool)
                num_special = min(len(special_basis_pool), state_card)
                basis = special_basis_pool[0:num_special]
            if (state_card - num_special) > 0:
                start = num_special
                basis = basis + [state_pref + str(i) for i in range(start, stop)]
        if varpref is None:
            varpref = ttype
        self.basis = basis
        self.ttype = ttype
        self.varpref = varpref
        self.evidence_ttypes = evidence_ttypes
        self.pmf_func = pmf_func

    def __call__(self, *args, **kwargs):
        return self.new_cpd(*args, **kwargs)

    def _cpdstr(self, *args, **kwargs):
        example_cpd = self.example_cpd()
        return example_cpd._cpdstr(*args, **kwargs)

    @ut.memoize
    def example_cpd(self, id_=0):
        kw = dict()
        if self.evidence_ttypes is None:
            kw['_id'] = ut.chr_range(id_, id_ + 1)[0]
        else:

            kw['evidence_cpds'] = [tcpd.example_cpd(i) for i, tcpd in enumerate(self.evidence_ttypes)]
        example_cpd = self.new_cpd(**kw)
        return example_cpd

    def new_cpd(self, _id=None, evidence_cpds=None, pmf_func=None):
        if pmf_func is None:
            pmf_func = self.pmf_func
        if _id is None:
            _id = ''.join([cpd._template_id for cpd in evidence_cpds])
        variable = ''.join([self.varpref, _id])
        variable_card = len(self.basis)
        statename_dict = {
            variable: self.basis,
        }
        if self.evidence_ttypes is not None:
            if any(cpd.ttype != tcpd.ttype
                   for cpd, tcpd in zip(evidence_cpds, evidence_cpds)):
                raise ValueError('Evidence is not of appropriate type')
            evidence_bases = [cpd.variable_statenames for cpd in evidence_cpds]
            evidence_card = list(map(len, evidence_bases))
            evidence_states = list(ut.iprod(*evidence_bases))

            for cpd in evidence_cpds:
                statename_dict.update(cpd.statename_dict)

            values = np.array([
                [pmf_func(vstate, *estates) for estates in evidence_states]
                for vstate in self.basis])
            if False:
                # ensure normalized
                values = values / values.sum(axis=0)
            evidence = [cpd.variable for cpd in evidence_cpds]
        else:
            if evidence_cpds is not None:
                raise ValueError('Gave evidence for evidence-less template')
            evidence = None
            evidence_card = None

        if pmf_func is None:
            # assume uniform
            values = [[1.0 / variable_card] * variable_card]

        cpd = pgmpy.factors.TabularCPD(
            variable=variable,
            variable_card=variable_card,
            values=values,
            evidence=evidence,
            evidence_card=evidence_card,
            statename_dict=statename_dict,
        )
        cpd.ttype = self.ttype
        cpd._template_id = _id
        return cpd


def show_model(model, evidence=None, suff='', factor_list=None, soft_evidence={}):
    """
    References:
        http://stackoverflow.com/questions/22207802/pygraphviz-networkx-set-node-level-or-layer


    sudo apt-get install libgraphviz-dev
    pip install git+git://github.com/pygraphviz/pygraphviz.git
    sudo pip install pygraphviz

    """
    import utool as ut
    import plottool as pt
    import networkx as netx
    import pygraphviz
    import matplotlib as mpl
    fnum = pt.ensure_fnum(None)
    fig = pt.figure(fnum=fnum, doclf=True)  # NOQA
    ax = pt.gca()
    netx_graph = (model)
    #netx_graph.graph.setdefault('graph', {})['size'] = '"10,5"'
    #netx_graph.graph.setdefault('graph', {})['rankdir'] = 'LR'

    def get_hacked_pos(netx_graph):
        # Add "invisible" edges to induce an ordering
        # Hack for layout (ordering of top level nodes)
        name_nodes = sorted(ut.list_getattr(model.ttype2_cpds['name'], 'variable'))
        #netx.set_node_attributes(netx_graph, 'label', {n: {'label': n} for n in all_nodes})
        #netx.set_node_attributes(netx_graph, 'rank', {n: {'rank': 'min'} for n in name_nodes})
        invis_edges = list(ut.itertwo(name_nodes))
        netx_graph2 = netx_graph.copy()
        netx_graph2.add_edges_from(invis_edges)
        A = netx.to_agraph(netx_graph2)
        A.add_subgraph(name_nodes, rank='same')
        args = ''
        prog = 'dot'
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
    pos = get_hacked_pos(netx_graph)
    #netx.pygraphviz_layout(netx_graph)
    #pos = netx.pydot_layout(netx_graph, prog='dot')
    #pos = netx.graphviz_layout(netx_graph)

    if evidence is not None:
        node_colors = [
            (pt.TRUE_BLUE
             if node not in soft_evidence else
             pt.LIGHT_PINK)
            if node not in evidence
            else pt.FALSE_RED
            for node in netx_graph.nodes()]
        netx.draw(netx_graph, pos=pos, ax=ax, node_color=node_colors, with_labels=True, node_size=2000)
    else:
        netx.draw(netx_graph, pos=pos, ax=ax, with_labels=True, node_size=2000)

    var2_post = {f.variables[0]: f for f in factor_list}

    if True:
        netx_nodes = model.nodes(data=True)
        node_key_list = ut.get_list_column(netx_nodes, 0)
        pos_list = ut.dict_take(pos, node_key_list)

        textprops = {
            'family': 'monospace',
            'horizontalalignment': 'left',
            #'size': 8,
            'size': 12,
        }

        def make_factor_text(factor, name):
            collapse_uniform = True
            if collapse_uniform and almost_allsame(factor.values):
                # Reduce uniform text
                ftext = name + ':\nuniform(%.2f)' % (factor.values[0],)
            else:
                values = factor.values
                rowstrs = ['p(%s)=%.2f' % (','.join(n), v,)
                           for n, v in zip(zip(*factor.statenames), values)]
                idxs = ut.list_argmaxima(values)
                for idx in idxs:
                    rowstrs[idx] += '*'
                thresh = 5
                if len(rowstrs) > thresh:
                    sortx = factor.values.argsort()[::-1]
                    rowstrs = ut.take(rowstrs, sortx[0:(thresh - 1)])
                    rowstrs += ['... %d more' % ((len(values) - len(rowstrs)),)]
                ftext = name + ': \n' + '\n'.join(rowstrs)
            return ftext

        def almost_allsame(vals):
            if len(vals) == 0:
                return True
            x = vals[0]
            return np.all([np.isclose(item, x) for item in vals])

        textkw = dict(
            xycoords='data', boxcoords="offset points", pad=0.25,
            frameon=True, arrowprops=dict(arrowstyle="->"),
            #bboxprops=dict(fc=node_attr['fillcolor']),
        )

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
                text = make_factor_text(post_marg, 'post_marginal')
                prior_text = make_factor_text(prior_marg, 'prior_marginal')
            else:
                if len(evidence) == 0:
                    prior_text = make_factor_text(prior_marg, 'prior_marginal')

            if text is not None:
                offset_box = mpl.offsetbox.TextArea(text, textprops)
                artist = mpl.offsetbox.AnnotationBbox(
                    offset_box, (x + 5, y), xybox=(20., 5.),
                    box_alignment=(0, 0), **textkw)
                offset_box_list.append(offset_box)
                artist_list.append(artist)

            if prior_text:
                offset_box2 = mpl.offsetbox.TextArea(prior_text, textprops)
                artist2 = mpl.offsetbox.AnnotationBbox(
                    offset_box2, (x - 5, y), xybox=(-20., -15.),
                    box_alignment=(1, 1), **textkw)
                offset_box_list.append(offset_box2)
                artist_list.append(artist2)

        for artist in artist_list:
            ax.add_artist(artist)

        xmin, ymin = np.array(pos_list).min(axis=0)
        xmax, ymax = np.array(pos_list).max(axis=0)
        ax.set_xlim((xmin - 40, xmax + 40))
        ax.set_ylim((ymin - 20, ymax + 20))
        fig = pt.gcf()
        fig.set_size_inches(20, 7)
        pt.set_figtitle('num_names=%r' % (model.num_names,), size=14)

        def hack_fix_centeralign():
            if textprops['horizontalalignment'] == 'center':
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
    #fpath = ('name_model_' + suff + '.png')
    #pt.plt.savefig(fpath)
    #return fpath


def print_ascii_graph(model_):
    """
    pip install img2txt.py

    python -c
    """
    from PIL import Image
    from six.moves import StringIO
    import networkx as netx
    import copy
    model = copy.deepcopy(model_)
    assert model is not model_
    # model.graph.setdefault('graph', {})['size'] = '".4,.4"'
    model.graph.setdefault('graph', {})['size'] = '".3,.3"'
    model.graph.setdefault('graph', {})['height'] = '".3,.3"'
    pydot_graph = netx.to_pydot(model)
    png_str = pydot_graph.create_png(prog='dot')
    sio = StringIO()
    sio.write(png_str)
    sio.seek(0)
    pil_img = Image.open(sio)
    print('pil_img.size = %r' % (pil_img.size,))
    #def print_ascii_image(pil_img):
    #    img2txt = ut.import_module_from_fpath('/home/joncrall/venv/bin/img2txt.py')
    #    import sys
    #    pixel = pil_img.load()
    #    width, height = pil_img.size
    #    bgcolor = None
    #    #fill_string = img2txt.getANSIbgstring_for_ANSIcolor(img2txt.getANSIcolor_for_rgb(bgcolor))
    #    fill_string = "\x1b[49m"
    #    fill_string += "\x1b[K"          # does not move the cursor
    #    sys.stdout.write(fill_string)
    #    img_ansii_str = img2txt.generate_ANSI_from_pixels(pixel, width, height, bgcolor)
    #    sys.stdout.write(img_ansii_str)
    #https://gist.github.com/cdiener/10491632
    SC = 1.0
    GCF = 1.0
    WCF = 1.0
    img = pil_img
    S = (int(round(img.size[0] * SC * WCF * 3)), int(round(img.size[1] * SC)))
    img = np.sum( np.asarray( img.resize(S) ), axis=2)
    print('img.shape = %r' % (img.shape,))
    img -= img.min()
    chars = np.asarray(list(' .,:;irsXA253hMHGS#9B&@'))
    img = (1.0 - img / img.max()) ** GCF * (chars.size - 1)
    print( "\n".join( ("".join(r) for r in chars[img.astype(int)]) ) )
    pil_img = pil_img.convert('RGB')
    #pil_img = pil_img.resize((10, 10))
    pil_img.close()
    pass


def _debug_repr_model(model):
    cpd_code_list = [_debug_repr_cpd(cpd) for cpd in model.cpds]
    code_fmt = ut.codeblock(
        '''
        import numpy as np
        import pgmpy
        import pgmpy.inference
        import pgmpy.factors
        import pgmpy.models

        {cpds}

        cpd_list = {nodes}
        input_graph = {edges}
        model = pgmpy.models.BayesianModel(input_graph)
        model.add_cpds(*cpd_list)
        model_inference = pgmpy.inference.BeliefPropagation(model)
        ''')

    code = code_fmt.format(
        cpds='\n'.join(cpd_code_list),
        nodes=ut.repr2(sorted(model.nodes()), strvals=True),
        edges=ut.repr2(sorted(model.edges()), nl=1),
    )
    ut.print_code(code)
    ut.copy_text_to_clipboard(code)


def _debug_repr_cpd(cpd):
    import re
    import utool as ut
    code_fmt = ut.codeblock(
        '''
        {variable} = pgmpy.factors.TabularCPD(
            variable={variable_repr},
            variable_card={variable_card_repr},
            values={get_cpd_repr},
            evidence={evidence_repr},
            evidence_card={evidence_card_repr},
        )
        ''')
    # Parse props that are needed for this fmtstr
    fmt_keys = [match.groups()[0] for match in re.finditer('{(.*?)}', code_fmt)]
    need_reprs = [key[:-5] for key in fmt_keys if key.endswith('_repr')]
    need_keys = [key for key in fmt_keys if not key.endswith('_repr')]
    # Get corresponding props
    # Call methods if needbe
    tmp = [(prop, getattr(cpd, prop)) for prop in need_reprs]
    tmp = [(x, y()) if ut.is_funclike(y) else (x, y) for (x, y) in tmp]
    fmtdict = dict(tmp)
    fmtdict = ut.map_dict_vals(ut.repr2, fmtdict)
    fmtdict = ut.map_dict_keys(lambda x: x + '_repr', fmtdict)
    tmp2 = [(prop, getattr(cpd, prop)) for prop in need_keys]
    fmtdict.update(dict(tmp2))
    code = code_fmt.format(**fmtdict)
    return code


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
