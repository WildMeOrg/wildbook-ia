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

_PRINT = print


def bayesnet_cases():
    r"""
    CommandLine:
        python -m ibeis.model.hots.bayes --exec-bayesnet_cases

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.bayes import *  # NOQA
        >>> result = bayesnet_cases()
        >>> print(result)
    """
    model, evidence = bayesnet(2)


def bayesnet(num_annots):
    """

    CommandLine:
        python -m ibeis.model.hots.bayes --exec-bayesnet --no-flask
        python -m ibeis.model.hots.bayes --exec-bayesnet --show
        python -m ibeis.model.hots.bayes --exec-bayesnet
        python bayes.py --exec-bayesnet --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.model.hots.bayes import *  # NOQA
        >>> #from bayes import *  # NOQA
        >>> num_annots = 4
        >>> model, evidence = bayesnet(num_annots)
        >>> ut.quit_if_noshow()
        >>> show_model(model, evidence)
    """
    annots = ut.chr_range(num_annots, base='a')

    # -- Define CPD Templates
    def match_pmf(match_type, n1, n2):
        val = None
        if n1 == n2 and match_type == 'same':
            val = 1.0
        elif n1 == n2 and match_type == 'diff':
            val = 0.0
        elif n1 != n2 and match_type == 'same':
            val = 0.0
        elif n1 != n2 and match_type == 'diff':
            val = 1.0
        return val

    def score_pmf(score_type, match_type):
        val = None
        if match_type == 'same':
            val = .1 if score_type == 'low' else .9
        elif match_type == 'diff':
            val = .9 if score_type == 'low' else .1
        else:
            assert False
        return val

    name_cpd = TemplateCPD('name', ('n', 2), varpref='N')

    match_cpd = TemplateCPD('match', ['diff', 'same'], varpref='M',
                            evidence_ttypes=[name_cpd, name_cpd],
                            pmf_func=match_pmf)

    score_cpd = TemplateCPD('score', ['low', 'high'], varpref='S',
                            evidence_ttypes=[match_cpd],
                            pmf_func=score_pmf)

    ut.colorprint('\n --- CPD Templates ---', 'blue')
    ut.colorprint(
        ut.hz_str(
            name_cpd._str('p', 'psql'),
            match_cpd._str('p', 'psql'),
            score_cpd._str('p', 'psql'),
        ),
        'turquoise')

    # -- Build CPDS
    name_cpds = [name_cpd.new_cpd(_id=aid) for aid in annots]

    match_cpds = [
        match_cpd.new_cpd(evidence_cpds=cpds)
        for cpds in list(ut.iter_window(name_cpds, 2, wrap=len(name_cpds) > 2))
    ]

    score_cpds = [
        score_cpd.new_cpd(evidence_cpds=cpds)
        for cpds in zip(match_cpds)
    ]

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

    def pretty_evidence(model, evidence):
        return [evar + '=' + str(model.var2_cpd[evar].variable_statenames[val])
                for evar, val in evidence.items()]
    ut.inject_func_as_method(model, pretty_evidence)
    #print_ascii_graph(model)

    # --- INFERENCE ---
    ut.colorprint('\n --- Inference ---', 'red')

    N0 = name_cpds[0]
    event_space_combos = {}
    event_space_combos[N0.variable] = 0  # Set ni to always be Fred
    for cpd in cpd_list:
        if cpd.ttype == 'score':
            #event_space_combos[cpd.variable] = list(range(cpd.variable_card))
            event_space_combos[cpd.variable] = [1]
    #del event_space_combos['Ski']
    print('Search Space = %s' % (ut.repr3(event_space_combos, nl=1)))
    evidence_dict = ut.all_dict_combinations(event_space_combos)

    name_belief = pgmpy.inference.BeliefPropagation(model)
    #name_belief = pgmpy.inference.VariableElimination(model)

    def try_query(evidence):
        print('+--------')
        query_vars = ut.setdiff_ordered(model.nodes(), list(evidence.keys()))
        evidence_str = ', '.join(model.pretty_evidence(evidence))
        print('P(' + ', '.join(query_vars) + ' | ' + evidence_str + ') = ')
        probs = name_belief.query(query_vars, evidence)
        factor_list = probs.values()
        joint_factor = pgmpy.factors.factor_product(*factor_list)
        # print(joint_factor.get_independencies())
        # print(model.local_independencies([Ni.variable]))
        ut.colorprint(joint_factor._str('phi', 'fancy_grid', sort=True), 'white')
        #name_vars = [v for v in joint_factor.scope() if is_semtype(v, 'name')]
        #marginal = joint_factor.marginalize(name_vars, inplace=False)
        #ut.colorprint(marginal._str('phi', 'psql', sort=-1), 'white')

        #factor = joint_factor  # NOQA
        #semtypes = [var2_cpd[f.variables[0]].ttype for f in factor_list]
        #for type_, factors in ut.group_items(factor_list, semtypes).items():
        #    factors = ut.sortedby(factors, [f.variables[0] for f in factors])
        #    ut.colorprint(ut.hz_str([f.__str__() for f in factors]), 'yellow')
        print('L_____\n')
        return factor_list

    for evidence in evidence_dict:
        factor_list = try_query(evidence)  # NOQA

    if False and len(annots) == 3:
        # Specific Cases
        evidence = {'Mij': 1, 'Mjk': 1, 'Mki': 1, 'Ni': 0}
        try_query(evidence)

        evidence = {'Mij': 1, 'Mjk': 1, 'Mki': 1, 'Ni': 0}
        try_query(evidence)

        evidence = {'Mij': 0, 'Mjk': 0, 'Mki': 0, 'Ni': 0}
        try_query(evidence)

        evidence = {'Ni': 0, 'Nj': 1, 'Sij': 0, 'Sjk': 0}
        try_query(evidence)

        evidence = {'Ni': 0, 'Sij': 1, 'Sjk': 1}
        try_query(evidence)

        evidence = {'Ni': 0, 'Sij': 0, 'Sjk': 1}
        try_query(evidence)

        evidence = {'Ni': 0, 'Sij': 0, 'Sjk': 0}
        try_query(evidence)

        evidence = {'Ni': 0, 'Sij': 0, 'Sjk': 0, 'Ski': 0}
        try_query(evidence)

        evidence = {'Ni': 0, 'Sij': 1, 'Ski': 1}
        try_query(evidence)
    # print_ascii_graph(model)

    return model, evidence


class TemplateCPD(object):
    """
    Factory for templated cpds
    """
    def __init__(self, ttype, basis, varpref, evidence_ttypes=None, pmf_func=None):
        if isinstance(basis, tuple):
            state_pref, state_card = basis
            basis = [state_pref + str(i) for i in range(state_card)]
        if varpref is None:
            varpref = ttype
        self.basis = basis
        self.ttype = ttype
        self.varpref = varpref
        self.evidence_ttypes = evidence_ttypes
        self.pmf_func = pmf_func

    def __call__(self, *args, **kwargs):
        return self.new_cpd(*args, **kwargs)

    def _str(self, *args, **kwargs):
        example_cpd = self.example_cpd()
        return example_cpd._str(*args, **kwargs)

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


def show_model(model, evidence):
    #ut.embed()
    # print('Independencies')
    # print(model.get_independencies())
    # print(model.local_independencies([Ni.variable]))
    # _ draw model

    import plottool as pt
    import networkx as netx
    fig = pt.figure(doclf=True)  # NOQA
    ax = pt.gca()
    netx_graph = pgm_to_netx(model)
    pos = netx.pydot_layout(netx_graph, prog='dot')
    #values = [[0, 0, 1]]
    #values = [[1, 0, 0]]
    #node_state = evidence.copy()
    #var2_factor = {f.variables[0]: None if f is None else f.values.max() for f in factor_list}
    #node_state.update(var2_factor)
    #node_colors = ut.dict_take(node_state, netx_graph.nodes(), None)
    node_colors = [pt.TRUE_BLUE if node not in evidence else pt.FALSE_RED for node in netx_graph.nodes()]
    netx.draw(netx_graph, pos=pos, ax=ax, node_color=node_colors, with_labels=True, node_size=2000)
    pt.plt.savefig('foo.png')
    ut.startfile('foo.png')


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


def pgm_to_netx(model):
    import networkx as netx
    if isinstance(model, (netx.Graph, netx.DiGraph)):
        return model
    netx_nodes = [(node, {}) for node in model.nodes()]
    netx_edges = [(etup[0], etup[1], {}) for etup in model.edges()]
    netx_graph = netx.DiGraph() if model.is_directed() else netx.Graph()
    netx_graph.add_nodes_from(netx_nodes)
    netx_graph.add_edges_from(netx_edges)
    return netx_graph


def network_transforms_fun(model):
    import plottool as pt
    import networkx as netx
    moralgraph = model.moralize()
    fig = pt.figure()  # NOQA
    fig.clf()
    ax = pt.gca()
    netx_graph = pgm_to_netx(moralgraph)
    pos = netx.pydot_layout(netx_graph, prog='dot')
    netx.draw(netx_graph, pos=pos, ax=ax, with_labels=True)
    pt.plt.savefig('foo2.png')
    ut.startfile('foo2.png')

    fig = pt.figure(doclf=True)  # NOQA
    ax = pt.gca()
    netx_graph = moralgraph.to_directed()
    pos = netx.pydot_layout(netx_graph, prog='dot')
    netx.draw(netx_graph, pos=pos, ax=ax, with_labels=True)
    pt.plt.savefig('foo3.png')
    ut.startfile('foo3.png')

    # a junction tree is a clique tree
    jtree = model.to_junction_tree()
    # build explicit sepsets, even though they are implicit for jtrees
    for n1, n2 in jtree.edges():
        sepset = list((set.intersection(set(n1), n2)))
        jtree[n1][n2]['sepset'] = sepset
        #jtree[n1][n2]['label'] = sepset

    fig = pt.figure(doclf=True)  # NOQA
    ax = pt.gca()
    netx_graph = pgm_to_netx(jtree)
    pos = netx.pydot_layout(netx_graph, prog='dot')
    netx.draw(netx_graph, pos=pos, ax=ax, with_labels=True, node_size=2000)
    netx.draw_networkx_edge_labels(netx_graph, pos, edge_labels=netx.get_edge_attributes(netx_graph, 'sepset'), font_size=8)
    pt.plt.savefig('foo4.png')
    ut.startfile('foo4.png')

    ut.help_members(model)
    ut.help_members(jtree)
    ut.help_members(moralgraph)


def bayesnet_examples():
    import pandas as pd
    student_model = pgmpy.models.BayesianModel(
        [('D', 'G'),
         ('I', 'G'),
         ('G', 'L'),
         ('I', 'S')])
    # we can generate some random data.
    raw_data = np.random.randint(low=0, high=2, size=(1000, 5))
    data = pd.DataFrame(raw_data, columns=['D', 'I', 'G', 'L', 'S'])
    data_train = data[: int(data.shape[0] * 0.75)]
    student_model.fit(data_train)
    student_model.get_cpds()

    data_test = data[int(0.75 * data.shape[0]): data.shape[0]]
    data_test.drop('D', axis=1, inplace=True)
    student_model.predict(data_test)

    grade_cpd = pgmpy.factors.TabularCPD(
        variable='G',
        variable_card=3,
        values=[[0.3, 0.05, 0.9, 0.5],
                [0.4, 0.25, 0.08, 0.3],
                [0.3, 0.7, 0.02, 0.2]],
        evidence=['I', 'D'],
        evidence_card=[2, 2])
    difficulty_cpd = pgmpy.factors.TabularCPD(
        variable='D',
        variable_card=2,
        values=[[0.6, 0.4]])
    intel_cpd = pgmpy.factors.TabularCPD(
        variable='I',
        variable_card=2,
        values=[[0.7, 0.3]])
    letter_cpd = pgmpy.factors.TabularCPD(
        variable='L',
        variable_card=2,
        values=[[0.1, 0.4, 0.99],
                [0.9, 0.6, 0.01]],
        evidence=['G'],
        evidence_card=[3])
    sat_cpd = pgmpy.factors.TabularCPD(
        variable='S',
        variable_card=2,
        values=[[0.95, 0.2],
                [0.05, 0.8]],
        evidence=['I'],
        evidence_card=[2])
    student_model.add_cpds(grade_cpd, difficulty_cpd,
                           intel_cpd, letter_cpd,
                           sat_cpd)


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

# def print_cpd(cpd):
#     print('CPT: %r' % (cpd,))
#     index = semtype2_nice[cpd.ttype]
#     if cpd.evidence is None:
#         columns = ['None']
#     else:
#         basis_lists = [semtype2_nice[var2_cpd[ename].ttype] for ename in cpd.evidence]
#         columns = [','.join(x) for x in ut.iprod(*basis_lists)]
#     data = cpd.get_cpd()
#     ut.colorprint(str(pd.DataFrame(data, index=index, columns=columns)), 'turquoise')

# def print_factor(factor):
#     row_cards = factor.cardinality
#     row_vars = factor.variables
#     values = factor.values.reshape(np.prod(row_cards), 1).flatten()
#     # col_cards = 1
#     # col_vars = ['']
#     basis_lists = list(zip(*list(ut.iprod(*[range(c) for c in row_cards]))))
#     nice_basis_lists = []
#     for varname, basis in zip(row_vars, basis_lists):
#         cpd = var2_cpd[varname]
#         _nice_basis = ut.take(semtype2_nice[cpd.ttype], basis)
#         nice_basis = ['%s=%s' % (varname, val) for val in _nice_basis]
#         nice_basis_lists.append(nice_basis)
#     row_lbls = [', '.join(sorted(x)) for x in zip(*nice_basis_lists)]
#     dict_ = dict(zip(row_lbls, values))
#     repr_ = ut.repr3(dict_, precision=3, align=True, key_order_metric='-val', maxlen=8)
#     print(repr_)

# # ProbMatch CPDS ---
# def probmatch_cpd(aid1, aid2):
#     """
#     aid1, aid2 = 'i', 'j'
#     """
#     ttype = 'probmatch'
#     variable = 'B' + aid1 + aid2
#     variable_basis = semtype2_nice[ttype]
#     statename_dict = {
#         variable: variable_basis,
#     }
#     evidence = ['M' + aid1 + aid2, 'S' + aid1 + aid2]
#     evidence_cpds = [var2_cpd[key] for key in evidence]
#     evidence_nice = [semtype2_nice[cpd.ttype] for cpd in evidence_cpds]
#     statename_dict.update(dict(zip(evidence, evidence_nice)))
#     evidence_card = list(map(len, evidence_nice))
#     evidence_states = list(ut.iprod(*evidence_nice))
#     def samediff_pmf(probmatch_type, match_type, score_type):
#         val = None
#         if match_type == 'same':
#             if probmatch_type == 'psame':
#                 val = .9 if score_type == 'high' else .5
#             elif probmatch_type == 'pdiff':
#                 val = .1 if score_type == 'high' else .5
#         elif match_type == 'diff':
#             if probmatch_type == 'psame':
#                 val = .5 if score_type == 'high' else .1
#             elif probmatch_type == 'pdiff':
#                 val = .5 if score_type == 'high' else .9
#         return val
#     variable_values = []
#     for score_type in variable_basis:
#         row = []
#         for state in evidence_states:
#             # row.append(samediff_pmf(score_type, state[0], state[1]))
#             row.append(samediff_pmf(score_type, state[0], state[1]))
#         variable_values.append(row)
#     cpd = pgmpy.factors.TabularCPD(
#         variable=variable,
#         variable_card=len(variable_basis),
#         values=variable_values,
#         evidence=evidence,
#         evidence_card=evidence_card,
#         statename_dict=statename_dict,
#     )
#     cpd.ttype = ttype
#     return cpd
# #probmatch_cdfs = [probmatch_cpd(*aids)
# #                  for aids in list(ut.iter_window(annots, 2, wrap=len(annots) > 2))]
# #var2_cpd.update(dict(zip([cpd.variable for cpd in probmatch_cdfs], probmatch_cdfs)))


# Match CPDS ---
# def samediff_cpd(aid1, aid2):
#     ttype = 'name'
#     ttype = 'match'
#     var_card = len(semtype2_nice[ttype])
#     variable = 'M' + aid1 + aid2
#     from pgmpy.factors import TabularCPD
#     cpd = TabularCPD(
#         variable=variable,
#         variable_card=var_card,
#         values=[[1.0 / var_card] * var_card])
#     cpd.ttype = ttype
#     return cpd
# samediff_cpds = [samediff_cpd(*aids)
#                  for aids in list(ut.iter_window(annots, 2, wrap=len(annots) > 2))]
# var2_cpd.update(dict(zip([cpd.variable for cpd in samediff_cpds], samediff_cpds)))

# Score CPDS ---
# def score_cpd(aid1, aid2):
#     """
#     aid1, aid2 = 'i', 'j'
#     """
#     variable = 'S' + aid1 + aid2
#     ttype = 'score'
#     variable_basis = semtype2_nice[ttype]
#     variable_values = [[.6, .4]]
#     cpd = pgmpy.factors.TabularCPD(
#         variable=variable,
#         variable_card=len(variable_basis),
#         values=variable_values,
#     )
#     cpd.ttype = ttype
#     return cpd
