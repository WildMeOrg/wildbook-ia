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
    http://www.csse.monash.edu.au/bai/book/BAI_Chapter2.pdf

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
               other_evidence={}, show_prior=False, **kwargs):
    verbose = ut.VERBOSE

    model = make_name_model(num_annots, num_names, verbose=verbose, **kwargs)

    if verbose:
        ut.colorprint('\n --- Priors ---', 'darkblue')
        for ttype, cpds in model.ttype2_cpds.items():
            if ttype != 'match':
                for fs_ in ut.ichunks(cpds, 4):
                    ut.colorprint(ut.hz_str([f._cpdstr('psql') for f in fs_]), 'purple')

    name_cpds = model.ttype2_cpds['name']
    score_cpds = model.ttype2_cpds['score']

    evidence = {}
    evidence.update(other_evidence)
    soft_evidence = {}
    #for key, val in other_evidence.items():
    #    if isinstance(val, int):
    #        evidence[key] = model.var2_cpd[key]._internal_varindex(val)
    #    else:
    #        evidence[key] = model.var2_cpd[key]._internal_varindex(val)

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

    if verbose:
        ut.colorprint('\n --- Soft Evidence ---', 'white')
        for ttype, cpds in model.ttype2_cpds.items():
            if ttype != 'match':
                for fs_ in ut.ichunks(cpds, 4):
                    ut.colorprint(ut.hz_str([f._cpdstr('psql') for f in fs_]),
                                  'green')

    if verbose:
        ut.colorprint('\n --- Inference ---', 'red')

    if len(evidence) > 0 or len(soft_evidence) > 0:
        model_inference = pgmpy.inference.BeliefPropagation(model)
        evidence = model_inference._ensure_internal_evidence(evidence, model)
        #model_inference = pgmpy.inference.VariableElimination(model)
        factor_list = try_query(model, model_inference, evidence, verbose=verbose)
    else:
        factor_list = []

    show_model(model, evidence, '', factor_list, soft_evidence, show_prior=show_prior)
    return (model,)
    # print_ascii_graph(model)


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


def make_name_model(num_annots, num_names=None, verbose=True, mode=1):
    """
    Defines the general name model
    """
    #annots = ut.chr_range(num_annots, base='a')
    mode = ut.get_argval('--mode', default=mode)
    annots = ut.chr_range(num_annots, base=ut.get_argval('--base', default='a'))
    # The indexes of match CPDs will not change if another annotation is added
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
        if match_type == 'same':
            val = .1 if score_type == 'low' else .9
        elif match_type == 'diff':
            val = .9 if score_type == 'low' else .1
        else:
            raise ValueError('Unknown match_type=%r' % (match_type,))
        return val

    def score_pmf2(score_type, n1, n2):
        if n1 == n2:
            val = .1 if score_type == 'low' else .9
        else:
            val = .9 if score_type == 'low' else .1
        return val

    special_basis_pool = ['fred', 'sue', 'paul']

    name_cpd = pgm_ext.TemplateCPD(
        'name', ('n', num_names), varpref='N',
        special_basis_pool=special_basis_pool)

    if mode == 1:
        match_cpd = pgm_ext.TemplateCPD(
            'match', ['diff', 'same'], varpref='M',
            evidence_ttypes=[name_cpd, name_cpd], pmf_func=match_pmf)
        score_cpd = pgm_ext.TemplateCPD(
            'score', ['low', 'high'], varpref='S',
            evidence_ttypes=[match_cpd], pmf_func=score_pmf)
        templates = [name_cpd, match_cpd, score_cpd]

    elif mode == 2:
        name_cpd = pgm_ext.TemplateCPD(
            'name', ('n', num_names), varpref='N',
            special_basis_pool=special_basis_pool)
        score_cpd = pgm_ext.TemplateCPD(
            'score', ['low', 'high'], varpref='S', evidence_ttypes=[name_cpd, name_cpd],
            pmf_func=score_pmf2)
        templates = [name_cpd, score_cpd]

    # Instanciate templates
    name_cpds = [name_cpd.new_cpd(parents=aid) for aid in annots]
    parent_cpds = ut.list_unflat_take(name_cpds, upper_diag_idxs)

    if mode == 1:
        match_cpds = [match_cpd.new_cpd(parents=cpds)
                      for cpds in parent_cpds]
        score_cpds = [score_cpd.new_cpd(parents=cpds)
                      for cpds in zip(match_cpds)]
        cpd_list = name_cpds + score_cpds + match_cpds
    elif mode == 2:
        score_cpds = [score_cpd.new_cpd(parents=cpds)
                      for cpds in parent_cpds]
        cpd_list = name_cpds + score_cpds

    if verbose:
        ut.colorprint('\n --- CPD Templates ---', 'blue')
        for temp_cpd in templates:
            ut.colorprint(temp_cpd._cpdstr('psql'), 'turquoise')

    # Make Model
    model = pgm_ext.define_model(cpd_list)

    model.num_names = num_names
    #print_ascii_graph(model)
    return model


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
    pt.set_figtitle('Junction Tree')


def get_hacked_pos(netx_graph, name_nodes=None, prog='dot'):
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


def show_model(model, evidence=None, suff='', factor_list=None, soft_evidence={}, show_prior=False):
    """
    References:
        http://stackoverflow.com/questions/22207802/pygraphviz-networkx-set-node-level-or-layer

    Ignore:
        sudo apt-get install libgraphviz-dev
        pip install git+git://github.com/pygraphviz/pygraphviz.git
        sudo pip install pygraphviz
    """
    import plottool as pt
    import networkx as netx
    import matplotlib as mpl
    fnum = pt.ensure_fnum(None)
    fig = pt.figure(fnum=fnum, doclf=True)  # NOQA
    ax = pt.gca()
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
            (pt.TRUE_BLUE
             if node not in soft_evidence else
             pt.LIGHT_PINK)
            if node not in evidence
            else pt.FALSE_RED
            for node in netx_graph.nodes()]
        drawkw['node_color'] = node_colors
    netx.draw(netx_graph, **drawkw)

    if True:
        textprops = {
            'family': 'monospace',
            'horizontalalignment': 'left',
            #'size': 8,
            'size': 12,
        }

        textkw = dict(
            xycoords='data', boxcoords="offset points", pad=0.25,
            frameon=True, arrowprops=dict(arrowstyle="->"),
            #bboxprops=dict(fc=node_attr['fillcolor']),
        )

        netx_nodes = model.nodes(data=True)
        node_key_list = ut.get_list_column(netx_nodes, 0)
        pos_list = ut.dict_take(pos, node_key_list)
        var2_post = {f.variables[0]: f for f in factor_list}

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
                    offset_box, (x + 5, y), xybox=(20., 5.),
                    box_alignment=(0, 0), **textkw)
                offset_box_list.append(offset_box)
                artist_list.append(artist)

            if show_prior and prior_text is not None:
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
        num_annots = len(model.ttype2_cpds['name'])
        if num_annots > 4:
            ax.set_xlim((xmin - 40, xmax + 40))
            ax.set_ylim((ymin - 30, ymax + 30))
            fig.set_size_inches(30, 7)
        else:
            ax.set_xlim((xmin - 42, xmax + 42))
            ax.set_ylim((ymin - 30, ymax + 30))
            fig.set_size_inches(23, 7)
        fig = pt.gcf()
        pt.set_figtitle('num_names=%r, num_annots=%r' %
                        (model.num_names, num_annots,), size=14)

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
