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

    fpath = test_model(count(), (2, 4), high_idx=[], force_names=[])  # init
    fpath = test_model(count(), (2, 4), high_idx=[0], force_names=['n0'])  # Start with 4 names.
    fpath = test_model(count(), (2, 5), high_idx=[0], force_names=['n0'])  # Add a name, Causes probability of match to go down
    fpath = test_model(count(), (3, 5), high_idx=[0], force_names=['n0'])  # Add Annotation
    fpath = test_model(count(), (3, 5), high_idx=[0, 2], force_names=['n0'])

    fpath = test_model(count(), (3, 5), high_idx=[0, 2], force_names=['n0', {('n0',): .9}])
    fpath = test_model(count(), (3, 5), high_idx=[0], force_names=['n0', {('n0',): .9}])
    fpath = test_model(count(), (3, 5), high_idx=[0], force_names=[{('n0',): .99}, {('n0',): .9}])
    fpath = test_model(count(), (3, 10), high_idx=[0], force_names=[{('n0',): .99}, {('n0',): .9}])
    fpath = test_model(count(), (3, 10), high_idx=[0], force_names=[{('n0',): .99}, {('n0',): .2, ('n1',): .7}])

    fpath = test_model(count(), (3, 10), high_idx=[0, 1], force_names=[{('n0',): .99}, {('n0',): .2, ('n1',): .7}])
    fpath = test_model(count(), (3, 10), high_idx=[0, 1], force_names=[{('n0',): .99}, {('n0',): .2, ('n1',): .7}, {('n0',): .32}])
    fpath = test_model(count(), (3, 10), high_idx=[0, 1, 2], force_names=[{('n0',): .99}, {('n0',): .2, ('n1',): .7}, {('n0',): .32}])
    # Fix indexing to move in diagonal order as opposed to row order
    fpath = test_model(count(), (4, 10), high_idx=[0, 1, 3], force_names=[{('n0',): .99}, {('n0',): .2, ('n1',): .7}, {('n0',): .32}])
    fpath = test_model(count(), (4, 10),
                       high_idx=[0, 1, 3], low_idx=[2], force_names=[{('n0',): .99}, {('n0',): .2, ('n1',): .7}, {('n0',): .32}])

    fpath = test_model(count(), (4, 10))

    ut.startfile(fpath)

    #model = make_name_model(5, 10)
    #evidence = test_model(model)


def test_model(test_idx, model=(2, 2), high_idx=[], low_idx=[], force_names=[]):
    #if test_idx < 13:
    #    # hack
    #    return

    verbose = False
    if verbose:
        print('___ TEST %d ___' % (test_idx,))

    if isinstance(model, tuple):
        model = make_name_model(*model, verbose=verbose)

    if verbose:
        ut.colorprint('\n --- Inference ---', 'red')

    name_cdfs = model.ttype2_cpds['name']
    score_cdfs = model.ttype2_cpds['score']

    evidence = {}

    # Set ni to always be Fred
    #N0 = name_cdfs[0]
    soft_evidence = {}
    for Ni, force in zip(name_cdfs, force_names):
        if isinstance(force, six.string_types):
            evidence[Ni.variable] = Ni.statename_to_index(Ni.variable, force)
        if isinstance(force, dict):
            # soft evidence
            fill = (1 - sum(force.values())) / (len(Ni.values) - len(force))
            assert fill >= 0
            row_labels = list(ut.iprod(*Ni.statenames))
            for i, lbl in enumerate(row_labels):
                if lbl in force:
                    Ni.values[i] = force[lbl]
                else:
                    Ni.values[i] = fill
            soft_evidence[Ni.variable] = True
            #ut.embed()
            pass

    for idx in high_idx:
        evidence[score_cdfs[idx].variable] = 1
    for idx in low_idx:
        evidence[score_cdfs[idx].variable] = 0

    model_inference = pgmpy.inference.BeliefPropagation(model)
    #model_inference = pgmpy.inference.VariableElimination(model)

    factor_list = try_query(model, model_inference, evidence, verbose=verbose)

    return show_model(model, evidence,  str(test_idx), factor_list, soft_evidence)
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

    name_cpd = TemplateCPD('name', ('n', num_names), varpref='N')

    match_cpd = TemplateCPD('match', ['diff', 'same'], varpref='M',
                            evidence_ttypes=[name_cpd, name_cpd],
                            pmf_func=match_pmf)

    score_cpd = TemplateCPD('score', ['low', 'high'], varpref='S',
                            evidence_ttypes=[match_cpd],
                            pmf_func=score_pmf)

    PRINT_TEMPLATES = False
    if PRINT_TEMPLATES:
        ut.colorprint('\n --- CPD Templates ---', 'blue')
        ut.colorprint(name_cpd._cpdstr('psql'), 'turquoise')
        ut.colorprint(match_cpd._cpdstr('psql'), 'turquoise')
        ut.colorprint(score_cpd._cpdstr('psql'), 'turquoise')

    # -- Build CPDS
    name_cpds = [name_cpd.new_cpd(_id=aid) for aid in annots]

    match_cpds = [
        match_cpd.new_cpd(evidence_cpds=cpds)
        #for cpds in list(ut.iter_window(name_cpds, 2, wrap=len(name_cpds) > 2))
        for cpds in list(ut.upper_diag_self_prodx(name_cpds))
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
    import utool as ut
    #ut.embed()
    # print('Independencies')
    # print(model.get_independencies())
    # print(model.local_independencies([Ni.variable]))
    # _ draw model

    import plottool as pt
    import networkx as netx
    fig = pt.figure(doclf=True)  # NOQA
    ax = pt.gca()
    netx_graph = (model)
    if True:
        pos = netx.pydot_layout(netx_graph, prog='dot')
    else:
        pos = netx.graphviz_layout(netx_graph)
    #model.graph.setdefault('graph', {})['size'] = '"10,5"'

    #values = [[0, 0, 1]]
    #values = [[1, 0, 0]]
    #node_state = evidence.copy()
    #var2_factor = {f.variables[0]: None if f is None else f.values.max() for f in factor_list}
    #node_state.update(var2_factor)
    #node_colors = ut.dict_take(node_state, netx_graph.nodes(), None)
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

    var2_factor = {f.variables[0]: f for f in factor_list}

    if True:
        import matplotlib as mpl
        pt.set_figtitle('num_names=%r' % (model.num_names,))

        #import utool
        netx_nodes = model.nodes(data=True)
        node_key_list = ut.get_list_column(netx_nodes, 0)
        pos_list = ut.dict_take(pos, node_key_list)

        textprops = {
            'family': 'monospace',
            'horizontalalignment': 'left',
            'size': 8,
        }

        artist_list = []
        offset_box_list = []
        for pos_, node in zip(pos_list, netx_nodes):
            x, y = pos_
            variable = node[0]

            text = None
            cpd = model.var2_cpd[variable]

            def almost_allsame(vals):
                if len(vals) == 0:
                    return True
                x = vals[0]
                return np.all([np.isclose(item, x) for item in vals])

            if cpd.evidence is None:
                prior = cpd
            else:
                prior = cpd.marginalize(cpd.evidence, inplace=False)
            if almost_allsame(prior.values):
                prior_text = 'prior:\nuniform'
            else:
                rowstrs = ['p(%s)=%.2f' % (','.join(n), v,) for n, v in zip(zip(*prior.statenames), prior.values)]
                if len(rowstrs) > 3:
                    sortx = prior.values.argsort()[::-1]
                    rowstrs = ut.take(rowstrs, sortx[0:3])
                    rowstrs += ['...']
                prior_text = 'prior: \n' + '\n'.join(rowstrs)

            if variable in evidence:
                text = cpd.variable_statenames[evidence[variable]]
                prior_text = None
            elif variable in var2_factor:
                factor = var2_factor[variable]
                if almost_allsame(factor.values):
                    text = 'post:\nuniform'
                else:
                    rowstrs = ['p(%s)=%.2f' % (','.join(n), v,) for n, v in zip(zip(*factor.statenames), factor.values)]
                    if len(rowstrs) > 3:
                        sortx = factor.values.argsort()[::-1]
                        rowstrs = ut.take(rowstrs, sortx[0:3])
                        rowstrs += ['...']
                    text = 'post: \n' + '\n'.join(rowstrs)

            if text is not None:
                offset_box = mpl.offsetbox.TextArea(text, textprops)
                artist = mpl.offsetbox.AnnotationBbox(
                    offset_box, (x + 5, y), xybox=(20., 5.),
                    xycoords='data', boxcoords="offset points",
                    pad=0.25, frameon=True,
                    box_alignment=(0, 0),
                    #bboxprops=dict(fc=node_attr['fillcolor']),
                    arrowprops=dict(arrowstyle="->"),
                )
                offset_box_list.append(offset_box)
                artist_list.append(artist)

            if prior_text:
                offset_box2 = mpl.offsetbox.TextArea(prior_text, textprops)
                artist2 = mpl.offsetbox.AnnotationBbox(
                    offset_box2, (x - 5, y), xybox=(-20., -15.),
                    xycoords='data', boxcoords="offset points",
                    pad=0.25, frameon=True,
                    box_alignment=(1, 1),
                    #bboxprops=dict(fc=node_attr['fillcolor']),
                    arrowprops=dict(arrowstyle="->"),
                )
                offset_box_list.append(offset_box2)
                artist_list.append(artist2)

        for artist in artist_list:
            ax.add_artist(artist)

        xmin, ymin = np.array(pos_list).min(axis=0)
        xmax, ymax = np.array(pos_list).max(axis=0)
        ax.set_xlim((xmin - 30, xmax + 30))
        fig = pt.gcf()
        fig.set_size_inches(14, 6)

        if textprops['horizontalalignment'] == 'center':
            fig = pt.gcf()
            fig.canvas.draw()

            # Superhack for centered text
            # Fix bug in
            # /usr/local/lib/python2.7/dist-packages/matplotlib/offsetbox.py
            # /usr/local/lib/python2.7/dist-packages/matplotlib/text.py
            for offset_box in offset_box_list:
                offset_box.set_offset
                #offset_box.get_offset
                #self = offset_box
                z = offset_box._text.get_window_extent()
                (z.x1 - z.x0) / 2
                offset_box._text
                T = offset_box._text.get_transform()
                A = mpl.transforms.Affine2D()
                A.clear()
                A.translate((z.x1 - z.x0) / 2, 0)
                offset_box._text.set_transform(T + A)
    fpath = ('foo' + suff + '.png')

    pt.plt.savefig(fpath)
    return fpath


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
#def test_model(model, high_idx=[0]):
#    print('___ TEST ___')
#    # --- INFERENCE ---
#    ut.colorprint('\n --- Inference ---', 'red')
#    event_space_combos = {}
#    # Set ni to always be Fred
#    N0 = model.ttype2_cpds['name'][0]

#    event_space_combos[N0.variable] = 0
#    for cpd in model.get_cpds():
#        if cpd.ttype == 'score':
#            #event_space_combos[cpd.variable] = list(range(cpd.variable_card))
#            event_space_combos[cpd.variable] = [1]
#    #del event_space_combos['Ski']
#    print('Search Space = %s' % (ut.repr3(event_space_combos, nl=1)))
#    evidence_dict = ut.all_dict_combinations(event_space_combos)
#    #_debug_repr_model(model)
#    model_inference = pgmpy.inference.BeliefPropagation(model)
#    #model_inference = pgmpy.inference.VariableElimination(model)

#    for evidence in evidence_dict:
#        try_query(model, model_inference, evidence)

#    # print_ascii_graph(model)
#    return evidence
