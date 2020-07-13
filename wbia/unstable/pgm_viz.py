# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import networkx as nx
import six  # NOQA
import utool as ut
import numpy as np
from six.moves import zip

print, rrr, profile = ut.inject2(__name__)


def print_ascii_graph(model_):
    """
    pip install img2txt.py

    python -c
    """
    from PIL import Image  # NOQA
    from six.moves import StringIO

    # import networkx as nx
    import copy

    model = copy.deepcopy(model_)
    assert model is not model_
    # model.graph.setdefault('graph', {})['size'] = '".4,.4"'
    model.graph.setdefault('graph', {})['size'] = '".3,.3"'
    model.graph.setdefault('graph', {})['height'] = '".3,.3"'
    pydot_graph = nx.to_pydot(model)
    png_str = pydot_graph.create_png(prog='dot')
    sio = StringIO()
    sio.write(png_str)
    sio.seek(0)
    pil_img = Image.open(sio)  # NOQA
    print('pil_img.size = %r' % (pil_img.size,))
    # def print_ascii_image(pil_img):
    #    img2txt = ut.import_module_from_fpath('/home/joncrall/venv/bin/img2txt.py')
    #    import sys
    #    pixel = pil_img.load()
    #    width, height = pil_img.size
    #    bgcolor = None
    #    #fill_string =
    #    img2txt.getANSIbgstring_for_ANSIcolor(img2txt.getANSIcolor_for_rgb(bgcolor))
    #    fill_string = "\x1b[49m"
    #    fill_string += "\x1b[K"          # does not move the cursor
    #    sys.stdout.write(fill_string)
    #    img_ansii_str = img2txt.generate_ANSI_from_pixels(pixel, width, height, bgcolor)
    #    sys.stdout.write(img_ansii_str)

    def print_ascii_image(pil_img):
        # https://gist.github.com/cdiener/10491632
        SC = 1.0
        GCF = 1.0
        WCF = 1.0
        img = pil_img
        S = (int(round(img.size[0] * SC * WCF * 3)), int(round(img.size[1] * SC)))
        img = np.sum(np.asarray(img.resize(S)), axis=2)
        print('img.shape = %r' % (img.shape,))
        img -= img.min()
        chars = np.asarray(list(' .,:;irsXA253hMHGS#9B&@'))
        img = (1.0 - img / img.max()) ** GCF * (chars.size - 1)
        print('\n'.join((''.join(r) for r in chars[img.astype(int)])))

    print_ascii_image(pil_img)
    pil_img.close()
    pass


def _debug_repr_model(model):
    cpd_code_list = [_debug_repr_cpd(cpd) for cpd in model.cpds]
    code_fmt = ut.codeblock(
        """
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
        infr = pgmpy.inference.BeliefPropagation(model)
        """
    )

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
        """
        {variable} = pgmpy.factors.TabularCPD(
            variable={variable_repr},
            variable_card={variable_card_repr},
            values={get_cpd_repr},
            evidence={evidence_repr},
            evidence_card={evidence_card_repr},
        )
        """
    )
    keys = ['variable', 'variable_card', 'values', 'evidence', 'evidence_card']
    dict_ = ut.odict(zip(keys, [getattr(cpd, key) for key in keys]))
    # HACK
    dict_['values'] = cpd.get_cpd()
    r = ut.repr2(dict_, explicit=True, nobraces=True, nl=True)
    print(r)

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


def make_factor_text(factor, name):
    collapse_uniform = True
    if collapse_uniform and ut.almost_allsame(factor.values):
        # Reduce uniform text
        ftext = name + ':\nuniform(%.3f)' % (factor.values[0],)
    else:
        values = factor.values
        try:
            rowstrs = [
                'p(%s)=%.3f' % (','.join(n), v,)
                for n, v in zip(zip(*factor.statenames), values)
            ]
        except Exception:
            rowstrs = [
                'p(%s)=%.3f' % (','.join(n), v,)
                for n, v in zip(factor._row_labels(False), values)
            ]
        idxs = ut.list_argmaxima(values)
        for idx in idxs:
            rowstrs[idx] += '*'
        thresh = 4
        always_sort = True
        if len(rowstrs) > thresh:
            sortx = factor.values.argsort()[::-1]
            rowstrs = ut.take(rowstrs, sortx[0 : (thresh - 1)])
            rowstrs += ['... %d more' % ((len(values) - len(rowstrs)),)]
        elif always_sort:
            sortx = factor.values.argsort()[::-1]
            rowstrs = ut.take(rowstrs, sortx)
        ftext = name + ': \n' + '\n'.join(rowstrs)
    return ftext


def get_bayesnet_layout(model, name_nodes=None, prog='dot'):
    """
    Ensures ordering of layers is in order of addition via templates
    """
    import pygraphviz
    import networkx as nx

    # Add "invisible" edges to induce an ordering
    # Hack for layout (ordering of top level nodes)
    netx_graph2 = model.copy()

    if getattr(model, 'ttype2_cpds', None) is not None:
        grouped_nodes = []
        for ttype in model.ttype2_cpds.keys():
            ttype_cpds = model.ttype2_cpds[ttype]
            # use defined ordering
            ttype_nodes = ut.list_getattr(ttype_cpds, 'variable')
            # ttype_nodes = sorted(ttype_nodes)
            invis_edges = list(ut.itertwo(ttype_nodes))
            netx_graph2.add_edges_from(invis_edges)
            grouped_nodes.append(ttype_nodes)

        agraph = nx.nx_agraph.to_agraph(netx_graph2)
        for nodes in grouped_nodes:
            agraph.add_subgraph(nodes, rank='same')
    else:
        agraph = nx.nx_agraph.to_agraph(netx_graph2)
    print(agraph)

    args = ''
    agraph.layout(prog=prog, args=args)
    # agraph.draw('example.png', prog='dot')
    node_pos = {}
    for n in model:
        node_ = pygraphviz.Node(agraph, n)
        try:
            xx, yy = node_.attr['pos'].split(',')
            node_pos[n] = (float(xx), float(yy))
        except Exception:
            print('no position for node', n)
            node_pos[n] = (0.0, 0.0)
    return node_pos


def draw_map_histogram(top_assignments, fnum=None, pnum=(1, 1, 1)):
    import wbia.plottool as pt

    bin_labels = ut.get_list_column(top_assignments, 0)
    bin_vals = ut.get_list_column(top_assignments, 1)
    fnum = pt.ensure_fnum(fnum)
    # bin_labels = ['\n'.join(ut.textwrap.wrap(_lbl, width=30)) for _lbl in bin_labels]
    pt.draw_histogram(
        bin_labels,
        bin_vals,
        fnum=fnum,
        pnum=pnum,
        transpose=True,
        use_darkbackground=False,
        # xtick_rotation=-10,
        ylabel='Prob',
        xlabel='assignment',
    )
    pt.set_title('Assignment probabilities')


def get_node_viz_attrs(
    model, evidence, soft_evidence, factor_list, ttype_colors, **kwargs
):
    import wbia.plottool as pt

    var2_post = {f.variables[0]: f for f in factor_list}

    pos_dict = get_bayesnet_layout(model)

    # pos_dict = nx.pygraphviz_layout(netx_graph)
    # pos_dict = nx.pydot_layout(netx_graph, prog='dot')
    # pos_dict = nx.graphviz_layout(netx_graph)

    netx_nodes = model.nodes(data=True)
    node_key_list = ut.get_list_column(netx_nodes, 0)
    pos_list = ut.dict_take(pos_dict, node_key_list)

    prior_text = None
    post_text = None
    evidence_tas = []
    post_tas = []
    prior_tas = []
    node_color = []

    has_inferred = evidence or var2_post
    if has_inferred:
        ignore_prior_with_ttype = ['score', 'match']
        show_prior = False
    else:
        ignore_prior_with_ttype = []
        # show_prior = True
        show_prior = False

    dpy = 5
    dbx, dby = (20, 20)
    takw1 = {'bbox_align': (0.5, 0), 'pos_offset': [0, dpy], 'bbox_offset': [dbx, dby]}
    takw2 = {
        'bbox_align': (0.5, 1),
        'pos_offset': [0, -dpy],
        'bbox_offset': [-dbx, -dby],
    }

    def get_name_color(phi):
        order = phi.values.argsort()[::-1]
        if len(order) < 2:
            dist_next = phi.values[order[0]]
        else:
            dist_next = phi.values[order[0]] - phi.values[order[1]]
        dist_total = phi.values[order[0]]
        confidence = (dist_total * dist_next) ** (2.5 / 4)
        # print('confidence = %r' % (confidence,))
        color = ttype_colors['name'][order[0]]
        color = pt.color_funcs.desaturate_rgb(color, 1 - confidence)
        color = np.array(color)
        return color

    for node, pos in zip(netx_nodes, pos_list):
        variable = node[0]
        cpd = model.var2_cpd[variable]
        prior_marg = (
            cpd if cpd.evidence is None else cpd.marginalize(cpd.evidence, inplace=False)
        )

        show_evidence = variable in evidence
        show_prior = cpd.ttype not in ignore_prior_with_ttype
        show_post = variable in var2_post
        show_prior |= cpd.ttype not in ignore_prior_with_ttype

        show_prior &= kwargs.get('show_prior', True)

        post_marg = None

        if show_post:
            post_marg = var2_post[variable]

        if variable in evidence:
            if cpd.ttype == 'score':
                color = ttype_colors['score'][evidence[variable]]
                color = np.array(color)
                node_color.append(color)
            elif cpd.ttype == 'name':
                color = ttype_colors['name'][evidence[variable]]
                color = np.array(color)
                node_color.append(color)
            else:
                color = pt.FALSE_RED
                node_color.append(color)
        # elif variable in soft_evidence:
        #    color = pt.LIGHT_PINK
        #    show_prior = True
        #    color = get_name_color(prior_marg)
        #    node_color.append(color)
        else:
            if cpd.ttype == 'name' and post_marg is not None:
                color = get_name_color(post_marg)
                node_color.append(color)
            elif cpd.ttype == 'match' and post_marg is not None:
                # color = cmap(post_marg.values[1])
                # color = pt.lighten_rgb(color, .4)
                # color = np.array(color)
                color = pt.NEUTRAL
                color = pt.LIGHT_PINK
                node_color.append(color)
            else:
                # color = pt.WHITE
                color = pt.NEUTRAL
                node_color.append(color)

        if show_prior:
            if variable in soft_evidence:
                prior_color = pt.LIGHT_PINK
            else:
                prior_color = None
            prior_text = make_factor_text(prior_marg, 'prior')
            prior_tas.append(dict(text=prior_text, pos=pos, color=prior_color, **takw2))
        if show_evidence:
            _takw1 = takw1
            if cpd.ttype == 'score':
                _takw1 = takw2
            evidence_text = cpd.variable_statenames[evidence[variable]]
            if isinstance(evidence_text, int):
                evidence_text = '%d/%d' % (evidence_text + 1, cpd.variable_card)
            evidence_tas.append(dict(text=evidence_text, pos=pos, color=color, **_takw1))
        if show_post:
            _takw1 = takw1
            if cpd.ttype == 'match':
                _takw1 = takw2
            post_text = make_factor_text(post_marg, 'post')
            post_tas.append(dict(text=post_text, pos=pos, color=None, **_takw1))

    def trnps_(dict_list):
        """ tranpose dict list """
        list_dict = ut.ddict(list)
        for dict_ in dict_list:
            for key, val in dict_.items():
                list_dict[key + '_list'].append(val)
        return list_dict

    takw1_ = trnps_(post_tas + evidence_tas)
    takw2_ = trnps_(prior_tas)
    takws = [takw1_, takw2_]
    return node_color, pos_list, pos_dict, takws


def make_colorcodes(model):
    """
    python -m wbia.algo.hots.bayes --exec-make_name_model --show
    python -m wbia.algo.hots.bayes --exec-cluster_query --show
    python -m wbia --tf demo_bayesnet --ev :nA=4,nS=2,Na=n0,rand_scores=True --show --verbose
    python -m wbia --tf demo_bayesnet --ev :nA=4,nS=3,Na=n0,rand_scores=True --show --verbose
    """
    import wbia.plottool as pt

    ttype_colors = {}
    ttype_scalars = {}

    # Hacked in custom colors

    if 'name' in model.ttype2_template:
        basis = model.ttype2_template['name'].basis
        card = len(basis)
        colors = pt.distinct_colors(max(card, 10))[:card]
        ttype_colors['name'] = colors
        ttype_scalars['name'] = np.linspace(0, 1, len(basis))

    if 'score' in model.ttype2_template:
        cmap_, mn, mx = 'plasma', 0.15, 1.0
        _cmap = pt.plt.get_cmap(cmap_)

        def cmap(x):
            return _cmap((x * mx) + mn)

        basis = model.ttype2_template['score'].basis
        scalars = np.linspace(0, 1, len(basis))
        # scalars = np.linspace(0, 1, 100)
        colors = pt.scores_to_color(
            scalars, cmap_=cmap_, reverse_cmap=False, cmap_range=(mn, mx)
        )
        colors = [pt.lighten_rgb(c, 0.4) for c in colors]
        ttype_scalars['score'] = scalars
        ttype_colors['score'] = colors

    return ttype_colors, ttype_scalars


def draw_bayesian_model(
    model, evidence={}, soft_evidence={}, fnum=None, pnum=None, **kwargs
):

    from pgmpy.models import BayesianModel

    if not isinstance(model, BayesianModel):
        model = model.to_bayesian_model()

    import wbia.plottool as pt
    import networkx as nx

    kwargs = kwargs.copy()
    factor_list = kwargs.pop('factor_list', [])

    ttype_colors, ttype_scalars = make_colorcodes(model)

    textprops = {
        'horizontalalignment': 'left',
        'family': 'monospace',
        'size': 8,
    }

    # build graph attrs
    tup = get_node_viz_attrs(
        model, evidence, soft_evidence, factor_list, ttype_colors, **kwargs
    )
    node_color, pos_list, pos_dict, takws = tup

    # draw graph
    has_inferred = evidence or 'factor_list' in kwargs

    if False:
        fig = pt.figure(fnum=fnum, pnum=pnum, doclf=True)  # NOQA
        ax = pt.gca()
        drawkw = dict(
            pos=pos_dict, ax=ax, with_labels=True, node_size=1100, node_color=node_color
        )
        nx.draw(model, **drawkw)
    else:
        # BE VERY CAREFUL
        if 1:
            graph = model.copy()
            graph.__class__ = nx.DiGraph
            graph.graph['groupattrs'] = ut.ddict(dict)
            # graph = model.
            if getattr(graph, 'ttype2_cpds', None) is not None:
                # Add invis edges and ttype groups
                for ttype in model.ttype2_cpds.keys():
                    ttype_cpds = model.ttype2_cpds[ttype]
                    # use defined ordering
                    ttype_nodes = ut.list_getattr(ttype_cpds, 'variable')
                    # ttype_nodes = sorted(ttype_nodes)
                    invis_edges = list(ut.itertwo(ttype_nodes))
                    graph.add_edges_from(invis_edges)
                    nx.set_edge_attributes(
                        graph,
                        name='style',
                        values={edge: 'invis' for edge in invis_edges},
                    )
                    nx.set_node_attributes(
                        graph,
                        name='groupid',
                        values={node: ttype for node in ttype_nodes},
                    )
                    graph.graph['groupattrs'][ttype]['rank'] = 'same'
                    graph.graph['groupattrs'][ttype]['cluster'] = False
        else:
            graph = model
        pt.show_nx(graph, layout_kw={'prog': 'dot'}, fnum=fnum, pnum=pnum, verbose=0)
        pt.zoom_factory()
        fig = pt.gcf()
        ax = pt.gca()
        pass
    hacks = [
        pt.draw_text_annotations(textprops=textprops, **takw) for takw in takws if takw
    ]

    xmin, ymin = np.array(pos_list).min(axis=0)
    xmax, ymax = np.array(pos_list).max(axis=0)
    if 'name' in model.ttype2_template:
        num_names = len(model.ttype2_template['name'].basis)
        num_annots = len(model.ttype2_cpds['name'])
        if num_annots > 4:
            ax.set_xlim((xmin - 40, xmax + 40))
            ax.set_ylim((ymin - 50, ymax + 50))
            fig.set_size_inches(30, 7)
        else:
            ax.set_xlim((xmin - 42, xmax + 42))
            ax.set_ylim((ymin - 50, ymax + 50))
            fig.set_size_inches(23, 7)
        title = 'num_names=%r, num_annots=%r' % (num_names, num_annots,)
    else:
        title = ''
    map_assign = kwargs.get('map_assign', None)

    def word_insert(text):
        return '' if len(text) == 0 else text + ' '

    top_assignments = kwargs.get('top_assignments', None)
    if top_assignments is not None:
        map_assign, map_prob = top_assignments[0]
        if map_assign is not None:
            title += '\n%sMAP: ' % (word_insert(kwargs.get('method', '')))
            title += map_assign + ' @' + '%.2f%%' % (100 * map_prob,)
    if kwargs.get('show_title', True):
        pt.set_figtitle(title, size=14)

    for hack in hacks:
        hack()

    if has_inferred:
        # Hack in colorbars
        # if ut.list_type(basis) is int:
        #     pt.colorbar(scalars, colors, lbl='score', ticklabels=np.array(basis) + 1)
        # else:
        #     pt.colorbar(scalars, colors, lbl='score', ticklabels=basis)
        keys = ['name', 'score']
        locs = ['left', 'right']
        for key, loc in zip(keys, locs):
            if key in ttype_colors:
                basis = model.ttype2_template[key].basis
                # scalars =
                colors = ttype_colors[key]
                scalars = ttype_scalars[key]
                pt.colorbar(scalars, colors, lbl=key, ticklabels=basis, ticklocation=loc)


def draw_markov_model(model, fnum=None, **kwargs):
    import wbia.plottool as pt

    fnum = pt.ensure_fnum(fnum)
    pt.figure(fnum=fnum, doclf=True)
    ax = pt.gca()
    from pgmpy.models import MarkovModel

    if isinstance(model, MarkovModel):
        markovmodel = model
    else:
        markovmodel = model.to_markov_model()
    # pos = nx.nx_agraph.pydot_layout(markovmodel)
    pos = nx.nx_agraph.pygraphviz_layout(markovmodel)
    # Referenecs:
    # https://groups.google.com/forum/#!topic/networkx-discuss/FwYk0ixLDuY

    # pos = nx.spring_layout(markovmodel)
    # pos = nx.circular_layout(markovmodel)
    # curved-arrow
    # markovmodel.edge_attr['curved-arrow'] = True
    # markovmodel.graph.setdefault('edge', {})['splines'] = 'curved'
    # markovmodel.graph.setdefault('graph', {})['splines'] = 'curved'
    # markovmodel.graph.setdefault('edge', {})['splines'] = 'curved'

    node_color = [pt.NEUTRAL] * len(pos)
    drawkw = dict(  # NOQA
        pos=pos, ax=ax, with_labels=True, node_color=node_color, node_size=1100
    )

    from matplotlib.patches import FancyArrowPatch, Circle
    import numpy as np

    def draw_network(G, pos, ax, sg=None):
        for n in G:
            c = Circle(pos[n], radius=10, alpha=0.5, color=pt.NEUTRAL_BLUE)
            ax.add_patch(c)
            G.nodes[n]['patch'] = c
            x, y = pos[n]
            pt.ax_absolute_text(x, y, n, ha='center', va='center')
        seen = {}
        for (u, v, d) in G.edges(data=True):
            n1 = G.nodes[u]['patch']
            n2 = G.nodes[v]['patch']
            rad = 0.1
            if (u, v) in seen:
                rad = seen.get((u, v))
                rad = (rad + np.sign(rad) * 0.1) * -1
            alpha = 0.5
            color = 'k'

            e = FancyArrowPatch(
                n1.center,
                n2.center,
                patchA=n1,
                patchB=n2,
                # arrowstyle='-|>',
                arrowstyle='-',
                connectionstyle='arc3,rad=%s' % rad,
                mutation_scale=10.0,
                lw=2,
                alpha=alpha,
                color=color,
            )
            seen[(u, v)] = rad
            ax.add_patch(e)
        return e

    # nx.draw(markovmodel, **drawkw)
    draw_network(markovmodel, pos, ax)
    ax.autoscale()
    pt.plt.axis('equal')
    pt.plt.axis('off')

    if kwargs.get('show_title', True):
        pt.set_figtitle('Markov Model')


def draw_junction_tree(model, fnum=None, **kwargs):
    import wbia.plottool as pt

    fnum = pt.ensure_fnum(fnum)
    pt.figure(fnum=fnum)
    ax = pt.gca()
    from pgmpy.models import JunctionTree

    if not isinstance(model, JunctionTree):
        netx_graph = model.to_junction_tree()
    else:
        netx_graph = model
    # prettify nodes

    def fixtupkeys(dict_):
        return {
            ', '.join(k) if isinstance(k, tuple) else k: fixtupkeys(v)
            for k, v in dict_.items()
        }

    n = fixtupkeys(netx_graph.nodes)
    e = fixtupkeys(netx_graph.edge)
    a = fixtupkeys(netx_graph.adj)
    netx_graph.nodes = n
    netx_graph.edge = e
    netx_graph.adj = a
    # netx_graph = model.to_markov_model()
    # pos = nx.nx_agraph.pygraphviz_layout(netx_graph)
    # pos = nx.nx_agraph.graphviz_layout(netx_graph)
    pos = nx.pydot_layout(netx_graph)
    node_color = [pt.NEUTRAL] * len(pos)
    drawkw = dict(pos=pos, ax=ax, with_labels=True, node_color=node_color, node_size=2000)
    nx.draw(netx_graph, **drawkw)
    if kwargs.get('show_title', True):
        pt.set_figtitle('Junction / Clique Tree / Cluster Graph')


def show_junction_tree(*args, **kwargs):
    return draw_junction_tree(*args, **kwargs)


def show_markov_model(*args, **kwargs):
    return draw_markov_model(*args, **kwargs)


def show_model(model, *args, **kwargs):
    modeltype = kwargs.pop('modeltype', 'bayes')
    if modeltype in ['bayesian', 'bayes']:
        return show_bayesian_model(model, *args, **kwargs)
    elif modeltype in ['markov']:
        return show_markov_model(model, *args, **kwargs)
    elif modeltype in ['junc', 'junction']:
        return show_junction_tree(model, *args, **kwargs)
    else:
        raise NotImplementedError('Unknown modeltype=%r' % (modeltype,))


def show_bayesian_model(model, evidence={}, soft_evidence={}, fnum=None, **kwargs):
    r"""
    References:
        http://stackoverflow.com/questions/22207802/networkx-node-level-or-layer

    Ignore:
        import nx2tikz
        print(nx2tikz.dumps_tikz(model, layout='layered', use_label=True))

    Ignore:
        sudo apt-get  install libgraphviz4 libgraphviz-dev -y
        sudo apt-get install libgraphviz-dev
        pip install pygraphviz
        sudo pip3 install pygraphviz \
            --install-option="--include-path=/usr/include/graphviz" \
            --install-option="--library-path=/usr/lib/graphviz/"
        python -c "import pygraphviz; print(pygraphviz.__file__)"
        python3 -c "import pygraphviz; print(pygraphviz.__file__)"

    CommandLine:
        python -m wbia.algo.hots.pgm_viz --exec-show_model --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.algo.hots.pgm_viz import *  # NOQA
        >>> model = '?'
        >>> evidence = {}
        >>> soft_evidence = {}
        >>> result = show_model(model, evidence, soft_evidence)
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> ut.show_if_requested()
    """
    import wbia.plottool as pt

    fnum = pt.ensure_fnum(fnum)
    top_assignments = kwargs.get('top_assignments', None)
    if evidence and top_assignments and 'factor_list' in kwargs:
        pnum1 = (3, 1, (slice(0, 2), 0))
        pnum2 = (3, 8, (2, slice(4, None)))
    else:
        pnum1 = (1, 1, 1)
    draw_bayesian_model(model, evidence, soft_evidence, fnum=fnum, pnum=pnum1, **kwargs)
    # Draw probability hist
    if top_assignments is not None:
        draw_map_histogram(top_assignments, fnum=fnum, pnum=pnum2)


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.algo.hots.pgm_viz
        python -m wbia.algo.hots.pgm_viz --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
