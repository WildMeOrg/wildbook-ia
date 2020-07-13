# -*- coding: utf-8 -*-
r"""
Helpers for graph plotting

References:
    http://www.graphviz.org/content/attrs
    http://www.graphviz.org/doc/info/attrs.html

Ignore:
    http://www.graphviz.org/pub/graphviz/stable/windows/graphviz-2.38.msi
    pip uninstall pydot
    pip uninstall pyparsing
    pip install -Iv https://pypi.python.org/packages/source/p/pyparsing/pyparsing-1.5.7.tar.gz#md5=9be0fcdcc595199c646ab317c1d9a709
    pip install pydot
    sudo apt-get  install libgraphviz4 libgraphviz-dev -y
    sudo apt-get install libgraphviz-dev
    pip install pygraphviz
    sudo pip3 install pygraphviz \
        --install-option="--include-path=/usr/include/graphviz" \
        --install-option="--library-path=/usr/lib/graphviz/"
    python -c "import pygraphviz; print(pygraphviz.__file__)"
    python3 -c "import pygraphviz; print(pygraphviz.__file__)"

"""
from __future__ import absolute_import, division, print_function
from six.moves import zip
import six

try:
    from wbia import dtool as dt
except ImportError:
    pass
import numpy as np
import utool as ut

# from wbia.plottool import colorfuncs
from six.moves import reduce

(print, rrr, profile) = ut.inject2(__name__)


LARGE_GRAPH = 100


def dump_nx_ondisk(graph, fpath):
    agraph = make_agraph(graph.copy())
    # agraph = nx.nx_agraph.to_agraph(graph)
    agraph.layout(prog='dot')
    agraph.draw(ut.truepath(fpath))


def ensure_nonhex_color(orig_color):
    # TODO: move to ensure color
    if isinstance(orig_color, six.string_types) and orig_color.startswith('#'):
        hex_color = orig_color
        import matplotlib.colors as colors

        color = colors.hex2color(hex_color[0:7])
        if len(hex_color) > 8:
            alpha_hex = hex_color[7:9]
            alpha_float = int(alpha_hex, 16) / 255.0
            color = color + (alpha_float,)
    else:
        color = orig_color
    return color


@profile
def show_nx(
    graph,
    with_labels=True,
    fnum=None,
    pnum=None,
    layout='agraph',
    ax=None,
    pos=None,
    img_dict=None,
    title=None,
    layoutkw=None,
    verbose=None,
    **kwargs
):
    r"""
    Args:
        graph (networkx.Graph):
        with_labels (bool): (default = True)
        fnum (int): figure number(default = None)
        pnum (tuple): plot number(default = None)
        layout (str): (default = 'agraph')
        ax (None): (default = None)
        pos (None): (default = None)
        img_dict (dict): (default = None)
        title (str):  (default = None)
        layoutkw (None): (default = None)
        verbose (bool):  verbosity flag(default = None)

    Kwargs:
        use_image, framewidth, modify_ax, as_directed, hacknoedge, hacknode,
        arrow_width, fontsize, fontweight, fontname, fontfamilty,
        fontproperties

    CommandLine:
        python -m wbia.plottool.nx_helpers show_nx --show
        python -m dtool --tf DependencyCache.make_graph --show
        python -m wbia.scripts.specialdraw double_depcache_graph --show --testmode
        python -m vtool.clustering2 unsupervised_multicut_labeling --show


    Example:
        >>> # ENABLE_DOCTEST
        >>> # xdoctest: +REQUIRES(module:pygraphviz)
        >>> from wbia.plottool.nx_helpers import *  # NOQA
        >>> import networkx as nx
        >>> graph = nx.DiGraph()
        >>> graph.add_nodes_from(['a', 'b', 'c', 'd'])
        >>> graph.add_edges_from({'a': 'b', 'b': 'c', 'b': 'd', 'c': 'd'}.items())
        >>> nx.set_node_attributes(graph, name='shape', values='rect')
        >>> nx.set_node_attributes(graph, name='image', values={'a': ut.grab_test_imgpath('carl.jpg')})
        >>> nx.set_node_attributes(graph, name='image', values={'d': ut.grab_test_imgpath('lena.png')})
        >>> #nx.set_node_attributes(graph, name='height', values=100)
        >>> with_labels = True
        >>> fnum = None
        >>> pnum = None
        >>> e = show_nx(graph, with_labels, fnum, pnum, layout='agraph')
        >>> import wbia.plottool as pt
        >>> pt.show_if_requested()
    """
    import wbia.plottool as pt
    import networkx as nx

    if ax is None:
        fnum = pt.ensure_fnum(fnum)
        pt.figure(fnum=fnum, pnum=pnum)
        ax = pt.gca()

    if img_dict is None:
        img_dict = nx.get_node_attributes(graph, 'image')

    if verbose is None:
        verbose = ut.VERBOSE

    use_image = kwargs.get('use_image', True)

    if verbose:
        print('Getting layout')
    layout_info = get_nx_layout(graph, layout, layoutkw=layoutkw, verbose=verbose)

    if verbose:
        print('Drawing graph')
    # zoom = kwargs.pop('zoom', .4)
    framewidth = kwargs.pop('framewidth', 1.0)
    patch_dict = draw_network2(graph, layout_info, ax, verbose=verbose, **kwargs)
    layout_info.update(patch_dict)
    if kwargs.get('modify_ax', True):
        ax.grid(False)
        pt.plt.axis('equal')
        ax.patch.set_facecolor('white')
        ax.autoscale()
        ax.autoscale_view(True, True, True)
    # axes.facecolor

    node_size = layout_info['node'].get('size')
    node_pos = layout_info['node'].get('pos')
    if node_size is not None:
        size_arr = np.array(ut.take(node_size, graph.nodes()))
        half_size_arr = size_arr / 2.0
        pos_arr = np.array(ut.take(node_pos, graph.nodes()))
        # autoscale does not seem to work
        # ul_pos = pos_arr - half_size_arr
        # br_pos = pos_arr + half_size_arr
        # hack because edges are cut off.
        # need to take into account extent of edges as well
        ul_pos = pos_arr - half_size_arr * 1.5
        br_pos = pos_arr + half_size_arr * 1.5
        xmin, ymin = ul_pos.min(axis=0)
        xmax, ymax = br_pos.max(axis=0)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    # pt.plt.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])

    if use_image and img_dict is not None and len(img_dict) > 0:
        if verbose:
            print('Drawing images')
        node_list = sorted(img_dict.keys())
        pos_list = ut.dict_take(node_pos, node_list)
        img_list = ut.dict_take(img_dict, node_list)
        size_list = ut.dict_take(node_size, node_list)
        # color_list = ut.dict_take(nx.get_node_attributes(graph, 'color'), node_list, None)
        color_list = ut.dict_take(
            nx.get_node_attributes(graph, 'framecolor'), node_list, None
        )
        framewidth_list = ut.dict_take(
            nx.get_node_attributes(graph, 'framewidth'), node_list, framewidth
        )
        pt.netx_draw_images_at_positions(
            img_list, pos_list, size_list, color_list, framewidth_list=framewidth_list
        )
        # Hack in older interface
        imgdat = {}
        imgdat['node_list'] = node_list
        layout_info['imgdat'] = imgdat
    else:
        if verbose:
            print('Not drawing images')

    if title is not None:
        pt.set_title(title)
    return layout_info


def netx_draw_images_at_positions(
    img_list, pos_list, size_list, color_list, framewidth_list
):
    """
    Overlays images on a networkx graph

    References:
        https://gist.github.com/shobhit/3236373
        http://matplotlib.org/examples/pylab_examples/demo_annotation_box.html
        http://stackoverflow.com/questions/11487797/mpl-overlay-small-image
        http://matplotlib.org/api/text_api.html
        http://matplotlib.org/api/offsetbox_api.html
    """
    import vtool as vt
    import wbia.plottool as pt

    # Ensure all images have been read
    img_list_ = [
        vt.convert_colorspace(vt.imread(img), 'RGB')
        if isinstance(img, six.string_types)
        else img
        for img in img_list
    ]
    size_list_ = [
        vt.get_size(img) if size is None else size
        for size, img in zip(size_list, img_list)
    ]

    for pos, img, size in zip(pos_list, img_list_, size_list_):
        bbox = vt.bbox_from_center_wh(pos, size)
        extent = vt.extent_from_bbox(bbox)
        pt.plt.imshow(img, extent=extent)


def parse_html_graphviz_attrs():
    # Parse the documentation table
    import bs4
    import requests

    r = requests.get(r'http://www.graphviz.org/doc/info/attrs.html')
    data = r.text
    soup = bs4.BeautifulSoup(data, 'html5lib')

    for table in soup.findAll('table'):
        if len(list(table.descendants)) > 2000:
            break

    columns = [th.text.strip() for th in table.find_all('th')]

    data = []
    for tr in table.find_all('tr'):
        row = [td.text.strip() for td in tr.find_all('td')]
        if row:
            data.append(row)

    import pandas as pd

    pd.options.display.max_rows = 20
    pd.options.display.max_columns = 40
    pd.options.display.width = 160
    pd.options.display.float_format = lambda x: '%.4f' % (x,)

    full_df = pd.DataFrame(data, columns=columns)
    # Find valid progs that can be used
    all_progs = []
    for n in full_df['Notes'].tolist():
        line = n.replace(' only', '').replace('not ', '')
        found = [_.strip() for _ in line.split(',')]
        all_progs.extend(found)
    all_progs = set(all_progs) - {''}

    # Find which progs are supported by which rows
    supported_progs = []
    for n in full_df['Notes'].tolist():
        line = n.replace(' only', '').replace('not ', '')
        if n.endswith('only'):
            only = {_.strip() for _ in line.split(',')}
            supported_progs.append(only)
        elif n.startswith('not'):
            noneof = {_.strip() for _ in line.split(',')}
            supported_progs.append(all_progs - noneof)
        else:
            supported_progs.append(all_progs)

    # Find subset that supports dot or neato
    dot_or_neato = [len({'dot', 'neato'}.intersection(p)) > 0 for p in supported_progs]
    df = full_df[dot_or_neato]
    df = full_df

    neato_ = [len({'neato'}.intersection(p)) > 0 for p in supported_progs]
    df = full_df

    # types are:
    # edges, nodes, the root graph, subgraphs and cluster subgraphs
    typed_keys = {}
    for t in {'E', 'N', 'G', 'S', 'C'}:
        flags = [t in x for x in df['Used By']]
        typed_keys[t] = df[flags]['Name'].tolist()
    print(ut.format_single_paragraph_sentences(', '.join(typed_keys['G'])))

    df = full_df[neato_]
    neato_keys = {}
    for t in {'E', 'N', 'G', 'S', 'C'}:
        flags = [t in x for x in df['Used By']]
        neato_keys[t] = df[flags]['Name'].tolist()
    print(ut.format_single_paragraph_sentences(', '.join(neato_keys['G'])))


class GRAPHVIZ_KEYS(object):  # NOQA
    N = {
        'URL',
        'area',
        'color',
        'colorscheme',
        'comment',
        'distortion',
        'fillcolor',
        'fixedsize',
        'fontcolor',
        'fontname',
        'fontsize',
        'gradientangle',
        'group',
        'height',
        'href',
        'id',
        'image',
        'imagepos',
        'imagescale',
        'label',
        'labelloc',
        'layer',
        'margin',
        'nojustify',
        'ordering',
        'orientation',
        'penwidth',
        'peripheries',
        'pin',
        'pos',
        'rects',
        'regular',
        'root',
        'samplepoints',
        'shape',
        'shapefile',
        'showboxes',
        'sides',
        'skew',
        'sortv',
        'style',
        'target',
        'tooltip',
        'vertices',
        'width',
        'xlabel',
        'xlp',
        'z',
    }

    E = {
        'URL',
        'arrowhead',
        'arrowsize',
        'arrowtail',
        'color',
        'colorscheme',
        'comment',
        'constraint',
        'decorate',
        'dir',
        'edgeURL',
        'edgehref',
        'edgetarget',
        'edgetooltip',
        'fillcolor',
        'fontcolor',
        'fontname',
        'fontsize',
        'headURL',
        'head_lp',
        'headclip',
        'headhref',
        'headlabel',
        'headport',
        'headtarget',
        'headtooltip',
        'href',
        'id',
        'label',
        'labelURL',
        'labelangle',
        'labeldistance',
        'labelfloat',
        'labelfontcolor',
        'labelfontname',
        'labelfontsize',
        'labelhref',
        'labeltarget',
        'labeltooltip',
        'layer',
        'len',
        'lhead',
        'lp',
        'ltail',
        'minlen',
        'nojustify',
        'penwidth',
        'pos',
        'samehead',
        'sametail',
        'showboxes',
        'style',
        'tailURL',
        'tail_lp',
        'tailclip',
        'tailhref',
        'taillabel',
        'tailport',
        'tailtarget',
        'tailtooltip',
        'target',
        'tooltip',
        'weight',
        'xlabel',
        'xlp',
    }

    G = {
        'Damping',
        'K',
        'URL',
        '_background',
        'bb',
        'bgcolor',
        'center',
        'charset',
        'clusterrank',
        'colorscheme',
        'comment',
        'compound',
        'concentrate',
        'defaultdist',
        'dim',
        'dimen',
        'diredgeconstraints',
        'dpi',
        'epsilon',
        'esep',
        'fontcolor',
        'fontname',
        'fontnames',
        'fontpath',
        'fontsize',
        'forcelabels',
        'gradientangle',
        'href',
        'id',
        'imagepath',
        'inputscale',
        'label',
        'label_scheme',
        'labeljust',
        'labelloc',
        'landscape',
        'layerlistsep',
        'layers',
        'layerselect',
        'layersep',
        'layout',
        'levels',
        'levelsgap',
        'lheight',
        'lp',
        'lwidth',
        'margin',
        'maxiter',
        'mclimit',
        'mindist',
        'mode',
        'model',
        'mosek',
        'newrank',
        'nodesep',
        'nojustify',
        'normalize',
        'notranslate',
        'nslimit\nnslimit1',
        'ordering',
        'orientation',
        'outputorder',
        'overlap',
        'overlap_scaling',
        'overlap_shrink',
        'pack',
        'packmode',
        'pad',
        'page',
        'pagedir',
        'quadtree',
        'quantum',
        'rankdir',
        'ranksep',
        'ratio',
        'remincross',
        'repulsiveforce',
        'resolution',
        'root',
        'rotate',
        'rotation',
        'scale',
        'searchsize',
        'sep',
        'showboxes',
        'size',
        'smoothing',
        'sortv',
        'splines',
        'start',
        'style',
        'stylesheet',
        'target',
        'truecolor',
        'viewport',
        'voro_margin',
        'xdotversion',
    }


try:

    class GraphVizLayoutConfig(dt.Config):
        r"""
        Ignore:
            Node Props:
                colorscheme    CEGN           string                NaN
                  fontcolor    CEGN            color                NaN
                   fontname    CEGN           string                NaN
                   fontsize    CEGN           double                NaN
                      label    CEGN        lblString                NaN
                  nojustify    CEGN             bool                NaN
                      style    CEGN            style                NaN
                      color     CEN   colorcolorList                NaN
                  fillcolor     CEN   colorcolorList                NaN
                      layer     CEN       layerRange                NaN
                   penwidth     CEN           double                NaN
               radientangle     CGN              int                NaN
                   labelloc     CGN           string                NaN
                     margin     CGN      doublepoint                NaN
                      sortv     CGN              int                NaN
                peripheries      CN              int                NaN
                  showboxes     EGN              int           dot only
                    comment     EGN           string                NaN
                        pos      EN  pointsplineType                NaN
                     xlabel      EN        lblString                NaN
                   ordering      GN           string           dot only
                      group       N           string           dot only
                        pin       N             bool   fdp | neato only
                 distortion       N           double                NaN
                  fixedsize       N       boolstring                NaN
                     height       N           double                NaN
                      image       N           string                NaN
                 imagescale       N       boolstring                NaN
                orientation       N           double                NaN
                    regular       N             bool                NaN
               samplepoints       N              int                NaN
                      shape       N            shape                NaN
                  shapefile       N           string                NaN
                      sides       N              int                NaN
                       skew       N           double                NaN
                      width       N           double                NaN
                          z       N           double                NaN
        """

        # TODO: make a gridsearchable config for layouts
        @staticmethod
        def get_param_info_list():
            param_info_list = [
                # GENERAL
                ut.ParamInfo(
                    'splines',
                    'spline',
                    valid_values=[
                        'none',
                        'line',
                        'polyline',
                        'curved',
                        'ortho',
                        'spline',
                    ],
                ),
                ut.ParamInfo('pack', True),
                ut.ParamInfo('packmode', 'cluster'),
                # ut.ParamInfo('nodesep', ?),
                # NOT DOT
                ut.ParamInfo(
                    'overlap', 'prism', valid_values=['true', 'false', 'prism', 'ipsep']
                ),
                ut.ParamInfo('sep', 1 / 8),
                ut.ParamInfo('esep', 1 / 8),  # stricly  less than sep
                # NEATO ONLY
                ut.ParamInfo('mode', 'major', valid_values=['heir', 'KK', 'ipsep']),
                # kwargs['diredgeconstraints'] = 'heir'
                # kwargs['inputscale'] = kwargs.get('inputscale', 72)
                # kwargs['Damping'] = kwargs.get('Damping', .1)
                # DOT ONLY
                ut.ParamInfo('rankdir', 'LR', valid_values=['LR', 'RL', 'TB', 'BT']),
                ut.ParamInfo('ranksep', 2.5),
                ut.ParamInfo('nodesep', 2.0),
                ut.ParamInfo('clusterrank', 'local', valid_values=['local', 'global'])
                # OUTPUT ONLY
                # kwargs['dpi'] = kwargs.get('dpi', 1.0)
            ]
            return param_info_list


except Exception:
    pass


def get_explicit_graph(graph):
    """
    Args:
        graph (nx.Graph)
    """
    import copy

    def get_nx_base(graph):
        import networkx as nx

        if isinstance(graph, nx.MultiDiGraph):
            base_class = nx.MultiDiGraph
        elif isinstance(graph, nx.MultiGraph):
            base_class = nx.MultiGraph
        elif isinstance(graph, nx.DiGraph):
            base_class = nx.DiGraph
        elif isinstance(graph, nx.Graph):
            base_class = nx.Graph
        else:
            assert False
        return base_class

    base_class = get_nx_base(graph)
    explicit_graph = base_class()
    explicit_graph.graph = copy.deepcopy(graph.graph)

    explicit_nodes = graph.nodes(data=True)
    explicit_edges = [
        (n1, n2, data)
        for (n1, n2, data) in graph.edges(data=True)
        if data.get('implicit', False) is not True
    ]
    explicit_graph.add_nodes_from(explicit_nodes)
    explicit_graph.add_edges_from(explicit_edges)
    return explicit_graph


def get_nx_layout(graph, layout, layoutkw=None, verbose=None):
    import networkx as nx

    if layoutkw is None:
        layoutkw = {}
    layout_info = {}

    if layout == 'custom':
        edge_keys = list(
            reduce(
                set.union,
                [set(edge[-1].keys()) for edge in graph.edges(data=True)],
                set([]),
            )
        )
        node_keys = list(
            reduce(
                set.union,
                [set(node[-1].keys()) for node in graph.nodes(data=True)],
                set([]),
            )
        )
        graph_keys = list(graph.graph.keys())
        layout_info = {
            'graph': {k: graph.graph.get(k) for k in graph_keys},
            'node': {k: nx.get_node_attributes(graph, k) for k in node_keys},
            'edge': {k: nx.get_edge_attributes(graph, k) for k in edge_keys},
        }
        # Post checks
        node_info = layout_info['node']
        if 'size' not in node_info:
            if 'width' in node_info and 'height' in node_info:
                node_info['size'] = {
                    node: (node_info['width'][node], node_info['height'][node])
                    for node in graph.nodes()
                }
                # node_info['size'] = list(zip(node_info['width'],
                #                             node_info['height']))

    elif layout == 'agraph':
        # PREFERED LAYOUT WITH MOST CONTROL
        _, layout_info = nx_agraph_layout(graph, verbose=verbose, **layoutkw)
    else:
        raise ValueError('Undefined layout = %r' % (layout,))
    return layout_info


def apply_graph_layout_attrs(graph, layout_info):
    import networkx as nx

    def noneish(v):
        isNone = v is None
        isNoneStr = isinstance(v, six.string_types) and v.lower() == 'none'
        return isNone or isNoneStr

    for key, vals in layout_info['node'].items():
        vals = {n: v for n, v in vals.items() if not noneish(n)}
        nx.set_node_attributes(graph, name=key, values=vals)
    for key, vals in layout_info['edge'].items():
        vals = {e: v for e, v in vals.items() if not noneish(e)}
        nx.set_edge_attributes(graph, name=key, values=vals)
    graph_attrs = {k: v for k, v in layout_info['graph'].items() if not noneish(k)}
    graph.graph.update(graph_attrs)


def patch_pygraphviz():
    """
    Hacks around a python3 problem in 1.3.1 of pygraphviz
    """
    import pygraphviz

    if pygraphviz.__version__ != '1.3.1':
        return
    if hasattr(pygraphviz.agraph.AGraph, '_run_prog_patch'):
        return

    def _run_prog(self, prog='nop', args=''):
        """Apply graphviz program to graph and return the result as a string.

        >>> A = AGraph()
        >>> s = A._run_prog() # doctest: +SKIP
        >>> s = A._run_prog(prog='acyclic') # doctest: +SKIP

        Use keyword args to add additional arguments to graphviz programs.
        """
        from pygraphviz.agraph import shlex, subprocess, PipeReader, warnings

        runprog = r'"%s"' % self._get_prog(prog)
        cmd = ' '.join([runprog, args])
        dotargs = shlex.split(cmd)
        p = subprocess.Popen(
            dotargs,
            shell=False,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=False,
        )
        (child_stdin, child_stdout, child_stderr) = (p.stdin, p.stdout, p.stderr)
        # Use threading to avoid blocking
        data = []
        errors = []
        threads = [PipeReader(data, child_stdout), PipeReader(errors, child_stderr)]
        for t in threads:
            t.start()

        self.write(child_stdin)
        child_stdin.close()

        for t in threads:
            t.join()

        if not data:
            raise IOError(b''.join(errors))

        if len(errors) > 0:
            warnings.warn(str(b''.join(errors)), RuntimeWarning)

        return b''.join(data)

    # Patch error in pygraphviz
    pygraphviz.agraph.AGraph._run_prog_patch = _run_prog
    pygraphviz.agraph.AGraph._run_prog_orig = pygraphviz.agraph.AGraph._run_prog
    pygraphviz.agraph.AGraph._run_prog = _run_prog


def make_agraph(graph_):
    # FIXME; make this not an inplace operation
    import networkx as nx
    import pygraphviz

    patch_pygraphviz()
    # Convert to agraph format

    num_nodes = len(graph_)
    is_large = num_nodes > LARGE_GRAPH

    if is_large:
        print('Making agraph for large graph %d nodes. ' 'May take time' % (num_nodes))

    ut.nx_ensure_agraph_color(graph_)
    # Reduce size to be in inches not pixels
    # FIXME: make robust to param settings
    # Hack to make the w/h of the node take thae max instead of
    # dot which takes the minimum
    shaped_nodes = [n for n, d in graph_.nodes(data=True) if 'width' in d]
    node_dict = ut.nx_node_dict(graph_)
    node_attrs = ut.dict_take(node_dict, shaped_nodes)

    width_px = np.array(ut.take_column(node_attrs, 'width'))
    height_px = np.array(ut.take_column(node_attrs, 'height'))
    scale = np.array(ut.dict_take_column(node_attrs, 'scale', default=1.0))

    inputscale = 72.0
    width_in = width_px / inputscale * scale
    height_in = height_px / inputscale * scale
    width_in_dict = dict(zip(shaped_nodes, width_in))
    height_in_dict = dict(zip(shaped_nodes, height_in))

    nx.set_node_attributes(graph_, name='width', values=width_in_dict)
    nx.set_node_attributes(graph_, name='height', values=height_in_dict)
    ut.nx_delete_node_attr(graph_, name='scale')

    # Check for any nodes with groupids
    node_to_groupid = nx.get_node_attributes(graph_, 'groupid')
    if node_to_groupid:
        groupid_to_nodes = ut.group_items(*zip(*node_to_groupid.items()))
    else:
        groupid_to_nodes = {}
    # Initialize agraph format
    # import utool
    # utool.embed()
    ut.nx_delete_None_edge_attr(graph_)
    agraph = nx.nx_agraph.to_agraph(graph_)
    # Add subgraphs labels
    # TODO: subgraph attrs
    group_attrs = graph_.graph.get('groupattrs', {})
    for groupid, nodes in groupid_to_nodes.items():
        # subgraph_attrs = {}
        subgraph_attrs = group_attrs.get(groupid, {}).copy()
        cluster_flag = True
        # FIXME: make this more natural to specify
        if 'cluster' in subgraph_attrs:
            cluster_flag = subgraph_attrs['cluster']
            del subgraph_attrs['cluster']
        # subgraph_attrs = dict(rankdir='LR')
        # subgraph_attrs = dict(rankdir='LR')
        # subgraph_attrs['rank'] = 'min'
        # subgraph_attrs['rank'] = 'source'
        name = groupid
        if cluster_flag:
            # graphviz treast subgraphs labeld with cluster differently
            name = 'cluster_' + groupid
        else:
            name = groupid
        agraph.add_subgraph(nodes, name, **subgraph_attrs)

    import re

    for node in graph_.nodes():
        anode = pygraphviz.Node(agraph, node)
        # TODO: Generally fix node positions
        ptstr_ = anode.attr['pos']
        if ptstr_ is not None and len(ptstr_) > 0 and not ptstr_.endswith('!'):
            ptstr = ptstr_.strip('[]').strip(' ').strip('()')
            ptstr_list = [x.rstrip(',') for x in re.split(r'\s+', ptstr)]
            pt_list = list(map(float, ptstr_list))
            pt_arr = np.array(pt_list) / inputscale
            new_ptstr_list = list(map(str, pt_arr))
            new_ptstr_ = ','.join(new_ptstr_list)
            if anode.attr['pin'] is True:
                anode.attr['pin'] = 'true'
            if anode.attr['pin'] == 'true':
                new_ptstr = new_ptstr_ + '!'
            else:
                new_ptstr = new_ptstr_
            anode.attr['pos'] = new_ptstr

    if graph_.graph.get('ignore_labels', False):
        for node in graph_.nodes():
            anode = pygraphviz.Node(agraph, node)
            if 'label' in anode.attr:
                try:
                    del anode.attr['label']
                except KeyError:
                    pass

    return agraph


def _groupby_prelayout(graph_, layoutkw, groupby):
    """
    sets `pin` attr of `graph_` inplace in order to nodes according to
    specified layout.
    """
    import networkx as nx

    has_pins = any(
        [v.lower() == 'true' for v in nx.get_node_attributes(graph_, 'pin').values()]
    )

    has_pins &= all('pos' in d for n, d in graph_.nodes(data=True))
    if not has_pins:
        # Layout groups separately
        node_to_group = nx.get_node_attributes(graph_, groupby)
        group_to_nodes = ut.invert_dict(node_to_group, unique_vals=False)
        subgraph_list = []

        def subgraph_grid(subgraphs, hpad=None, vpad=None):
            n_cols = int(np.ceil(np.sqrt(len(subgraphs))))
            columns = [
                ut.stack_graphs(chunk, vert=False, pad=hpad)
                for chunk in ut.ichunks(subgraphs, n_cols)
            ]
            new_graph = ut.stack_graphs(columns, vert=True, pad=vpad)
            return new_graph

        group_grid = graph_.graph.get('group_grid', None)

        for group, nodes in group_to_nodes.items():
            if group_grid:
                subnode_list = [graph_.subgraph([node]) for node in nodes]
                for sub in subnode_list:
                    sub.graph.update(graph_.graph)
                    nx_agraph_layout(sub, inplace=True, groupby=None, **layoutkw)
                subgraph = subgraph_grid(subnode_list)
                # subgraph = graph_.subgraph(nodes)
            else:
                subgraph = graph_.subgraph(nodes)
            subgraph.graph.update(graph_.graph)
            nx_agraph_layout(subgraph, inplace=True, groupby=None, **layoutkw)
            subgraph_list.append(subgraph)

        hpad = graph_.graph.get('hpad', None)
        vpad = graph_.graph.get('vpad', None)
        graph_ = subgraph_grid(subgraph_list, hpad, vpad)

        # graph_ = ut.stack_graphs(subgraph_list)
        nx.set_node_attributes(graph_, name='pin', values='true')
        return True, graph_
    else:
        return False, graph_
        # print('WARNING: GROUPING WOULD CLOBBER PINS. NOT GROUPING')


def nx_agraph_layout(
    orig_graph, inplace=False, verbose=None, return_agraph=False, groupby=None, **layoutkw
):
    r"""
    Uses graphviz and custom code to determine position attributes of nodes and
    edges.

    Args:
        groupby (str): if not None then nodes will be grouped by this
            attributes and groups will be layed out separately and then stacked
            together in a grid

    Ignore:
        orig_graph = graph
        graph = layout_graph

    References:
        http://www.graphviz.org/content/attrs
        http://www.graphviz.org/doc/info/attrs.html

    CommandLine:
        python -m wbia.plottool.nx_helpers nx_agraph_layout --show

    Doctest:
        >>> # xdoctest: +REQUIRES(module:pygraphviz)
        >>> from wbia.plottool.nx_helpers import *  # NOQA
        >>> import wbia.plottool as pt
        >>> import networkx as nx
        >>> import utool as ut
        >>> n, s = 9, 4
        >>> offsets = list(range(0, (1 + n) * s, s))
        >>> node_groups = [ut.lmap(str, range(*o)) for o in ut.itertwo(offsets)]
        >>> edge_groups = [ut.combinations(nodes, 2) for nodes in node_groups]
        >>> graph = nx.Graph()
        >>> [graph.add_nodes_from(nodes) for nodes in node_groups]
        >>> [graph.add_edges_from(edges) for edges in edge_groups]
        >>> for count, nodes in enumerate(node_groups):
        ...     nx.set_node_attributes(graph, name='id', values=ut.dzip(nodes, [count]))
        >>> layoutkw = dict(prog='neato')
        >>> graph1, info1 = nx_agraph_layout(graph.copy(), inplace=True, groupby='id', **layoutkw)
        >>> graph2, info2 = nx_agraph_layout(graph.copy(), inplace=True, **layoutkw)
        >>> graph3, _ = nx_agraph_layout(graph1.copy(), inplace=True, **layoutkw)
        >>> nx.set_node_attributes(graph1, name='pin', values='true')
        >>> graph4, _ = nx_agraph_layout(graph1.copy(), inplace=True, **layoutkw)
        >>> if pt.show_was_requested():
        >>>     pt.show_nx(graph1, layout='custom', pnum=(2, 2, 1), fnum=1)
        >>>     pt.show_nx(graph2, layout='custom', pnum=(2, 2, 2), fnum=1)
        >>>     pt.show_nx(graph3, layout='custom', pnum=(2, 2, 3), fnum=1)
        >>>     pt.show_nx(graph4, layout='custom', pnum=(2, 2, 4), fnum=1)
        >>>     pt.show_if_requested()
        >>> g1pos = nx.get_node_attributes(graph1, 'pos')['1']
        >>> g4pos = nx.get_node_attributes(graph4, 'pos')['1']
        >>> g2pos = nx.get_node_attributes(graph2, 'pos')['1']
        >>> g3pos = nx.get_node_attributes(graph3, 'pos')['1']
        >>> print('g1pos = {!r}'.format(g1pos))
        >>> print('g4pos = {!r}'.format(g4pos))
        >>> print('g2pos = {!r}'.format(g2pos))
        >>> print('g3pos = {!r}'.format(g3pos))
        >>> assert np.all(g1pos == g4pos), 'points between 1 and 4 were pinned so they should be equal'
        >>> #assert np.all(g2pos != g3pos), 'points between 2 and 3 were not pinned, so they should be different'

        assert np.all(nx.get_node_attributes(graph1, 'pos')['1'] == nx.get_node_attributes(graph4, 'pos')['1'])
        assert np.all(nx.get_node_attributes(graph2, 'pos')['1'] == nx.get_node_attributes(graph3, 'pos')['1'])
    """
    # import networkx as nx
    import pygraphviz

    # graph_ = get_explicit_graph(orig_graph).copy()
    graph_ = get_explicit_graph(orig_graph)

    # only_explicit = True
    # if only_explicit:
    num_nodes = len(graph_)
    is_large = num_nodes > LARGE_GRAPH

    # layoutkw = layoutkw.copy()
    draw_implicit = layoutkw.pop('draw_implicit', True)

    pinned_groups = False

    if groupby is not None:
        pinned_groups, graph_ = _groupby_prelayout(
            graph_, layoutkw=layoutkw, groupby=groupby
        )

    prog = layoutkw.pop('prog', 'dot')

    if prog != 'dot':
        layoutkw['overlap'] = layoutkw.get('overlap', 'false')
    layoutkw['splines'] = layoutkw.get('splines', 'spline')
    if prog == 'neato':
        layoutkw['notranslate'] = 'true'  # for neato postprocessing

    if True:
        argparts = ['-G%s=%s' % (key, str(val)) for key, val in layoutkw.items()]
        splines = layoutkw['splines']
    else:
        # layoutkw is allowed to overwrite graph.graph['graph']
        args_kw = graph_.graph.get('graph', {}).copy()
        for key, val in layoutkw.items():
            if key in GRAPHVIZ_KEYS.G and val is not None:
                if key not in args_kw:
                    args_kw[key] = val

        # del args_kw['sep']
        # del args_kw['nodesep']
        # del args_kw['overlap']
        # del args_kw['notranslate']
        argparts = ['-G{}={}'.format(key, val) for key, val in args_kw.items()]
        splines = args_kw['splines']

    args = ' '.join(argparts)

    if verbose is None:
        verbose = ut.VERBOSE
    if verbose or is_large:
        print('[nx_agraph_layout] args = %r' % (args,))
    # Convert to agraph format

    agraph = make_agraph(graph_)

    # Run layout
    # print('prog = %r' % (prog,))

    if verbose > 3:
        print('BEFORE LAYOUT\n' + str(agraph))

    if is_large:
        print(
            'Preforming agraph layout on graph with %d nodes.'
            'May take time' % (num_nodes)
        )

    # import warnings
    # warnings.filterwarnings("error")
    # import warnings
    # flag = False

    # for node in graph_.nodes():
    #    anode = pygraphviz.Node(agraph, node)
    #    ptstr_ = anode.attr['pos']
    #    print('ptstr_ = %r' % (ptstr_,))

    # FIXME; This spits out warnings on weird color input
    # import warnings
    # with warnings.catch_warnings(record=True):
    #     # warnings.filterwarnings('error')
    #     warnings.filterwarnings('ignore')
    try:
        agraph.layout(prog=prog, args=args)
    except Exception as ex:
        ut.printex(ex, tb=True)
        # import utool
        # utool.embed()
        raise
    # except RuntimeWarning as ex:
    #    ut.printex(ex, iswarning=True)
    #    flag = True
    # if flag:
    #    import utool
    #    utool.embed()

    if is_large:
        print('Finished agraph layout.')

    if 0:
        test_fpath = ut.truepath('~/test_graphviz_draw.png')
        agraph.draw(test_fpath)
        ut.startfile(test_fpath)
    if verbose > 3:
        print('AFTER LAYOUT\n' + str(agraph))

    # TODO: just replace with a single dict of attributes
    node_layout_attrs = ut.ddict(dict)
    edge_layout_attrs = ut.ddict(dict)

    # for node in agraph.nodes():
    for node in graph_.nodes():
        anode = pygraphviz.Node(agraph, node)
        node_attrs = parse_anode_layout_attrs(anode)
        for key, val in node_attrs.items():
            node_layout_attrs[key][node] = val

    edges = list(ut.nx_edges(graph_, keys=True))

    for edge in edges:
        aedge = pygraphviz.Edge(agraph, *edge)
        edge_attrs = parse_aedge_layout_attrs(aedge)
        for key, val in edge_attrs.items():
            edge_layout_attrs[key][edge] = val

    if draw_implicit:
        # ADD IN IMPLICIT EDGES
        layout_edges = set(ut.nx_edges(graph_, keys=True))
        orig_edges = set(ut.nx_edges(orig_graph, keys=True))
        implicit_edges = list(orig_edges - layout_edges)
        # all_edges = list(set.union(orig_edges, layout_edges))
        needs_implicit = len(implicit_edges) > 0
        if needs_implicit:
            # Pin down positions
            for node in agraph.nodes():
                anode = pygraphviz.Node(agraph, node)
                anode.attr['pin'] = 'true'
                anode.attr['pos'] += '!'

            # Add new edges to route
            for iedge in implicit_edges:
                data = orig_graph.get_edge_data(*iedge)
                agraph.add_edge(*iedge, **data)

            if ut.VERBOSE or verbose:
                print('BEFORE IMPLICIT LAYOUT\n' + str(agraph))
            # Route the implicit edges (must use neato)

            control_node = pygraphviz.Node(agraph, node)
            # print('control_node = %r' % (control_node,))
            node1_attr1 = parse_anode_layout_attrs(control_node)
            # print('node1_attr1 = %r' % (node1_attr1,))

            implicit_kw = layoutkw.copy()
            implicit_kw['overlap'] = 'true'
            # del implicit_kw['overlap']  # can cause node positions to change
            argparts = ['-G%s=%s' % (key, str(val)) for key, val in implicit_kw.items()]
            args = ' '.join(argparts)

            if is_large:
                print(
                    '[nx_agraph_layout] About to draw implicit layout ' 'for large graph.'
                )

            agraph.layout(prog='neato', args='-n ' + args)

            if is_large:
                print('[nx_agraph_layout] done with implicit layout for ' 'large graph.')

            if False:
                agraph.draw(ut.truepath('~/implicit_test_graphviz_draw.png'))
            if ut.VERBOSE or verbose:
                print('AFTER IMPLICIT LAYOUT\n' + str(agraph))

            control_node = pygraphviz.Node(agraph, node)
            # print('control_node = %r' % (control_node,))
            node1_attr2 = parse_anode_layout_attrs(control_node)
            # print('node1_attr2 = %r' % (node1_attr2,))

            # graph positions shifted
            # This is not the right place to divide by 72
            translation = node1_attr1['pos'] - node1_attr2['pos']
            # print('translation = %r' % (translation,))
            # translation = np.array([0, 0])
            # print('translation = %r' % (translation,))

            # for iedge in all_edges:
            for iedge in implicit_edges:
                aedge = pygraphviz.Edge(agraph, *iedge)
                iedge_attrs = parse_aedge_layout_attrs(aedge, translation)
                for key, val in iedge_attrs.items():
                    edge_layout_attrs[key][iedge] = val

    if pinned_groups:
        # Remove temporary pins put in place by groups
        ut.nx_delete_node_attr(graph_, 'pin')

    graph_layout_attrs = dict(splines=splines)

    layout_info = {
        'graph': graph_layout_attrs,
        'edge': dict(edge_layout_attrs),
        'node': dict(node_layout_attrs),
    }

    if inplace:
        apply_graph_layout_attrs(orig_graph, layout_info)
        graph = orig_graph
    else:
        # FIXME: there is really no point to returning graph unless we actually
        # modify its attributes
        graph = graph_
    if return_agraph:
        return graph, layout_info, agraph
    else:
        return graph, layout_info


def parse_point(ptstr):
    try:
        xx, yy = ptstr.strip('!').split(',')
        xy = np.array((float(xx), float(yy)))
    except Exception:
        xy = None
    return xy


def parse_anode_layout_attrs(anode):
    node_attrs = {}
    # try:
    xx, yy = anode.attr['pos'].strip('!').split(',')
    xy = np.array((float(xx), float(yy)))
    # except Exception:
    #    xy = np.array((0.0, 0.0))
    adpi = 72.0
    width = float(anode.attr['width']) * adpi
    height = float(anode.attr['height']) * adpi
    node_attrs['width'] = width
    node_attrs['height'] = height
    node_attrs['size'] = (width, height)
    node_attrs['pos'] = xy
    return node_attrs


def parse_aedge_layout_attrs(aedge, translation=None):
    """
    parse grpahviz splineType
    """
    if translation is None:
        translation = np.array([0, 0])
    edge_attrs = {}
    apos = aedge.attr['pos']
    # print('apos = %r' % (apos,))
    end_pt = None
    start_pt = None
    # if '-' in apos:
    #    import utool
    #    utool.embed()

    def safeadd(x, y):
        if x is None or y is None:
            return None
        return x + y

    strpos_list = apos.split(' ')
    strtup_list = [ea.split(',') for ea in strpos_list]

    ctrl_ptstrs = [ea for ea in strtup_list if ea[0] not in 'es']
    end_ptstrs = [ea[1:] for ea in strtup_list[0:2] if ea[0] == 'e']
    start_ptstrs = [ea[1:] for ea in strtup_list[0:2] if ea[0] == 's']
    assert len(end_ptstrs) <= 1
    assert len(start_ptstrs) <= 1
    if len(end_ptstrs) == 1:
        end_pt = np.array([float(f) for f in end_ptstrs[0]])
    if len(start_ptstrs) == 1:
        start_pt = np.array([float(f) for f in start_ptstrs[0]])
    ctrl_pts = np.array([tuple([float(f) for f in ea]) for ea in ctrl_ptstrs])
    adata = aedge.attr
    ctrl_pts = ctrl_pts
    edge_attrs['pos'] = apos
    edge_attrs['ctrl_pts'] = safeadd(ctrl_pts, translation)
    edge_attrs['start_pt'] = safeadd(start_pt, translation)
    edge_attrs['end_pt'] = safeadd(end_pt, translation)
    edge_attrs['lp'] = safeadd(parse_point(adata.get('lp', None)), translation)
    edge_attrs['label'] = adata.get('label', None)
    edge_attrs['headlabel'] = adata.get('headlabel', None)
    edge_attrs['taillabel'] = adata.get('taillabel', None)
    edge_attrs['head_lp'] = safeadd(parse_point(adata.get('head_lp', None)), translation)
    edge_attrs['tail_lp'] = safeadd(parse_point(adata.get('tail_lp', None)), translation)
    return edge_attrs


def format_anode_pos(xy, pin=True):
    xx, yy = xy
    return '%f,%f%s' % (xx, yy, '!' * pin)


def _get_node_size(graph, node, node_size):
    if node_size is not None and node in node_size:
        return node_size[node]
    node_dict = ut.nx_node_dict(graph)
    nattrs = node_dict[node]
    scale = nattrs.get('scale', 1.0)
    if 'width' in nattrs and 'height' in nattrs:
        width = nattrs['width'] * scale
        height = nattrs['height'] * scale
    elif 'radius' in nattrs:
        width = height = nattrs['radius'] * scale
    else:
        if 'image' in nattrs:
            img_fpath = nattrs['image']
            import vtool as vt

            width, height = vt.image.open_image_size(img_fpath)
        else:
            height = width = 1100 / 50 * scale
    return width, height


@profile
def draw_network2(
    graph,
    layout_info,
    ax,
    as_directed=None,
    hacknoedge=False,
    hacknode=False,
    verbose=None,
    **kwargs
):
    """
    Kwargs:
        use_image, arrow_width, fontsize, fontweight, fontname, fontfamilty,
        fontproperties

    fancy way to draw networkx graphs without directly using networkx

    # python -m wbia.annotmatch_funcs review_tagged_joins --dpath ~/latex/crall-candidacy-2015/ --save figures4/mergecase.png --figsize=15,15 --clipwhite --diskshow
    # python -m dtool --tf DependencyCache.make_graph --show
    """
    import wbia.plottool as pt
    import matplotlib as mpl

    figsize = ut.get_argval('--figsize', type_=list, default=None)

    patch_dict = {
        'patch_frame_dict': {},
        'node_patch_dict': {},
        'edge_patch_dict': {},
        'arrow_patch_list': {},
    }

    text_pseudo_objects = []

    # TODO: get font properties from nodes as well
    font_prop = pt.parse_fontkw(**kwargs)
    # print('font_prop = %r' % (font_prop,))
    # print('font_prop.get_name() = %r' % (font_prop.get_name() ,))

    node_pos = layout_info['node'].get('pos', {})
    node_size = layout_info['node'].get('size', {})
    splines = layout_info['graph'].get('splines', 'line')
    # edge_startpoints = layout_info['edge']['start_pt']

    if as_directed is None:
        as_directed = graph.is_directed()

    # Draw nodes
    large_graph = len(graph) > LARGE_GRAPH
    # for edge, pts in ut.ProgIter(edge_pos.items(), length=len(edge_pos), enabled=large_graph, lbl='drawing edges'):

    for node, nattrs in ut.ProgIter(
        graph.nodes(data=True),
        length=len(graph),
        lbl='drawing nodes',
        enabled=large_graph,
    ):
        # shape = nattrs.get('shape', 'circle')
        if nattrs is None:
            nattrs = {}
        label = nattrs.get('label', None)
        alpha = nattrs.get('alpha', 1.0)
        node_color = nattrs.get('color', pt.NEUTRAL_BLUE)
        if node_color is None:
            node_color = pt.NEUTRAL_BLUE
        xy = node_pos[node]
        using_image = kwargs.get('use_image', True) and 'image' in nattrs
        if using_image:
            if hacknode:
                alpha_ = 0.7
            else:
                alpha_ = 0.0
        else:
            alpha_ = alpha

        node_color = ensure_nonhex_color(node_color)
        # intcolor = int(node_color.replace('#', '0x'), 16)
        node_color = node_color[0:3]
        patch_kw = dict(alpha=alpha_, color=node_color)
        node_shape = nattrs.get('shape', 'ellipse')
        if node_shape == 'circle':
            # divide by 2 seems to work for agraph
            radius = max(_get_node_size(graph, node, node_size)) / 2.0
            patch = mpl.patches.Circle(xy, radius=radius, **patch_kw)
        elif node_shape == 'ellipse':
            # divide by 2 seems to work for agraph
            width, height = np.array(_get_node_size(graph, node, node_size))
            patch = mpl.patches.Ellipse(xy, width, height, **patch_kw)
        elif node_shape in ['none', 'box', 'rect', 'rectangle', 'rhombus']:
            width, height = _get_node_size(graph, node, node_size)
            angle = 45 if node_shape == 'rhombus' else 0
            # Convert xy to center position
            xy_bl = (xy[0] - width // 2, xy[1] - height // 2)

            # rounded = angle == 0
            node_dict = ut.nx_node_dict(graph)
            rounded = 'rounded' in node_dict.get(node, {}).get('style', '')
            isdiag = 'diagonals' in node_dict.get(node, {}).get('style', '')

            from matplotlib import patches

            if rounded:
                rpad = 20
                xy_bl = np.array(xy_bl) + rpad
                width -= rpad * 2
                height -= rpad * 2
                boxstyle = patches.BoxStyle.Round(pad=rpad)
                patch = patches.FancyBboxPatch(
                    xy_bl, width, height, boxstyle=boxstyle, **patch_kw
                )
            else:
                bbox = list(xy_bl) + [width, height]
                if isdiag:
                    import vtool as vt

                    center_xy = vt.bbox_center(bbox)
                    _xy = np.array(center_xy)
                    newverts_ = [
                        _xy + [0, -height / 2],
                        _xy + [-width / 2, 0],
                        _xy + [0, height / 2],
                        _xy + [width / 2, 0],
                    ]
                    patch = patches.Polygon(newverts_, **patch_kw)
                else:
                    patch = patches.Rectangle(
                        xy_bl, width, height, angle=angle, **patch_kw
                    )

            patch.center = xy
        # if style == 'rounded'
        # elif node_shape in ['roundbox']:
        elif node_shape == 'stack':
            width, height = _get_node_size(graph, node, node_size)
            xy_bl = (xy[0] - width // 2, xy[1] - height // 2)
            depth = nattrs.get('depth', 4)
            stackkw = patch_kw.copy()
            stackkw['linewidths'] = 0.2
            stackkw['edgecolor'] = 'k'
            # xshift = -width * (.1 / (depth ** (1 / 3))) / 3
            # yshift = height * (.1 / (depth ** (1 / 3))) / 2
            # xshift = -width * (.05) / 6
            # yshift = height * (.05) / 2
            xshift = -200 * (0.05) / 6
            yshift = 200 * (0.05) / 2
            stackkw['shift'] = np.array([xshift, yshift])
            patch = pt.cartoon_stacked_rects(xy_bl, width, height, num=depth, **stackkw)
            patch.xy = xy
        else:
            raise NotImplementedError('Unknown node_shape=%r' % (node_shape,))

        show_center = 0
        if show_center:
            pt.plot(xy[0], xy[1], 'xr')

        zorder = nattrs.get('zorder', None)
        if True:
            # Add a frame around the node
            framewidth = nattrs.get('framewidth', 0)
            framealpha = nattrs.get('framealpha', 1.0)
            framealign = nattrs.get('framealign', 'center')
            if framewidth > 0:
                framecolor = nattrs.get('framecolor', node_color)
                framecolor = ensure_nonhex_color(framecolor)

                # print('framecolor = %r' % (framecolor,))
                if framecolor is None:
                    framecolor = pt.BLACK
                    framealpha = 0.0
                if framewidth is True:
                    if figsize is not None:
                        # HACK
                        graphsize = max(figsize)
                        framewidth = graphsize / 4
                    else:
                        framewidth = 3.0
                lw = framewidth
                frame = pt.make_bbox(
                    bbox,
                    bbox_color=framecolor,
                    ax=ax,
                    lw=lw,
                    align=framealign,
                    alpha=framealpha,
                )
                if zorder is not None:
                    frame.set_zorder(zorder)
                # frame.set_zorder()
                patch_dict['patch_frame_dict'][node] = frame
        # import utool
        # utool.embed()
        picker = nattrs.get('picker', True)
        patch.set_picker(picker)
        if zorder is not None:
            patch.set_zorder(zorder)
        pt.set_plotdat(patch, 'node_data', nattrs)
        pt.set_plotdat(patch, 'node', node)

        x, y = xy
        text = str(node)
        if label is not None:
            # text += ': ' + str(label)
            text = label
        if kwargs.get('node_labels', hacknode or not using_image):
            text_args = (
                (x, y, text),
                dict(ax=ax, ha='center', va='center', fontproperties=font_prop),
            )
            text_pseudo_objects.append((text_args, zorder))
        patch_dict['node_patch_dict'][node] = patch

    def get_default_edge_data(graph, edge):
        data = graph.get_edge_data(*edge)
        if data is None:
            if len(edge) == 3 and edge[2] is not None:
                data = graph.get_edge_data(edge[0], edge[1], int(edge[2]))
            else:
                data = graph.get_edge_data(edge[0], edge[1])
        if data is None:
            data = {}
        return data

    ###
    # Draw Edges
    # NEW WAY OF DRAWING EDGEES
    edge_pos = layout_info['edge'].get('ctrl_pts', None)
    n_invis_edge = 0
    if edge_pos is not None:
        for edge, pts in ut.ProgIter(
            edge_pos.items(),
            length=len(edge_pos),
            enabled=large_graph,
            lbl='drawing edges',
        ):
            data = get_default_edge_data(graph, edge)

            if data.get('style', None) == 'invis':
                n_invis_edge += 1
                continue

            alpha = data.get('alpha', None)

            defaultcolor = pt.BLACK[0:3]
            if alpha is None:
                if data.get('implicit', False):
                    alpha = 0.5
                    defaultcolor = pt.GREEN[0:3]
                else:
                    alpha = 1.0
            color = data.get('color', defaultcolor)
            if color is None:
                color = defaultcolor
            color = ensure_nonhex_color(color)
            color = color[0:3]

            # layout_info['edge']['ctrl_pts'][edge]
            # layout_info['edge']['start_pt'][edge]

            offset = 0 if graph.is_directed() else 0
            # color = data.get('color', color)[0:3]
            start_point = pts[offset]
            other_points = pts[offset + 1 :].tolist()  # [0:3]
            verts = [start_point] + other_points

            MOVETO = mpl.path.Path.MOVETO
            LINETO = mpl.path.Path.LINETO
            # STOP = mpl.path.Path.STOP

            if splines in ['line', 'polyline', 'ortho']:
                CODE = LINETO
            elif splines == 'curved':
                # CODE = mpl.path.Path.CURVE3
                # CODE = mpl.path.Path.CURVE3
                CODE = mpl.path.Path.CURVE4
            elif splines == 'spline':
                CODE = mpl.path.Path.CURVE4
            else:
                raise AssertionError('splines = %r' % (splines,))

            astart_code = MOVETO
            astart_code = MOVETO

            verts = [start_point] + other_points
            codes = [astart_code] + [CODE] * len(other_points)

            end_pt = layout_info['edge'].get('end_pt', {}).get(edge, None)

            # HACK THE ENDPOINTS TO TOUCH THE BOUNDING BOXES
            if end_pt is not None:
                verts += [end_pt]
                codes += [LINETO]

            path = mpl.path.Path(verts, codes)

            figsize = ut.get_argval('--figsize', type_=list, default=None)
            if figsize is not None:
                # HACK
                graphsize = max(figsize)
                lw = graphsize / 8
                width = graphsize / 15
                width = ut.get_argval('--arrow-width', default=width)
                lw = ut.get_argval('--line-width', default=lw)
                # print('width = %r' % (width,))
            else:
                width = 0.5
                lw = 1.0
                try:
                    import vtool as vt

                    # Compute arrow width using estimated graph size
                    if node_size is not None and node_pos is not None:
                        xys = np.array(ut.take(node_pos, node_pos.keys())).T
                        whs = np.array(ut.take(node_size, node_pos.keys())).T
                        bboxes = vt.bbox_from_xywh(xys, whs, [0.5, 0.5])
                        extents = vt.extent_from_bbox(bboxes)
                        tl_pts = np.array([extents[0], extents[2]]).T
                        br_pts = np.array([extents[1], extents[3]]).T
                        pts = np.vstack([tl_pts, br_pts])
                        extent = vt.get_pointset_extents(pts)
                        graph_w, graph_h = vt.bbox_from_extent(extent)[2:4]
                        graph_dim = np.sqrt(graph_w ** 2 + graph_h ** 2)

                        # width = graph_dim * .0005
                        width = graph_dim * 0.005
                except Exception:
                    pass
            arrow_width = kwargs.get('arrow_width', width)

            if not as_directed and end_pt is not None:
                pass

            lw = data.get('linewidth', data.get('lw', lw))
            linestyle = 'solid'
            linestyle = data.get('linestyle', linestyle)
            hatch = data.get('hatch', '')

            # keep track of the linewidth as path effects (like stroke) are
            # added
            full_lw = lw

            # effects = data.get('stroke', None)
            from matplotlib import patheffects

            path_effects = []

            sketch_params = data.get('sketch')
            if sketch_params is not None:
                if sketch_params is True:
                    # scale, length, randomness
                    # sketch_params = (10.0, 128.0, 16.0)
                    sketch_params = dict(scale=10.0, length=128.0, randomness=16.0,)

            stroke_info = data.get('stroke', None)
            if stroke_info not in [None, False]:
                if stroke_info is True:
                    strokekw = {}
                elif isinstance(stroke_info, dict):
                    strokekw = stroke_info.copy()
                else:
                    # linewidth=3, foreground='r'
                    assert False
                if strokekw is not None:
                    # Hack to increase lw
                    full_lw = lw + strokekw.get('linewidth', 3)
                    strokekw['linewidth'] = full_lw
                    path_effects += [patheffects.withStroke(**strokekw)]

            # http://matplotlib.org/1.2.1/examples/api/clippath_demo.html
            if data.get('shadow', None) is not None:
                shadowkw = data['shadow']
                if shadowkw is not False:
                    if shadowkw is True:
                        shadowkw = {}
                    linewidth = shadowkw.pop('linewidth', full_lw)
                    scale = shadowkw.pop('scale', 1.0)
                    shadow_color = shadowkw.pop('color', 'k')
                    shadow_color = shadowkw.pop('shadow_color', shadow_color)
                    offset = ut.ensure_iterable(shadowkw.pop('offset', (2, -2)))
                    if len(offset) == 1:
                        offset = offset * 2
                    shadowkw_ = dict(
                        offset=offset,
                        shadow_color=shadow_color,
                        alpha=0.3,
                        rho=0.3,
                        linewidth=linewidth * scale,
                    )
                    shadowkw_.update(shadowkw)
                    path_effects += [patheffects.SimpleLineShadow(**shadowkw_)]

            # for vert, code in path.iter_segments():
            #    print('code = %r' % (code,))
            #    print('vert = %r' % (vert,))
            #    if code == MOVETO:
            #        pass

            # for verts, code in path.cleaned().iter_segments():
            #    print('code = %r' % (code,))
            #    print('verts = %r' % (verts,))
            #    pass
            path_effects += [patheffects.Normal()]

            picker = data.get('picker', 5)
            zorder = data.get('zorder', 5)
            patch = mpl.patches.PathPatch(
                path,
                facecolor='none',
                lw=lw,
                path_effects=path_effects,
                edgecolor=color,
                picker=picker,
                # facecolor=color,
                linestyle=linestyle,
                alpha=alpha,
                joinstyle='bevel',
                hatch=hatch,
                # sketch_params=sketch_params,
                zorder=zorder,
            )
            if sketch_params is not None:
                patch.set_sketch_params(**sketch_params)

            pt.set_plotdat(patch, 'edge_data', data)
            pt.set_plotdat(patch, 'edge', edge)

            if as_directed:
                if end_pt is not None:
                    dxy = np.array(end_pt) - other_points[-1]
                    dxy = (dxy / np.sqrt(np.sum(dxy ** 2))) * 0.1
                    dx, dy = dxy
                    rx, ry = end_pt[0], end_pt[1]
                    patch1 = mpl.patches.FancyArrow(
                        rx,
                        ry,
                        dx,
                        dy,
                        width=arrow_width,
                        length_includes_head=True,
                        color=color,
                        head_starts_at_zero=False,
                    )
                else:
                    dxy = np.array(other_points[-1]) - other_points[-2]
                    dxy = (dxy / np.sqrt(np.sum(dxy ** 2))) * 0.1
                    dx, dy = dxy
                    rx, ry = other_points[-1][0], other_points[-1][1]
                    patch1 = mpl.patches.FancyArrow(
                        rx,
                        ry,
                        dx,
                        dy,
                        width=arrow_width,
                        length_includes_head=True,
                        color=color,
                        head_starts_at_zero=True,
                    )
                # ax.add_patch(patch1)
                patch_dict['arrow_patch_list'][edge] = patch1

            taillabel = layout_info['edge'].get('taillabel', {}).get(edge, None)
            headlabel = layout_info['edge'].get('headlabel', {}).get(edge, None)
            label = layout_info['edge'].get('label', {}).get(edge, None)
            # hack
            if isinstance(taillabel, six.string_types) and taillabel == 'None':
                taillabel = None
            if isinstance(headlabel, six.string_types) and headlabel == 'None':
                headlabel = None
            if isinstance(label, six.string_types) and label == 'None':
                label = None
            # ha = 'left'
            # ha = 'right'
            ha = 'center'
            va = 'center'
            labelcolor = color  # TODO allow for different colors

            labelcolor = data.get('labelcolor', color)
            labelcolor = ensure_nonhex_color(labelcolor)
            labelcolor = labelcolor[0:3]

            if taillabel:
                taillabel_pos = layout_info['edge']['tail_lp'][edge]
                ax.annotate(
                    taillabel,
                    xy=taillabel_pos,
                    xycoords='data',
                    color=labelcolor,
                    va=va,
                    ha=ha,
                    fontproperties=font_prop,
                )
            if headlabel:
                headlabel_pos = layout_info['edge']['head_lp'][edge]
                ax.annotate(
                    headlabel,
                    xy=headlabel_pos,
                    xycoords='data',
                    color=labelcolor,
                    va=va,
                    ha=ha,
                    fontproperties=font_prop,
                )
            if label:
                label_pos = layout_info['edge']['lp'][edge]
                ax.annotate(
                    label,
                    xy=label_pos,
                    xycoords='data',
                    color=labelcolor,
                    va=va,
                    ha=ha,
                    fontproperties=font_prop,
                )
            patch_dict['edge_patch_dict'][edge] = patch

            # ax.add_patch(patch)

    if verbose:
        print('Adding %r node patches ' % (len(patch_dict['node_patch_dict'],)))
        print('Adding %r edge patches ' % (len(patch_dict['edge_patch_dict'],)))
        print('n_invis_edge = %r' % (n_invis_edge,))

    for frame in patch_dict['patch_frame_dict'].values():
        ax.add_patch(frame)

    for patch1 in patch_dict['arrow_patch_list'].values():
        ax.add_patch(patch1)

    use_collections = False
    if use_collections:
        edge_coll = mpl.collections.PatchCollection(
            patch_dict['edge_patch_dict'].values()
        )
        node_coll = mpl.collections.PatchCollection(
            patch_dict['node_patch_dict'].values()
        )
        # coll.set_facecolor(fcolor)
        # coll.set_alpha(alpha)
        # coll.set_linewidth(lw)
        # coll.set_edgecolor(color)
        # coll.set_transform(ax.transData)
        ax.add_collection(node_coll)
        ax.add_collection(edge_coll)
    else:
        for patch in patch_dict['node_patch_dict'].values():
            if isinstance(patch, mpl.collections.PatchCollection):
                ax.add_collection(patch)
            else:
                ax.add_patch(patch)
        if not hacknoedge:
            for patch in patch_dict['edge_patch_dict'].values():
                ax.add_patch(patch)

    for text_args, zorder in text_pseudo_objects:
        textobj = pt.ax_absolute_text(*text_args[0], **text_args[1])
        if zorder is not None:
            textobj.set_zorder(zorder)
    return patch_dict


# def arrowed_spines(ax=None, arrow_length=20, labels=('', ''), arrowprops=None):
#     """
#     TODO arrow splines not spines
#     References:
#         https://gist.github.com/joferkington/3845684
#     """
#     xlabel, ylabel = labels
#     import wbia.plottool as pt
#     if ax is None:
#         ax = pt.plt.gca()
#     if arrowprops is None:
#         arrowprops = dict(arrowstyle='<|-', facecolor='black')

#     for i, spine in enumerate(['left', 'bottom']):
#         # Set up the annotation parameters
#         t = ax.spines[spine].get_transform()
#         xy, xycoords = [1, 0], ('axes fraction', t)
#         xytext, textcoords = [arrow_length, 0], ('offset points', t)
#         ha, va = 'left', 'bottom'

#         # If axis is reversed, draw the arrow the other way
#         top, bottom = ax.spines[spine].axis.get_view_interval()
#         if top < bottom:
#             xy[0] = 0
#             xytext[0] *= -1
#             ha, va = 'right', 'top'

#         if spine is 'bottom':
#             xarrow = ax.annotate(xlabel, xy, xycoords=xycoords, xytext=xytext,
#                                  textcoords=textcoords, ha=ha, va='center',
#                                  arrowprops=arrowprops)
#         else:
#             yarrow = ax.annotate(ylabel, xy[::-1], xycoords=xycoords[::-1],
#                                  xytext=xytext[::-1], textcoords=textcoords[::-1],
#                                  ha='center', va=va, arrowprops=arrowprops)
#     return xarrow, yarrow


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.plottool.nx_helpers
        python -m wbia.plottool.nx_helpers --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
