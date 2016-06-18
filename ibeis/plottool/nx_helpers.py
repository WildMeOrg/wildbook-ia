# -*- coding: utf-8 -*-
"""
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
import numpy as np
import matplotlib as mpl
import utool as ut
import vtool as vt
import six
import dtool
(print, rrr, profile) = ut.inject2(__name__, '[nxhelpers]')


def dump_nx_ondisk(graph, fpath):
    agraph = make_agraph(graph)
    agraph.layout(prog='dot')
    agraph.draw(ut.truepath(fpath))


def fix_color(color):
    if isinstance(color, six.string_types) and color.startswith('#'):
        import matplotlib.colors as colors
        color = colors.hex2color(color[0:7])
    return color


def show_nx(graph, with_labels=True, fnum=None, pnum=None, layout='agraph',
            ax=None, pos=None, img_dict=None, title=None, layoutkw=None,
            verbose=None, **kwargs):
    r"""
    Args:
        graph (networkx.Graph):
        with_labels (bool): (default = True)
        node_size (int): (default = 1100)
        fnum (int):  figure number(default = None)
        pnum (tuple):  plot number(default = None)

    CommandLine:
        python -m plottool.nx_helpers --exec-show_nx --show
        python -m dtool --tf DependencyCache.make_graph --show
        python -m ibeis.scripts.specialdraw double_depcache_graph --show --testmode
        python -m vtool.clustering2 unsupervised_multicut_labeling --show


    Example:
        >>> # DISABLE_DOCTEST
        >>> from plottool.nx_helpers import *  # NOQA
        >>> import networkx as nx
        >>> graph = nx.DiGraph()
        >>> graph.add_nodes_from(['a', 'b', 'c', 'd'])
        >>> graph.add_edges_from({'a': 'b', 'b': 'c', 'b': 'd', 'c': 'd'}.items())
        >>> nx.set_node_attributes(graph, 'shape', 'rect')
        >>> nx.set_node_attributes(graph, 'image', {'a': ut.grab_test_imgpath('carl.jpg')})
        >>> nx.set_node_attributes(graph, 'image', {'d': ut.grab_test_imgpath('lena.png')})
        >>> #nx.set_node_attributes(graph, 'height', 100)
        >>> with_labels = True
        >>> fnum = None
        >>> pnum = None
        >>> e = show_nx(graph, with_labels, fnum, pnum, layout='agraph')
        >>> ut.show_if_requested()
    """
    import plottool as pt
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
    layout_info = get_nx_layout(graph, layout, layoutkw=layoutkw,
                                verbose=verbose)

    if verbose:
        print('Drawing graph')
    # zoom = kwargs.pop('zoom', .4)
    framewidth = kwargs.pop('framewidth', 1.0)
    patch_dict = draw_network2(graph, layout_info, ax, verbose=verbose, **kwargs)
    layout_info.update(patch_dict)
    ax.grid(False)
    pt.plt.axis('equal')
    ax.axesPatch.set_facecolor('white')
    #axes.facecolor
    #import utool
    #utool.embed()

    ax.autoscale()
    ax.autoscale_view(True, True, True)

    node_size = layout_info['node']['size']
    node_pos = layout_info['node']['pos']
    if node_size is not None:
        half_size_arr = np.array(ut.take(node_size, graph.nodes())) / 2.
        pos_arr = np.array(ut.take(node_pos, graph.nodes()))
        # autoscale does not seem to work
        #ul_pos = pos_arr - half_size_arr
        #br_pos = pos_arr + half_size_arr
        # hack because edges are cut off.
        # need to take into account extent of edges as well
        ul_pos = pos_arr - half_size_arr * 1.5
        br_pos = pos_arr + half_size_arr * 1.5
        xmin, ymin = ul_pos.min(axis=0)
        xmax, ymax = br_pos.max(axis=0)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    #pt.plt.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])

    if use_image and img_dict is not None and len(img_dict) > 0:
        if verbose:
            print('Drawing images')
        node_list = sorted(img_dict.keys())
        pos_list = ut.dict_take(node_pos, node_list)
        img_list = ut.dict_take(img_dict, node_list)
        size_list = ut.dict_take(node_size, node_list)
        #color_list = ut.dict_take(nx.get_node_attributes(graph, 'color'), node_list, None)
        color_list = ut.dict_take(nx.get_node_attributes(graph, 'framecolor'), node_list, None)
        framewidth_list = ut.dict_take(nx.get_node_attributes(graph, 'framewidth'),
                                       node_list, framewidth)
        # TODO; frames without images
        imgdat = pt.netx_draw_images_at_positions(img_list, pos_list,
                                                  size_list, color_list,
                                                  framewidth_list=framewidth_list)
        imgdat['node_list'] = node_list
        layout_info['imgdat'] = imgdat
    else:
        if verbose:
            print('Not drawing images')

    if title is not None:
        pt.set_title(title)
    return layout_info


def netx_draw_images_at_positions(img_list, pos_list, size_list, color_list,
                                  framewidth_list):
    """
    Overlays images on a networkx graph

    References:
        https://gist.github.com/shobhit/3236373
        http://matplotlib.org/examples/pylab_examples/demo_annotation_box.html
        http://stackoverflow.com/questions/11487797/mpl-overlay-small-image
        http://matplotlib.org/api/text_api.html
        http://matplotlib.org/api/offsetbox_api.html

    TODO: look into DraggableAnnotation
    """
    #print('[viz_graph] drawing %d images' % len(img_list))
    # Thumb stackartist
    import plottool as pt
    #ax  = pt.gca()

    # Ensure all images have been read
    img_list_ = [vt.convert_colorspace(vt.imread(img), 'RGB')
                 if isinstance(img, six.string_types) else img
                 for img in img_list]
    size_list_ = [vt.get_size(img) if size is None else size
                  for size, img in zip(size_list, img_list)]

    as_offset_image = False

    if as_offset_image:
        offset_img_list = []
        artist_list = []
        # THIS DOES NOT DO WHAT I WANT
        # Scales the image with data coords
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        bboxkw = dict(
            xycoords='data',
            boxcoords='offset points',
            #boxcoords='data',
            pad=0.25,
            # frameon=False, bboxprops=dict(fc="cyan"),
            # arrowprops=dict(arrowstyle="->"))
        )
        for pos, img, framewidth in zip(pos_list, img_list_, framewidth_list):
            offset_img = OffsetImage(img, zoom=.4)
            bboxkw['frameon'] = framewidth > 0
            artist = AnnotationBbox(offset_img, pos, xybox=(-0., 0.), **bboxkw)
            offset_img_list.append(offset_img)
            artist_list.append(artist)
    else:
        # THIS DOES EXACTLY WHAT I WANT
        # Ties the image to data coords
        #for pos, img, size, color, framewidth in zip(pos_list, img_list_,
        #                                             size_list_, color_list,
        #                                             framewidth_list):
        for pos, img, size in zip(pos_list, img_list_, size_list_):
            bbox = vt.bbox_from_center_wh(pos, size)
            extent = vt.extent_from_bbox(bbox)
            pt.plt.imshow(img, extent=extent)

    imgdat = {
        #'offset_img_list': offset_img_list,
        #'artist_list': artist_list,
    }
    return imgdat


class GraphVizLayoutConfig(dtool.Config):
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
            ut.ParamInfo('splines', 'spline', valid_values=[
                'none', 'line', 'polyline', 'curved', 'ortho', 'spline']),
            ut.ParamInfo('pack', True),
            ut.ParamInfo('packmode', 'cluster'),
            #ut.ParamInfo('nodesep', ?),
            # NOT DOT
            ut.ParamInfo('overlap', 'prism', valid_values=[
                'true', 'false', 'prism', 'ipsep']),
            ut.ParamInfo('sep', 1 / 8),
            ut.ParamInfo('esep', 1 / 8),  # stricly  less than sep
            # NEATO ONLY
            ut.ParamInfo('mode', 'major', valid_values=['heir', 'KK', 'ipsep']),
            #kwargs['diredgeconstraints'] = 'heir'
            #kwargs['inputscale'] = kwargs.get('inputscale', 72)
            #kwargs['Damping'] = kwargs.get('Damping', .1)
            # DOT ONLY
            ut.ParamInfo('rankdir', 'LR', valid_values=['LR', 'RL', 'TB', 'BT']),
            ut.ParamInfo('ranksep', 2.5),
            ut.ParamInfo('nodesep', 2.0),
            ut.ParamInfo('clusterrank', 'local', valid_values=['local', 'global'])
            # OUTPUT ONLY
            #kwargs['dpi'] = kwargs.get('dpi', 1.0)
        ]
        return param_info_list


def get_explicit_graph(graph):
    """
    Args:
        graph (nx.Graph)
    """

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
    # base_class = graph.__class__
    # import utool
    # utool.embed()
    # explicit_graph = graph.copy()
    # explicit_graph.clear()
    explicit_graph = base_class()
    import copy
    explicit_graph.graph = copy.deepcopy(graph.graph)

    explicit_nodes = graph.nodes(data=True)
    explicit_edges = [
        (n1, n2, data) for (n1, n2, data) in graph.edges(data=True)
        if data.get('implicit', False) is not True
    ]
    explicit_graph.add_nodes_from(explicit_nodes)
    explicit_graph.add_edges_from(explicit_edges)
    return explicit_graph


def get_nx_layout(graph, layout, layoutkw=None, verbose=None):
    import networkx as nx
    only_explicit = True
    if only_explicit:
        explicit_graph = get_explicit_graph(graph)
        layout_graph = explicit_graph
    else:
        layout_graph = graph

    if layoutkw is None:
        layoutkw = {}
    layout_info = {}

    if layout == 'custom':

        edge_keys = list(reduce(set.union, [set(edge[-1].keys()) for edge in
                                            graph.edges(data=True)]))
        node_keys = list(reduce(set.union, [set(node[-1].keys()) for node in
                                            graph.nodes(data=True)]))
        graph_keys = list(graph.graph.keys())
        #graph_keys = [
        #    'splines',
        #]
        layout_info = {
            'graph': {k: graph.graph.get(k) for k in graph_keys},
            'node': {k: nx.get_node_attributes(graph, k) for k in node_keys},
            'edge': {k: nx.get_edge_attributes(graph, k) for k in edge_keys},
        }
    elif layout == 'agraph':
        # PREFERED LAYOUT WITH MOST CONTROL
        _, layout_info = nx_agraph_layout(layout_graph, orig_graph=graph, verbose=verbose,
                                          **layoutkw)
    else:
        raise ValueError('Undefined layout = %r' % (layout,))
    return layout_info


def apply_graph_layout_attrs(graph, layout_info):
    import networkx as nx
    for key, vals in layout_info['node'].items():
        nx.set_node_attributes(graph, key, vals)

    for key, vals in layout_info['edge'].items():
        nx.set_edge_attributes(graph, key, vals)
    graph.graph.update(layout_info['graph'])


def make_agraph(graph):
    # FIXME; use this in nx_agraph_layout instead to comparementalize more
    import networkx as nx
    import pygraphviz
    # Convert to agraph format
    graph_ = graph.copy()

    ut.nx_ensure_agraph_color(graph_)
    # Reduce size to be in inches not pixels
    # FIXME: make robust to param settings
    # Hack to make the w/h of the node take thae max instead of
    # dot which takes the minimum
    shaped_nodes = [n for n, d in graph_.nodes(data=True) if 'width' in d]
    node_attrs = ut.dict_take(graph_.node, shaped_nodes)
    width_px = np.array(ut.take_column(node_attrs, 'width'))
    height_px = np.array(ut.take_column(node_attrs, 'height'))
    scale = np.array(ut.dict_take_column(node_attrs, 'scale', default=1.0))

    width_in = width_px / 72.0 * scale
    height_in = height_px / 72.0 * scale
    width_in_dict = dict(zip(shaped_nodes, width_in))
    height_in_dict = dict(zip(shaped_nodes, height_in))
    nx.set_node_attributes(graph_, 'width', width_in_dict)
    nx.set_node_attributes(graph_, 'height', height_in_dict)
    ut.nx_delete_node_attr(graph_, 'scale')

    # Check for any nodes with groupids
    node_to_groupid = nx.get_node_attributes(graph_, 'groupid')
    if node_to_groupid:
        groupid_to_nodes = ut.group_items(*zip(*node_to_groupid.items()))
    else:
        groupid_to_nodes = {}
    # Initialize agraph format
    #import utool
    #utool.embed()
    ut.nx_delete_None_edge_attr(graph_)
    agraph = nx.nx_agraph.to_agraph(graph_)
    # Add subgraphs labels
    # TODO: subgraph attrs
    for groupid, nodes in groupid_to_nodes.items():
        subgraph_attrs = {}
        #subgraph_attrs = dict(rankdir='LR')
        #subgraph_attrs['rank'] = 'min'
        subgraph_attrs['rank'] = 'same'
        name = groupid
        name = 'cluster_' + groupid
        agraph.add_subgraph(nodes, name, **subgraph_attrs)
    for node in graph_.nodes():
        # force pinning of node points
        anode = pygraphviz.Node(agraph, node)
        if anode.attr['pin'] == 'true':
            if anode.attr['pos'] is not None and not anode.attr['pos'].endswith('!'):
                import re
                #utool.embed()
                ptstr = anode.attr['pos'].strip('[]').strip(' ')
                ptstr_list = re.split(r'\s+', ptstr)
                pt_arr = np.array(list(map(float, ptstr_list))) / 72.0
                #print('pt_arr = %r' % (pt_arr,))
                new_ptstr_list = list(map(str, pt_arr))
                new_ptstr = ','.join(new_ptstr_list) + '!'
                #print('new_ptstr = %r' % (new_ptstr,))
                anode.attr['pos'] = new_ptstr
    return agraph


def make_agraph_args(kwargs):
    kwargs = kwargs.copy()
    prog = kwargs.pop('prog', 'dot')
    if prog != 'dot':
        kwargs['overlap'] = kwargs.get('overlap', 'false')
    kwargs['splines'] = kwargs.get('splines', 'spline')
    kwargs['notranslate'] = 'true'  # for neato postprocessing
    argparts = ['-G%s=%s' % (key, str(val))
                for key, val in kwargs.items()]
    args = ' '.join(argparts)
    return args


def nx_agraph_layout(graph, orig_graph=None, inplace=False, verbose=None, **kwargs):
    r"""
    orig_graph = graph
    graph = layout_graph

    References:
        http://www.graphviz.org/content/attrs
        http://www.graphviz.org/doc/info/attrs.html
    """
    import networkx as nx
    import pygraphviz

    kwargs = kwargs.copy()
    prog = kwargs.pop('prog', 'dot')
    if prog != 'dot':
        kwargs['overlap'] = kwargs.get('overlap', 'false')
    kwargs['splines'] = kwargs.get('splines', 'spline')
    kwargs['notranslate'] = 'true'  # for neato postprocessing
    argparts = ['-G%s=%s' % (key, str(val))
                for key, val in kwargs.items()]
    args = ' '.join(argparts)
    splines = kwargs['splines']
    if verbose is None:
        verbose = ut.VERBOSE
    if verbose:
        print('args = %r' % (args,))
    # Convert to agraph format
    graph_ = graph.copy()

    ut.nx_ensure_agraph_color(graph_)

    # Reduce size to be in inches not pixels
    # FIXME: make robust to param settings
    # Hack to make the w/h of the node take thae max instead of
    # dot which takes the minimum
    shaped_nodes = [n for n, d in graph_.nodes(data=True) if 'width' in d]
    node_attrs = ut.dict_take(graph_.node, shaped_nodes)
    width_px = np.array(ut.take_column(node_attrs, 'width'))
    height_px = np.array(ut.take_column(node_attrs, 'height'))
    scale = np.array(ut.dict_take_column(node_attrs, 'scale', default=1.0))

    width_in = width_px / 72.0 * scale
    height_in = height_px / 72.0 * scale
    width_in_dict = dict(zip(shaped_nodes, width_in))
    height_in_dict = dict(zip(shaped_nodes, height_in))
    nx.set_node_attributes(graph_, 'width', width_in_dict)
    nx.set_node_attributes(graph_, 'height', height_in_dict)
    ut.nx_delete_node_attr(graph_, 'scale')

    # Check for any nodes with groupids
    node_to_groupid = nx.get_node_attributes(graph_, 'groupid')
    if node_to_groupid:
        groupid_to_nodes = ut.group_items(*zip(*node_to_groupid.items()))
    else:
        groupid_to_nodes = {}
    # Initialize agraph format
    #import utool
    #utool.embed()
    ut.nx_delete_None_edge_attr(graph_)
    agraph = nx.nx_agraph.to_agraph(graph_)
    # Add subgraphs labels
    # TODO: subgraph attrs
    group_attrs = graph.graph.get('groupattrs', {})
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
    for node in graph_.nodes():
        # force pinning of node points
        anode = pygraphviz.Node(agraph, node)
        if anode.attr['pin'] == 'true':
            if anode.attr['pos'] is not None and len(anode.attr['pos']) > 0 and not anode.attr['pos'].endswith('!'):
                import re
                #utool.embed()
                ptstr_ = anode.attr['pos']
                #print('ptstr_ = %r' % (ptstr_,))
                ptstr = ptstr_.strip('[]').strip(' ').strip('()')
                #print('ptstr = %r' % (ptstr,))
                ptstr_list = [x.rstrip(',') for x in re.split(r'\s+', ptstr)]
                #print('ptstr_list = %r' % (ptstr_list,))
                pt_list = list(map(float, ptstr_list))
                #print('pt_list = %r' % (pt_list,))
                pt_arr = np.array(pt_list) / 72.0
                #print('pt_arr = %r' % (pt_arr,))
                new_ptstr_list = list(map(str, pt_arr))
                new_ptstr = ','.join(new_ptstr_list) + '!'
                #print('new_ptstr = %r' % (new_ptstr,))
                anode.attr['pos'] = new_ptstr

    # Run layout
    #print('prog = %r' % (prog,))
    if ut.VERBOSE or verbose > 0:
        print('BEFORE LAYOUT\n' + str(agraph))
    agraph.layout(prog=prog, args=args)
    agraph.draw(ut.truepath('~/test_graphviz_draw.png'))
    if ut.VERBOSE or verbose > 1:
        print('AFTER LAYOUT\n' + str(agraph))

    # TODO: just replace with a single dict of attributes
    node_layout_attrs = ut.ddict(dict)
    edge_layout_attrs = ut.ddict(dict)

    #for node in agraph.nodes():
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

    if orig_graph is not None and kwargs.get('draw_implicit', True):
        # ADD IN IMPLICIT EDGES
        layout_edges = set(ut.nx_edges(graph_, keys=True))
        orig_edges = set(ut.nx_edges(orig_graph, keys=True))
        implicit_edges = list(orig_edges - layout_edges)
        #all_edges = list(set.union(orig_edges, layout_edges))
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
            #print('control_node = %r' % (control_node,))
            node1_attr1 = parse_anode_layout_attrs(control_node)
            #print('node1_attr1 = %r' % (node1_attr1,))

            implicit_kw = kwargs.copy()
            implicit_kw['overlap'] = 'true'
            #del implicit_kw['overlap']  # can cause node positions to change
            argparts = ['-G%s=%s' % (key, str(val))
                        for key, val in implicit_kw.items()]
            args = ' '.join(argparts)
            #print('args = %r' % (args,))

            #import utool
            #utool.embed()

            agraph.layout(prog='neato', args='-n ' + args)
            agraph.draw(ut.truepath('~/implicit_test_graphviz_draw.png'))
            if ut.VERBOSE or verbose:
                print('AFTER IMPLICIT LAYOUT\n' + str(agraph))

            control_node = pygraphviz.Node(agraph, node)
            print('control_node = %r' % (control_node,))
            node1_attr2 = parse_anode_layout_attrs(control_node)
            print('node1_attr2 = %r' % (node1_attr2,))

            # graph positions shifted
            # This is not the right place to divide by 72
            translation = (node1_attr1['pos'] - node1_attr2['pos'] )
            #print('translation = %r' % (translation,))
            #translation = np.array([0, 0])
            print('translation = %r' % (translation,))

            #for iedge in all_edges:
            for iedge in implicit_edges:
                aedge = pygraphviz.Edge(agraph, *iedge)
                iedge_attrs = parse_aedge_layout_attrs(aedge, translation)
                for key, val in iedge_attrs.items():
                    edge_layout_attrs[key][iedge] = val

    graph_layout_attrs = dict(
        splines=splines
    )

    layout_info = {
        'graph': graph_layout_attrs,
        'edge': dict(edge_layout_attrs),
        'node': dict(node_layout_attrs),
    }

    if inplace:
        if orig_graph is not None:
            graph = orig_graph
        apply_graph_layout_attrs(graph, layout_info)

    return graph, layout_info


def parse_point(ptstr):
    try:
        xx, yy = ptstr.strip('!').split(',')
        xy = np.array((float(xx), float(yy)))
    except:
        xy = None
    return xy


def parse_anode_layout_attrs(anode):
    node_attrs = {}
    #try:
    xx, yy = anode.attr['pos'].strip('!').split(',')
    xy = np.array((float(xx), float(yy)))
    #except:
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
    #print('apos = %r' % (apos,))
    end_pt = None
    start_pt = None
    #if '-' in apos:
    #    import utool
    #    utool.embed()

    def safeadd(x, y):
        if x is None or y is None:
            return None
        return x + y

    strpos_list = apos.split(' ')
    strtup_list = [ea.split(',') for ea in strpos_list]

    ctrl_ptstrs = [ea for ea in  strtup_list if ea[0] not in 'es']
    end_ptstrs = [ea[1:] for ea in  strtup_list[0:2] if ea[0] == 'e']
    start_ptstrs = [ea[1:] for ea in  strtup_list[0:2] if ea[0] == 's']
    assert len(end_ptstrs) <= 1
    assert len(start_ptstrs) <= 1
    if len(end_ptstrs) == 1:
        end_pt = np.array([float(f) for f in end_ptstrs[0]])
    if len(start_ptstrs) == 1:
        start_pt = np.array([float(f) for f in start_ptstrs[0]])
    ctrl_pts = np.array([tuple([float(f) for f in ea])
                         for ea in ctrl_ptstrs])
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
    edge_attrs['head_lp'] = safeadd(parse_point(adata.get('head_lp', None)),
                                    translation)
    edge_attrs['tail_lp'] = safeadd(parse_point(adata.get('tail_lp', None)),
                                    translation)
    return edge_attrs


def format_anode_pos(xy, pin=True):
    xx, yy = xy
    return '%f,%f%s' % (xx, yy, '!' * pin)


def _get_node_size(graph, node, node_size):
    if node_size is not None and node in node_size:
        return node_size[node]
    nattrs = graph.node[node]
    scale = nattrs.get('scale', 1.0)
    if 'width' in nattrs and 'height' in nattrs:
        width = nattrs['width'] * scale
        height = nattrs['height'] * scale
    elif 'radius' in nattrs:
        width = height = nattrs['radius'] * scale
    else:
        if 'image' in nattrs:
            img_fpath = nattrs['image']
            width, height = vt.image.open_image_size(img_fpath)
        else:
            height = width = 1100 / 50 * scale
    return width, height


def draw_network2(graph, layout_info, ax, as_directed=None, hacknoedge=False,
                  hacknode=False, verbose=None, **kwargs):
    """
    fancy way to draw networkx graphs without directly using networkx

    # python -m ibeis.annotmatch_funcs review_tagged_joins --dpath ~/latex/crall-candidacy-2015/ --save figures4/mergecase.png --figsize=15,15 --clipwhite --diskshow
    # python -m dtool --tf DependencyCache.make_graph --show
    """
    import plottool as pt

    patch_dict = {
        'patch_frame_dict': {},
        'node_patch_dict': {},
        'edge_patch_dict': {},
        'arrow_patch_list': {},
    }

    font_prop = pt.parse_fontkw(**kwargs)
    #print('font_prop = %r' % (font_prop,))
    #print('font_prop.get_name() = %r' % (font_prop.get_name() ,))

    # print('layout_info = %r' % (layout_info,))
    node_pos = layout_info['node']['pos']
    node_size = layout_info['node']['size']
    splines = layout_info['graph']['splines']
    # edge_startpoints = layout_info['edge']['start_pt']

    if as_directed is None:
        as_directed = graph.is_directed()

    # Draw nodes
    for node, nattrs in graph.nodes(data=True):
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

        if isinstance(node_color, six.string_types) and node_color.startswith('#'):
            import matplotlib.colors as colors
            node_color = colors.hex2color(node_color[0:7])
            #intcolor = int(node_color.replace('#', '0x'), 16)
        node_color = node_color[0:3]
        patch_kw = dict(alpha=alpha_, color=node_color)
        node_shape = nattrs.get('shape', 'ellipse')
        if node_shape == 'circle':
            # divide by 2 seems to work for agraph
            radius = min(_get_node_size(graph, node, node_size)) / 2.0
            patch = mpl.patches.Circle(xy, radius=radius, **patch_kw)
        elif node_shape == 'ellipse':
            # divide by 2 seems to work for agraph
            width, height = np.array(_get_node_size(graph, node, node_size))
            patch = mpl.patches.Ellipse(xy, width, height, **patch_kw)
        elif node_shape in ['none', 'box', 'rect', 'rectangle', 'rhombus']:
            width, height = _get_node_size(graph, node, node_size)
            angle = 45 if node_shape == 'rhombus' else 0
            xy_bl = (xy[0] - width // 2, xy[1] - height // 2)

            # rounded = angle == 0
            rounded = 'rounded' in graph.node.get(node, {}).get('style', '')
            isdiag = 'diagonals' in graph.node.get(node, {}).get('style', '')

            if rounded:
                from matplotlib import patches
                rpad = 20
                xy_bl = np.array(xy_bl) + rpad
                width -= rpad
                height -= rpad
                boxstyle = patches.BoxStyle.Round(pad=rpad)
                patch = mpl.patches.FancyBboxPatch(
                    xy_bl, width, height, boxstyle=boxstyle, **patch_kw)
            else:
                bbox = list(xy_bl) + [width, height]
                if isdiag:
                    center_xy  = vt.bbox_center(bbox)
                    _xy =  np.array(center_xy)
                    newverts_ = [
                        _xy + [         0, -height / 2],
                        _xy + [-width / 2,           0],
                        _xy + [         0,  height / 2],
                        _xy + [ width / 2,           0],
                    ]
                    patch = mpl.patches.Polygon(newverts_, **patch_kw)
                else:
                    # patch = pt.make_bbox(bbox, theta=angle, fill=True,
                    #                      **patch_kw)
                    patch = mpl.patches.Rectangle(
                        xy_bl, width, height, angle=angle,
                        **patch_kw)
            patch.center = xy
        #if style == 'rounded'
        #elif node_shape in ['roundbox']:
        elif node_shape == 'stack':
            width, height = _get_node_size(graph, node, node_size)
            xy_bl = (xy[0] - width // 2, xy[1] - height // 2)
            patch = pt.cartoon_stacked_rects(xy_bl, width, height, **patch_kw)
            patch.xy = xy

        if True:
            framewidth = nattrs.get('framewidth', 0)
            if framewidth > 0:
                framecolor = nattrs.get('framecolor', node_color)
                if isinstance(framecolor, six.string_types) and framecolor.startswith('#'):
                    import matplotlib.colors as colors
                    framecolor = colors.hex2color(framecolor[0:7])
                #print('framecolor = %r' % (framecolor,))
                alpha = 1.0
                if framecolor is None:
                    framecolor = pt.BLACK
                    alpha = 0.0
                if framewidth is True:
                    figsize = ut.get_argval('--figsize', type_=list, default=None)
                    if figsize is not None:
                        # HACK
                        graphsize = max(figsize)
                        framewidth = graphsize / 4
                    else:
                        framewidth = 3.0
                lw = framewidth
                frame = pt.make_bbox(bbox, bbox_color=framecolor, ax=ax, lw=lw, alpha=alpha)
                patch_dict['patch_frame_dict'][node] = frame

        #patch_dict[node] = patch
        x, y = xy
        text = str(node)
        if label is not None:
            #text += ': ' + str(label)
            text = label
        if hacknode or not using_image:
            pt.ax_absolute_text(x, y, text, ha='center', va='center',
                                fontproperties=font_prop)
        patch_dict['node_patch_dict'][node] = (patch)

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
    if edge_pos is not None:
        for edge, pts in edge_pos.items():

            data = get_default_edge_data(graph, edge)

            if data.get('style', None) == 'invis':
                continue

            alpha = data.get('alpha', None)

            defaultcolor = pt.BLACK[0:3]
            if alpha is None:
                if data.get('implicit', False):
                    alpha = .5
                    defaultcolor = pt.GREEN[0:3]
                else:
                    alpha = 1.0
            color = data.get('color', defaultcolor)
            if color is None:
                color = defaultcolor
            if isinstance(color, six.string_types) and color.startswith('#'):
                import matplotlib.colors as colors
                color = colors.hex2color(color[0:7])
            color = color[0:3]

            layout_info['edge']['ctrl_pts'][edge]
            layout_info['edge']['start_pt'][edge]

            offset = 0 if graph.is_directed() else 0
            #color = data.get('color', color)[0:3]
            start_point = pts[offset]
            other_points = pts[offset + 1:].tolist()  # [0:3]
            verts = [start_point] + other_points

            MOVETO = mpl.path.Path.MOVETO
            LINETO = mpl.path.Path.LINETO

            if splines in ['line', 'polyline', 'ortho']:
                CODE = LINETO
            elif splines == 'curved':
                #CODE = mpl.path.Path.CURVE3
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

            end_pt = layout_info['edge']['end_pt'][edge]

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
                width =  graphsize / 15
                width = ut.get_argval('--arrow-width', default=width)
                lw = ut.get_argval('--line-width', default=lw)
                #print('width = %r' % (width,))
            else:
                width = .5
                lw = 1.0
                try:
                    # Compute arrow width using estimated graph size
                    if node_size is not None and node_pos is not None:
                        xys = np.array(ut.take(node_pos, node_pos.keys())).T
                        whs = np.array(ut.take(node_size, node_pos.keys())).T
                        bboxes = vt.bbox_from_xywh(xys, whs, [.5, .5])
                        extents = vt.extent_from_bbox(bboxes)
                        tl_pts = np.array([extents[0], extents[2]]).T
                        br_pts = np.array([extents[1], extents[3]]).T
                        pts = np.vstack([tl_pts, br_pts])
                        extent = vt.get_pointset_extents(pts)
                        graph_w, graph_h = vt.bbox_from_extent(extent)[2:4]
                        graph_dim = np.sqrt(graph_w ** 2 + graph_h ** 2)
                        width = graph_dim * .0005
                except Exception:
                    pass

            if not as_directed and end_pt is not None:
                pass

            lw = data.get('lw', lw)

            patch = mpl.patches.PathPatch(path, facecolor='none', lw=lw,
                                          edgecolor=color,
                                          alpha=alpha,
                                          joinstyle='bevel')
            if as_directed:
                if end_pt is not None:
                    dxy = (np.array(end_pt) - other_points[-1])
                    dxy = (dxy / np.sqrt(np.sum(dxy ** 2))) * .1
                    dx, dy = dxy
                    rx, ry = end_pt[0], end_pt[1]
                    patch1 = mpl.patches.FancyArrow(rx, ry, dx, dy, width=width,
                                                    length_includes_head=True,
                                                    color=color,
                                                    head_starts_at_zero=False)
                else:
                    dxy = (np.array(other_points[-1]) - other_points[-2])
                    dxy = (dxy / np.sqrt(np.sum(dxy ** 2))) * .1
                    dx, dy = dxy
                    rx, ry = other_points[-1][0], other_points[-1][1]
                    patch1 = mpl.patches.FancyArrow(rx, ry, dx, dy, width=width,
                                                    length_includes_head=True,
                                                    color=color,
                                                    head_starts_at_zero=True)
                #ax.add_patch(patch1)
                patch_dict['arrow_patch_list'][edge] = (patch1)

            taillabel = layout_info['edge']['taillabel'][edge]
            #ha = 'left'
            #ha = 'right'
            ha = 'center'
            va = 'center'
            labelcolor = color  # TODO allow for different colors

            labelcolor = data.get('labelcolor', color)
            if isinstance(color, six.string_types) and labelcolor.startswith('#'):
                import matplotlib.colors as colors
                labelcolor = colors.hex2color(color[0:7])
            labelcolor = labelcolor[0:3]

            if taillabel:
                taillabel_pos = layout_info['edge']['tail_lp'][edge]
                ax.annotate(taillabel, xy=taillabel_pos, xycoords='data',
                            color=labelcolor,
                            va=va, ha=ha, fontproperties=font_prop)
            headlabel = layout_info['edge']['headlabel'][edge]
            if headlabel:
                headlabel_pos = layout_info['edge']['head_lp'][edge]
                ax.annotate(headlabel, xy=headlabel_pos, xycoords='data',
                            color=labelcolor,
                            va=va, ha=ha, fontproperties=font_prop)
            label = layout_info['edge']['label'][edge]
            if label:
                label_pos = layout_info['edge']['lp'][edge]
                ax.annotate(label, xy=label_pos, xycoords='data',
                            color=labelcolor,
                            va=va, ha=ha, fontproperties=font_prop)
            patch_dict['edge_patch_dict'][edge] = patch
            #ax.add_patch(patch)

    if verbose:
        print('Adding %r node patches ' % (len(patch_dict['node_patch_dict'],)))
        print('Adding %r edge patches ' % (len(patch_dict['edge_patch_dict'],)))

    for frame in patch_dict['patch_frame_dict'].values():
        ax.add_patch(frame)

    for patch1 in patch_dict['arrow_patch_list'].values():
        ax.add_patch(patch1)

    use_collections = False
    if use_collections:
        edge_coll = mpl.collections.PatchCollection(patch_dict['edge_patch_dict'].values())
        node_coll = mpl.collections.PatchCollection(patch_dict['node_patch_dict'].values())
        #coll.set_facecolor(fcolor)
        #coll.set_alpha(alpha)
        #coll.set_linewidth(lw)
        #coll.set_edgecolor(color)
        #coll.set_transform(ax.transData)
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
    return patch_dict


# def arrowed_spines(ax=None, arrow_length=20, labels=('', ''), arrowprops=None):
#     """
#     TODO arrow splines not spines
#     References:
#         https://gist.github.com/joferkington/3845684
#     """
#     xlabel, ylabel = labels
#     import plottool as pt
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
        python -m plottool.nx_helpers
        python -m plottool.nx_helpers --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
