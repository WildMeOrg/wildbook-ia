r"""
Helpers for graph plotting

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
(print, rrr, profile) = ut.inject2(__name__, '[df2]')


def show_nx(graph, with_labels=True, fnum=None, pnum=None, layout='pydot',
            ax=None, pos=None, img_dict=None, title=None, layoutkw=None,
            **kwargs):
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
        for node in img_dict.keys():
            nattr = graph.node[node]
            if 'width' not in nattr:
                img_fpath = nattr['image']
                width, height = vt.image.open_image_size(img_fpath)
                # nx.set_node_attributes(graph, 'width', {node: width})
                # nx.set_node_attributes(graph, 'height', {node: height})
                # nx.set_node_attributes(graph, 'width', {node: width / 72.0})
                # nx.set_node_attributes(graph, 'height', {node: height / 72.0})

    node_pos = pos
    edge_pos = None
    layout_dict = {}
    edge_endpoints = None
    if node_pos is None:
        layout_dict = get_nx_layout(graph, layout, layoutkw=layoutkw)
        node_pos = layout_dict['node_pos']
        edge_pos = layout_dict['edge_pos']
        edge_endpoints = layout_dict['edge_endpoints']

    # zoom = kwargs.pop('zoom', .4)
    node_size = layout_dict['node_size']
    frameon = kwargs.pop('frameon', True)
    splines = layout_dict['splines']
    draw_network2(graph, node_pos, ax, edge_pos=edge_pos, splines=splines,
                  node_size=node_size, edge_endpoints=edge_endpoints, **kwargs)
    ax.grid(False)
    pt.plt.axis('equal')

    ax.autoscale()
    ax.autoscale_view(True, True, True)

    if node_size is not None:
        half_size_arr = np.array(ut.take(node_size, graph.nodes())) / 2.
        pos_arr = np.array(ut.take(node_pos, graph.nodes()))
        # autoscale does not seem to work
        ul_pos = pos_arr - half_size_arr
        br_pos = pos_arr + half_size_arr
        xmin, ymin = ul_pos.min(axis=0)
        xmax, ymax = br_pos.max(axis=0)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    #pt.plt.axis('off')
    ax.set_xticks([])
    ax.set_yticks([])

    plotinfo = {
        'pos': node_pos,
        'node_size': node_size,
    }

    if img_dict is not None and len(img_dict) > 0:
        node_list = sorted(img_dict.keys())
        pos_list = ut.dict_take(node_pos, node_list)
        img_list = ut.dict_take(img_dict, node_list)
        size_list = ut.dict_take(node_size, node_list)
        color_list = ut.dict_take(nx.get_node_attributes(graph, 'color'), node_list, None)
        #node_attrs = ut.dict_take(graph.node, node_list)
        # Rename to node_scale?
        #scale_list = np.array(ut.dict_take_column(node_attrs, 'scale',
        #                                          default=None))
        #img_list = [img if scale is None else vt.resize_image_by_scale(img, scale)
        #            for scale, img in zip(scale_list, img_list)]

        # TODO; frames without images
        imgdat = pt.netx_draw_images_at_positions(img_list, pos_list, size_list, color_list, frameon=frameon)
        imgdat['node_list'] = node_list
        plotinfo['imgdat'] = imgdat

    if title is not None:
        pt.set_title(title)
    return plotinfo


def netx_draw_images_at_positions(img_list, pos_list, size_list, color_list, frameon=True):
    """
    Overlays images on a networkx graph

    References:
        https://gist.github.com/shobhit/3236373
        http://matplotlib.org/examples/pylab_examples/demo_annotation_box.html
        http://stackoverflow.com/questions/11487797/mpl-basemap-overlay-small-image
        http://matplotlib.org/api/text_api.html
        http://matplotlib.org/api/offsetbox_api.html

    TODO: look into DraggableAnnotation
    """
    print('[viz_graph] drawing %d images' % len(img_list))
    # Thumb stackartist
    import plottool as pt
    ax  = pt.gca()
    artist_list = []
    offset_img_list = []

    # Ensure all images have been read
    img_list_ = [vt.convert_colorspace(vt.imread(img), 'RGB')
                 if isinstance(img, six.string_types) else img
                 for img in img_list]
    size_list_ = [vt.get_size(img) if size is None else size
                  for size, img in zip(size_list, img_list)]

    as_offset_image = False

    if as_offset_image:
        # THIS DOES NOT DO WHAT I WANT
        # Scales the image with data coords
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        bboxkw = dict(
            xycoords='data',
            boxcoords='offset points',
            #boxcoords='data',
            pad=0.25, frameon=frameon,
            # frameon=False, bboxprops=dict(fc="cyan"),
            # arrowprops=dict(arrowstyle="->"))
        )
        for pos, img in zip(pos_list, img_list_):
            offset_img = OffsetImage(img, zoom=.4)
            artist = AnnotationBbox(offset_img, pos, xybox=(-0., 0.), **bboxkw)
            offset_img_list.append(offset_img)
            artist_list.append(artist)
    else:
        # THIS DOES EXACTLY WHAT I WANT
        # Ties the image to data coords
        for pos, img, size, color in zip(pos_list, img_list_, size_list_, color_list):
            bbox = vt.bbox_from_center_wh(pos, size)
            extent = vt.extent_from_bbox(bbox)
            pt.plt.imshow(img, extent=extent)
            if frameon:
                alpha = 1
                if color is None:
                    color = pt.BLACK
                    alpha = 0
                    #color = pt.WHITE
                #color = pt.ORANGE
                patch = pt.make_bbox(bbox, bbox_color=color, ax=ax, lw=3, alpha=alpha)
                artist_list.append(patch)
    for artist in artist_list:
        ax.add_artist(artist)

    imgdat = {
        'offset_img_list': offset_img_list,
        'artist_list': artist_list,
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


def get_nx_layout(graph, layout, layoutkw=None):
    import networkx as nx
    only_explicit = True
    if only_explicit:
        explicit_graph = graph.__class__()
        explicit_nodes = graph.nodes(data=True)
        explicit_edges = [(n1, n2, data) for (n1, n2, data) in graph.edges(data=True)
                          if data.get('implicit', False) is not True]
        explicit_graph.add_nodes_from(explicit_nodes)
        explicit_graph.add_edges_from(explicit_edges)
        layout_graph = explicit_graph
    else:
        layout_graph = graph

    if layoutkw is None:
        layoutkw = {}
    layout_info = {}

    if layout == 'agraph':
        # PREFERED LAYOUT WITH MOST CONTROL
        _, layout_info = nx_agraph_layout(layout_graph, orig_graph=graph, **layoutkw)
        node_pos = layout_info['node_pos']
    elif layout == 'pydot':
        node_pos = nx.nx_pydot.pydot_layout(layout_graph, prog='dot')
    elif layout == 'graphviz':
        node_pos = nx.nx_agraph.graphviz_layout(layout_graph)
    elif layout == 'pygraphviz':
        node_pos = nx.nx_agraph.pygraphviz_layout(layout_graph)
    elif layout == 'spring':
        _pos = nx.spectral_layout(layout_graph, scale=500)
        node_pos = nx.fruchterman_reingold_layout(layout_graph, pos=_pos,
                                                  scale=500, iterations=100)
        node_pos = ut.map_dict_vals(lambda x: x * 500, node_pos)
    elif layout == 'circular':
        node_pos = nx.circular_layout(layout_graph, scale=100)
    elif layout == 'spectral':
        node_pos = nx.spectral_layout(layout_graph, scale=500)
    elif layout == 'shell':
        node_pos = nx.shell_layout(layout_graph, scale=100)
    else:
        raise ValueError('Undefined layout = %r' % (layout,))
    layout_dict = {}
    layout_dict['node_pos'] = layout_info.get('node_pos', node_pos)
    layout_dict['edge_pos'] = layout_info.get('edge_pos', None)
    layout_dict['splines'] = layout_info.get('splines', 'line')
    layout_dict['node_size'] = layout_info.get('node_size', None)
    layout_dict['edge_endpoints'] = layout_info.get('edge_endpoints', None)
    return layout_dict


def nx_agraph_layout(graph, orig_graph=None, inplace=False, **kwargs):
    r"""
    orig_graph = graph
    graph = layout_graph

    References:
        http://www.graphviz.org/doc/info/attrs.html
    """
    import networkx as nx
    import pygraphviz

    if not inplace:
        graph = graph.copy()

    kwargs = kwargs.copy()
    factor = kwargs.pop('factor', 1.0)
    prog = kwargs.pop('prog', 'dot')

    if True:
        kwargs['splines'] = kwargs.get('splines', 'spline')
        kwargs['pack'] = kwargs.get('pack', 'true')
        kwargs['packmode'] = kwargs.get('packmode', 'cluster')
    if prog == 'dot':
        kwargs['ranksep'] = kwargs.get('ranksep', 1.5 * factor)
        #kwargs['rankdir'] = kwargs.get('rankdir', 'LR')
        kwargs['nodesep'] = kwargs.get('nodesep', 1 * factor)
        kwargs['clusterrank'] = kwargs.get('clusterrank', 'local')
    if prog != 'dot':
        kwargs['overlap'] = kwargs.get('overlap', 'prism')
        kwargs['sep'] = kwargs.get('sep', 1 / 8.)
        kwargs['esep'] = kwargs.get('esep', (1 / 8) * .8)
        #assert kwargs['esep']  < kwargs['sep']
    if prog == 'neato':
        kwargs['mode'] = 'major'
        if kwargs['mode'] == 'ipsep':
            pass
            #kwargs['overlap'] = 'ipsep'
        pass

    splines = kwargs['splines']

    kwargs['notranslate'] = 'true'  # for neato postprocessing

    argparts = ['-G%s=%s' % (key, str(val))
                for key, val in kwargs.items()]
    args = ' '.join(argparts)
    print('args = %r' % (args,))
    # Convert to agraph format
    graph_ = graph.copy()
    from plottool import color_funcs

    def _fix_agraph_color(data):
        try:
            orig_color = data.get('color', None)
            color = orig_color
            if color is not None and not isinstance(color, six.string_types):
                #if isinstance(color, np.ndarray):
                #    color = color.tolist()
                color = tuple(color_funcs.ensure_base255(color))
                if len(color) == 3:
                    data['color'] = '#%02x%02x%02x' % color
                else:
                    data['color'] = '#%02x%02x%02x%02x' % color
        except Exception as ex:
            ut.printex(ex, keys=['color', 'orig_color', 'data'])
            raise

    for node, node_data in graph_.nodes(data=True):
        _fix_agraph_color(node_data)

    for u, v, edge_data in graph_.edges(data=True):
        _fix_agraph_color(edge_data)

    # Reduce size to be in inches not pixels
    # FIXME; make robust to param settings
    # Hack to make the w/h of the node take thae max instead of
    # dot which takes the minimum
    shaped_nodes = [n for n, d in graph_.nodes(data=True) if 'width' in d]
    node_attrs = ut.dict_take(graph_.node, shaped_nodes)
    width_px = np.array(ut.take_column(node_attrs, 'width'))
    height_px = np.array(ut.take_column(node_attrs, 'height'))
    scale = np.array(ut.dict_take_column(node_attrs, 'scale', default=1.0))

    dimsize_in = np.maximum(width_px, height_px)
    dimsize_in = dimsize_in / 72.0 * scale
    dimsize_in_dict = dict(zip(shaped_nodes, dimsize_in))
    width_in = dimsize_in_dict
    height_in = dimsize_in_dict
    nx.set_node_attributes(graph_, 'width', width_in)
    nx.set_node_attributes(graph_, 'height', height_in)
    ut.nx_delete_node_attr(graph_, 'scale')

    # Check for any nodes with groupids
    node_to_groupid = nx.get_node_attributes(graph_, 'groupid')
    if node_to_groupid:
        groupid_to_nodes = ut.group_items(*zip(*node_to_groupid.items()))
    else:
        groupid_to_nodes = {}

    # Initialize agraph format
    agraph = nx.nx_agraph.to_agraph(graph_)

    # Add subgraphs labels
    for groupid, nodes in groupid_to_nodes.items():
        subgraph_attrs = {}
        # TODO: subgraph attrs
        #subgraph_attrs = dict(rankdir='LR')
        #subgraph_attrs['rank'] = 'min'
        subgraph_attrs['rank'] = 'source'
        name = groupid
        name = 'cluster_' + groupid
        agraph.add_subgraph(nodes, name, **subgraph_attrs)

    # Run layout
    # print('BEFORE LAYOUT')
    print('prog = %r' % (prog,))
    #print(agraph)
    agraph.layout(prog=prog, args=args)
    agraph.draw('test_graphviz_draw.png')
    # print('AFTER LAYOUT')
    #print(agraph)

    adpi = 72.0

    def parse_anode_pos(anode):
        try:
            xx, yy = anode.attr['pos'].split(',')
            xy = np.array((float(xx), float(yy))) / factor
        except:
            xy = np.array((0.0, 0.0))
        return xy

    def parse_aedge_pos(aedge):
        """
        parse grpahviz splineType
        """
        apos = aedge.attr['pos']
        endp = None
        startp = None
        strpos_list = apos.split(' ')
        strtup_list = [ea.split(',') for ea in strpos_list]
        edge_ctrlpts = [tuple([float(f) for f in ea if f not in 'es'])
                        for ea in strtup_list]
        edge_ctrlpts = np.array(edge_ctrlpts)
        if len(strtup_list) > 0 and strtup_list[0][0] == 'e':
            ea0 = strtup_list[0]
            endp = tuple([float(f) for f in ea0[1:]])
        if len(strtup_list) > 0 and strtup_list[0][0] == 's':
            ea0 = strtup_list[0]
            startp = tuple([float(f) for f in ea0[1:]])
        elif len(strtup_list) > 1 and strtup_list[1][0] == 's':
            ea1 = strtup_list[1]
            startp = tuple([float(f) for f in ea1[1:]])
        if startp:
            startp = np.array(startp) / factor
        if endp:
            endp = np.array(endp) / factor
        edge_ctrlpts = edge_ctrlpts / factor
        return edge_ctrlpts, startp, endp

    def parse_anode_size(anode):
        width = float(anode.attr['width']) * adpi / factor
        height = float(anode.attr['height']) * adpi / factor
        return width, height

    def format_anode_pos(xy, pin=True):
        xx, yy = xy * factor
        return '%f,%f%s' % (xx, yy, '!' * pin)

    node_pos = {}
    node_size = {}
    edge_pos = {}
    edge_endpoints = {}

    #for node in agraph.nodes():
    for node in graph.nodes():
        anode = pygraphviz.Node(agraph, node)
        node_pos[node] = parse_anode_pos(anode)
        node_size[node] = parse_anode_size(anode)

    if graph.is_multigraph():
        edges = graph.edges(keys=True)
    else:
        edges = graph.edges()

    for edge in edges:
        aedge = pygraphviz.Edge(agraph, *edge)
        edge_ctrlpts, startp, endp = parse_aedge_pos(aedge)
        edge_pos[edge] = edge_ctrlpts
        edge_endpoints[edge] = endp

    if orig_graph is not None:
        if graph.is_multigraph():
            layout_edges = set(graph.edges(keys=True))
            orig_edges = set(orig_graph.edges(keys=True))
        else:
            layout_edges = set(graph.edges())
            orig_edges = set(orig_graph.edges())
        implicit_edges = orig_edges - layout_edges
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

            agraph.draw('test_graphviz_draw_implicit.png')

            # Route the implicit edges (must use neato)
            #agraph.layout(prog=prog, args=args)
            agraph.layout(prog='neato', args='-n ' + args)
            #print(agraph)
            #agraph.layout(prog='neato', args='-n ' + args)

            #for node in agraph.nodes():
            #    anode = pygraphviz.Node(agraph, node)
            #    #print(parse_anode_pos(anode) / node_pos[node])
            #    print(np.array(parse_anode_size(anode)) / np.array(node_size[node]))

            for iedge in implicit_edges:
                aedge = pygraphviz.Edge(agraph, *iedge)
                edge_ctrlpts, startp, endp = parse_aedge_pos(aedge)
                edge_pos[iedge] = edge_ctrlpts
                edge_endpoints[iedge] = endp

    layout_info = dict(
        node_pos=node_pos,
        splines=splines,
        edge_pos=edge_pos,
        node_size=node_size,
        edge_endpoints=edge_endpoints,
    )

    return graph, layout_info


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


def draw_network2(graph, node_pos, ax,
                  hacknoedge=False, hacknonode=False, splines='line',
                  as_directed=None, edge_pos=None, edge_endpoints=None, node_size=None, use_arc=True):
    """
    fancy way to draw networkx graphs without directly using networkx
    """
    import plottool as pt

    node_patch_list = []
    edge_patch_list = []

    patches = {}

    ###
    # Draw nodes
    for node, nattrs in graph.nodes(data=True):
        # shape = nattrs.get('shape', 'circle')
        if nattrs is None:
            nattrs = {}
        label = nattrs.get('label', None)
        alpha = nattrs.get('alpha', .5)
        node_color = nattrs.get('color', pt.NEUTRAL_BLUE)
        if node_color is None:
            node_color = pt.NEUTRAL_BLUE
        node_color = node_color[0:3]
        xy = node_pos[node]
        if 'image' in nattrs:
            alpha_ = 0.0
        else:
            alpha_ = alpha
        patch_kw = dict(alpha=alpha_, color=node_color)
        node_shape = nattrs.get('shape', 'circle')
        if node_shape == 'circle':
            # divide by 2 seems to work for agraph
            radius = min(_get_node_size(graph, node, node_size)) / 2.0
            patch = mpl.patches.Circle(xy, radius=radius, **patch_kw)
        elif node_shape in ['rect', 'rhombus']:
            width, height = _get_node_size(graph, node, node_size)
            angle = 0 if node_shape == 'rect' else 45
            xy_bl = (xy[0] - width // 2, xy[1] - height // 2)
            patch = mpl.patches.Rectangle(
                xy_bl, width, height, angle=angle, **patch_kw)
            patch.center = xy
        patches[node] = patch
        x, y = xy
        text = node
        if label is not None:
            text += ': ' + str(label)
        if not hacknonode and 'image' not in nattrs:
            pt.ax_absolute_text(x, y, text, ha='center', va='center')
        node_patch_list.append(patch)
    ###
    # Draw Edges
    if  edge_pos is None:
        # TODO: rectify with spline method
        seen = {}
        edge_list = graph.edges(data=True)
        for (u, v, data) in edge_list:
            edge = (u, v)
            #n1 = graph.node[u]['patch']
            #n2 = graph.node[v]['patch']
            n1 = patches[u]
            n2 = patches[v]

            # Bend left / right depending on node positions
            dir_ = np.sign(n1.center[0] - n2.center[0])
            inc = dir_ * 0.1
            rad = dir_ * 0.2
            posA = list(n1.center)
            posB = list(n2.center)
            # Make duplicate edges more bendy to see them
            if edge in seen:
                posB[0] += 10
                rad = seen[edge] + inc
            seen[edge] = rad

            if (v, u) in seen:
                rad = seen[edge] * -1

            if not use_arc:
                rad = 0

            if data.get('implicit', False):
                alpha = .2
                color = pt.GREEN[0:3]
            else:
                alpha = 0.5
                color = pt.BLACK[0:3]

            color = data.get('color', color)[0:3]

            arrowstyle = '-' if not graph.is_directed() else '-|>'

            arrow_patch = mpl.patches.FancyArrowPatch(
                posA, posB, patchA=n1, patchB=n2,
                arrowstyle=arrowstyle, connectionstyle='arc3,rad=%s' % rad,
                mutation_scale=10.0, lw=2, alpha=alpha, color=color)

            # endpoint1 = edge_verts[0]
            # endpoint2 = edge_verts[len(edge_verts) // 2 - 1]
            if (data.get('ismulti', False) or data.get('isnwise', False) or
                 data.get('local_input_id', False)):
                pt1 = np.array(arrow_patch.patchA.center)
                pt2 = np.array(arrow_patch.patchB.center)
                frac_thru = 4
                edge_verts = arrow_patch.get_verts()
                edge_verts = vt.unique_rows(edge_verts)
                sorted_verts = edge_verts[vt.L2(edge_verts, pt1).argsort()]
                if len(sorted_verts) <= 4:
                    mpl_bbox = arrow_patch.get_extents()
                    bbox = [mpl_bbox.x0, mpl_bbox.y0, mpl_bbox.width, mpl_bbox.height]
                    endpoint1 = vt.closest_point_on_bbox(pt1, bbox)
                    endpoint2 = vt.closest_point_on_bbox(pt2, bbox)
                    beta = (1 / frac_thru)
                    alpha = 1 - beta
                    text_point1 = (alpha * endpoint1) + (beta * endpoint2)
                else:
                    #print('sorted_verts = %r' % (sorted_verts,))
                    #text_point1 = sorted_verts[len(sorted_verts) // (frac_thru)]
                    #frac_thru = 3
                    frac_thru = 6

                    text_point1 = edge_verts[(len(edge_verts) - 2) // (frac_thru) + 1]

                font_prop = mpl.font_manager.FontProperties(family='monospace',
                                                            weight='light',
                                                            size=14)
                if data.get('local_input_id', False):
                    text = data['local_input_id']
                    if text == '1':
                        text = ''
                elif data.get('ismulti', False):
                    text = '*'
                else:
                    text = str(data.get('nwise_idx', '!'))
                ax.annotate(text, xy=text_point1, xycoords='data', va='center',
                            ha='center', fontproperties=font_prop)
                #bbox=dict(boxstyle='round', fc=None, alpha=1.0))
            if data.get('label', False):
                pt1 = np.array(arrow_patch.patchA.center)
                pt2 = np.array(arrow_patch.patchB.center)
                frac_thru = 2
                edge_verts = arrow_patch.get_verts()
                edge_verts = vt.unique_rows(edge_verts)
                sorted_verts = edge_verts[vt.L2(edge_verts, pt1).argsort()]
                if len(sorted_verts) <= 4:
                    mpl_bbox = arrow_patch.get_extents()
                    bbox = [mpl_bbox.x0, mpl_bbox.y0, mpl_bbox.width, mpl_bbox.height]
                    endpoint1 = vt.closest_point_on_bbox(pt1, bbox)
                    endpoint2 = vt.closest_point_on_bbox(pt2, bbox)
                    print('sorted_verts = %r' % (sorted_verts,))
                    beta = (1 / frac_thru)
                    alpha = 1 - beta
                    text_point1 = (alpha * endpoint1) + (beta * endpoint2)
                else:
                    text_point1 = sorted_verts[len(sorted_verts) // (frac_thru)]
                    ax.annotate(data['label'], xy=text_point1, xycoords='data',
                                va='center', ha='center',
                                bbox=dict(boxstyle='round', fc='w'))
            #ax.add_patch(arrow_patch)
            edge_patch_list.append(arrow_patch)
    if edge_pos is not None:
        # NEW WAY OF DRAWING EDGEES
        if as_directed is None:
            as_directed = graph.is_directed()
        for edge, pts in edge_pos.items():
            data = graph.get_edge_data(*edge)
            if data is None:
                if len(edge) == 3 and edge[2] is not None:
                    data = graph.get_edge_data(edge[0], edge[1], int(edge[2]))
                else:
                    data = graph.get_edge_data(edge[0], edge[1])

            if data is None:
                data = {}

            if data.get('style', None) == 'invis':
                continue

            if data.get('implicit', False):
                #alpha = .2
                alpha = .5
                defaultcolor = pt.GREEN[0:3]
            else:
                #alpha = 0.5
                alpha = 1.0
                defaultcolor = pt.BLACK[0:3]
            color = data.get('color', defaultcolor)
            if color is None:
                color = defaultcolor
            color = color[0:3]

            offset = 1 if graph.is_directed() else 0
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
                CODE = mpl.path.Path.CURVE3
            elif splines == 'spline':
                CODE = mpl.path.Path.CURVE4
            else:
                raise AssertionError('splines = %r' % (splines,))

            #if offset == 1:
            #    astart_code = LINETO
            #else:
            astart_code = MOVETO

            verts = [start_point] + other_points
            codes = [astart_code] + [CODE] * len(other_points)

            #if offset == 1:
            #    verts = [start_point[0]] + verts
            #    codes = [MOVETO] + codes
            #print('verts = %r' % (verts,))
            #print('codes = %r' % (codes,))

            # HACK THE ENDPOINTS TO TOUCH THE BOUNDING BOXES
            if not as_directed:
                if edge_endpoints is not None:
                    endpoint = edge_endpoints.get(edge, None)
                    if endpoint is not None:
                        #print('endpoint = %r' % (endpoint,))
                        verts += [endpoint]
                        codes += [LINETO]
                        #endpoint = np.array(other_points[-1])
                        #xy2 = node_pos[edge[1]]
                        #wh2 = _get_node_size(graph, edge[0], node_size)
                        #bbox2 = vt.bbox_from_xywh(xy2, wh2, [.5, .5])
                        #close_point2 = vt.closest_point_on_bbox(endpoint, bbox2)
                        #verts += [close_point2]
                        #codes += [LINETO]
                        #print('verts = %r' % (verts,))
                        #print('codes = %r' % (codes,))
                        #print('close_point2 = %r' % (close_point2,))
                        #print('endpoint = %r' % (endpoint,))
                        #print('#other_points = %r' % (#other_points,))

            path = mpl.path.Path(verts, codes)
            #lw = 5
            # python -m ibeis.annotmatch_funcs review_tagged_joins --dpath ~/latex/crall-candidacy-2015/ --save figures4/mergecase.png --figsize=15,15 --clipwhite --diskshow
            # python -m dtool --tf DependencyCache.make_graph --show

            figsize = ut.get_argval('--figsize', type_=list, default=None)
            if figsize is not None:
                # HACK
                graphsize = max(figsize)
                lw = graphsize / 5
                width =  graphsize / 15
            else:
                width = .5
                lw = 1.0
            patch = mpl.patches.PathPatch(path, facecolor='none', lw=lw,
                                          edgecolor=color,
                                          alpha=alpha,
                                          joinstyle='bevel')
            if as_directed:
                dxy = (np.array(other_points[-1]) - other_points[-2])
                dxy = (dxy / np.sqrt(np.sum(dxy ** 2))) * .1
                dx, dy = dxy
                rx, ry = other_points[-1][0], other_points[-1][1]
                patch1 = mpl.patches.FancyArrow(rx, ry, dx, dy, width=width,
                                                length_includes_head=True,
                                                color=color,
                                                head_starts_at_zero=True)
                ax.add_patch(patch1)

            # endpoint1 = edge_verts[0]
            # endpoint2 = edge_verts[len(edge_verts) // 2 - 1]
            u, v = edge[0:2]
            n1 = patches[u]
            n2 = patches[v]
            if (data.get('ismulti', False) or data.get('isnwise', False) or
                 data.get('local_input_id', False)):
                pt1 = np.array(n1.center)
                pt2 = np.array(n2.center)
                frac_thru = 4
                edge_verts = path.vertices
                edge_verts = vt.unique_rows(edge_verts)
                sorted_verts = edge_verts[vt.L2(edge_verts, pt1).argsort()]
                if len(sorted_verts) <= 4:
                    mpl_bbox = path.get_extents()
                    bbox = [mpl_bbox.x0, mpl_bbox.y0, mpl_bbox.width, mpl_bbox.height]
                    endpoint1 = vt.closest_point_on_bbox(pt1, bbox)
                    endpoint2 = vt.closest_point_on_bbox(pt2, bbox)
                    beta = (1 / frac_thru)
                    alpha = 1 - beta
                    text_point1 = (alpha * endpoint1) + (beta * endpoint2)
                else:
                    #print('sorted_verts = %r' % (sorted_verts,))
                    #text_point1 = sorted_verts[len(sorted_verts) // (frac_thru)]
                    #frac_thru = 3
                    frac_thru = 6

                    text_point1 = edge_verts[(len(edge_verts) - 2) // (frac_thru) + 1]

                font_prop = mpl.font_manager.FontProperties(family='monospace',
                                                            weight='light',
                                                            size=14)
                if data.get('local_input_id', False):
                    text = data['local_input_id']
                    if text == '1':
                        text = ''
                elif data.get('ismulti', False):
                    text = '*'
                else:
                    text = str(data.get('nwise_idx', '!'))
                ax.annotate(text, xy=text_point1, xycoords='data', va='center',
                            ha='center', fontproperties=font_prop)
                #bbox=dict(boxstyle='round', fc=None, alpha=1.0))
            if data.get('label', False):
                pt1 = np.array(n1.center)
                pt2 = np.array(n2.center)
                frac_thru = 2
                edge_verts = path.vertices
                edge_verts = vt.unique_rows(edge_verts)
                sorted_verts = edge_verts[vt.L2(edge_verts, pt1).argsort()]
                if len(sorted_verts) <= 4:
                    mpl_bbox = path.get_extents()
                    bbox = [mpl_bbox.x0, mpl_bbox.y0, mpl_bbox.width, mpl_bbox.height]
                    endpoint1 = vt.closest_point_on_bbox(pt1, bbox)
                    endpoint2 = vt.closest_point_on_bbox(pt2, bbox)
                    print('sorted_verts = %r' % (sorted_verts,))
                    beta = (1 / frac_thru)
                    alpha = 1 - beta
                    text_point1 = (alpha * endpoint1) + (beta * endpoint2)
                else:
                    text_point1 = sorted_verts[len(sorted_verts) // (frac_thru)]
                    ax.annotate(data['label'], xy=text_point1, xycoords='data',
                                va='center', ha='center',
                                bbox=dict(boxstyle='round', fc='w'))
            #patch = mpl.patches.PathPatch(path, facecolor='none', lw=1)
            ax.add_patch(patch)

    use_collections = False
    if use_collections:
        edge_coll = mpl.collections.PatchCollection(edge_patch_list)
        node_coll = mpl.collections.PatchCollection(node_patch_list)
        #coll.set_facecolor(fcolor)
        #coll.set_alpha(alpha)
        #coll.set_linewidth(lw)
        #coll.set_edgecolor(color)
        #coll.set_transform(ax.transData)
        ax.add_collection(node_coll)
        ax.add_collection(edge_coll)
    else:
        if not hacknonode:
            for patch in node_patch_list:
                ax.add_patch(patch)
        if not hacknoedge:
            for patch in edge_patch_list:
                ax.add_patch(patch)


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
