from __future__ import absolute_import, division, print_function
from six.moves import zip
import numpy as np
import matplotlib as mpl
import utool as ut
import vtool as vt
import six

try:
    import cv2
except ImportError as ex:
    print('ERROR PLOTTOOL CANNOT IMPORT CV2')
    print(ex)
(print, rrr, profile) = ut.inject2(__name__, '[df2]')


class GraphVizConfig(object):
    r"""
    Ignore:
        # Parse param definitions;
        text = ut.readfrom(ut.truepath('~/rawhtml.txt'))
        import re
        subrepl_list = [
            ('<TR>', r'\n\n'),
            ('<\/TR>', r'\n\n'),
            ('<A HREF=.*?>', r''),
            ('<A NAME=.*?>', r''),
            ('<\/A>', ''),
            (ut.negative_lookbehind('\n') + '\n' + ut.negative_lookahead('\n'), ''),
            ('<TD ALIGN="CENTER">', r'<TD>'),
            (' *, *', r' | '),
            ('<\/TD>\n*<TD>', ', '),
            ('<TD>\n*<\TD>', ', '),
            ('^<TD>', ''),
            ('<\/TD>', ''),
            ('<BR>', ''),
            ('<BR>', ''),
            ('<TABLE ALIGN=CENTER>', ''),
            ('<TH.*\n', ''),
            ('</TABLE>', ''),
        ]
        new_text = text
        for sub, repl in subrepl_list:
            new_text = re.sub(sub, repl, new_text, flags=re.MULTILINE)
        new_text = new_text.replace('&#60;', '<')
        new_text = new_text.replace('&#62;', '>')
        new_text = ut.remove_doublenewlines(new_text)
        print(new_text)
        row_list = [r.split(',') for r in new_text.split('\n') if r]
        row_list = [[c.strip() for c in r] for r in row_list]

        column_lbls = ['Name', 'Used By', 'Type', 'Default', 'Minimum', 'Notes']
        column_list = ut.listT(row_list)

        csv_str = (ut.make_standard_csv(column_list, column_lbls))
        ut.writeto('tmp.csv', csv_str)
        df = pandas.read_csv('tmp.csv')

        usedby = [''.join(sorted(tags)) for tags in df['Used By'].tolist()]
        df['Used By'][:] = usedby

        minim = ['-' if tags is np.nan else tags for tags in df['Minimum'].tolist()]
        df['Minimum'][:] = minim
        print(df.to_string())

        notes = ['' if tags is np.nan else tags.replace(' only', '').strip() for tags in df['Notes'].tolist()]
        unique_tags = ut.unique(ut.flatten([t.strip().split('|') for t in notes]))
        unique_tags = ut.unique([t.strip().replace('not ', '') for t in unique_tags])
        valid_progs = ut.unique([t for t in unique_tags if t])
        #df['Notes'][:] = notes
        #print(df.to_string())

        flags = ['N' in tags for tags in df['Used By'].tolist()]
        sortx = ut.argsort(flags)[::-1]
        idx = ut.where(flags)
        df_nodes = df.take(idx)

        progs = ['dot', 'neato', 'svg', 'postscript', 'map', 'patchwork', 'write']

        def expand_tags(tag):
            extag = [t.strip() for t in tag.split('|')]
            if len(''.join(extag)) == 0:
                extag = valid_progs
            extag2 = []
            remove_tags = []
            for t in extag:
                if t.startswith('not '):
                    remove = t[4:]
                    remove_tags.append(remove)
                else:
                    extag2.append(t)
            extag2.extend(ut.setdiff(valid_progs, remove_tags))
            return extag

        def usedby(df, opts):
            notes = df['Notes'].tolist()
            notes = ['' if t is np.nan else t for t in notes]
            notes = [t.replace(' only', '').strip() for t in notes]
            extags = [expand_tags(t) for t in notes]
            flag_list = [[prog in exts for exts in extags] for prog in opts]
            flags = ut.or_lists(*flag_list)
            hasany_idx = ut.where(flags)
            subdf = df.take(hasany_idx)
            return subdf

        def print_useful(df):
            print(df.sort(['Used By', 'Notes', 'Name']).to_string())

        # In the Used By field, the characters E, N, G, S and C represent edges,
        # nodes, the root graph, subgraphs and cluster subgraphs, respectively.
        #T his field indicates which graph component uses the attribute.

        # At present, most device-independent units are either inches or points,
        # which we take as 72 points per inch.


        df_nd = usedby(df, ['neato', 'dot'])
        print_useful(df_nd)

        dotneato_nodes = df_nodes.take(idx2)

        dotneato_nodes = df.take(idx2)

        print(df_.to_string())
        print(dotneato_nodes.sort('Used By').to_string())

        Node Props:
            colorscheme    CEGN           string                                       ""              NaN
              fontcolor    CEGN            color                                    black              NaN
               fontname    CEGN           string                            "Times-Roman"              NaN
               fontsize    CEGN           double                                     14.0              NaN
                  label    CEGN        lblString           "&#92;N" (nodes)"" (otherwise)              NaN
              nojustify    CEGN             bool                                    false              NaN
                  style    CEGN            style                                       ""              NaN
                  color     CEN   colorcolorList                                    black              NaN
              fillcolor     CEN   colorcolorList          lightgrey(nodes)black(clusters)              NaN
                  layer     CEN       layerRange                                       ""              NaN
               penwidth     CEN           double                                      1.0              NaN
           radientangle     CGN              int                                       ""              NaN
               labelloc     CGN           string  "t"(clusters)"b"(root graphs)"c"(nodes)              NaN
                 margin     CGN      doublepoint                       <device-dependent>              NaN
                  sortv     CGN              int                                        0              NaN
            peripheries      CN              int          shape default(nodes)1(clusters)              NaN
              showboxes     EGN              int                                        0         dot only
                comment     EGN           string                                       ""              NaN
                    pos      EN  pointsplineType                                      NaN              NaN
                 xlabel      EN        lblString                                       ""              NaN
               ordering      GN           string                                       ""         dot only
                  group       N           string                                       ""         dot only
                    pin       N             bool                                    false fdp | neato only
             distortion       N           double                                      0.0              NaN
              fixedsize       N       boolstring                                    false              NaN
                 height       N           double                                      0.5              NaN
                  image       N           string                                       ""              NaN
             imagescale       N       boolstring                                    false              NaN
            orientation       N           double                                      0.0              NaN
                regular       N             bool                                    false              NaN
           samplepoints       N              int      8(output)20(overlap and image maps)              NaN
                  shape       N            shape                                  ellipse              NaN
              shapefile       N           string                                       ""              NaN
                  sides       N              int                                        4              NaN
                   skew       N           double                                      0.0              NaN
                  width       N           double                                     0.75              NaN
                      z       N           double                                      0.0              NaN
    """
    # TODO: make a gridsearchable config for layouts
    def get_param_info_list(self):
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
    if node_pos is None:
        layout_dict = get_nx_layout(graph, layout, layoutkw=layoutkw)
        node_pos = layout_dict['node_pos']
        edge_pos = layout_dict['edge_pos']

    # zoom = kwargs.pop('zoom', .4)
    node_size = layout_dict['node_size']
    frameon = kwargs.pop('frameon', True)
    splines = layout_dict['splines']
    draw_network2(graph, node_pos, ax, edge_pos=edge_pos, splines=splines,
                  node_size=node_size, **kwargs)
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
    }

    if img_dict is not None and len(img_dict) > 0:
        node_list = img_dict.keys()
        pos_list = ut.take(node_pos, node_list)
        img_list = ut.take(img_dict, node_list)
        size_list = ut.take(node_size, node_list)
        #node_attrs = ut.dict_take(graph.node, node_list)
        # Rename to node_scale?
        #scale_list = np.array(ut.dict_take_column(node_attrs, 'scale',
        #                                          default=None))
        #img_list = [img if scale is None else vt.resize_image_by_scale(img, scale)
        #            for scale, img in zip(scale_list, img_list)]
        imgdat = pt.netx_draw_images_at_positions(img_list, pos_list, size_list, frameon=frameon)
        plotinfo['imgdat'] = imgdat

    if title is not None:
        pt.set_title(title)
    return plotinfo


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

    #print('layout = %r' % (layout,))
    if layout == 'agraph':
        # PREFERED LAYOUT WITH MOST CONTROL
        _, layout_info = nx_agraph_layout(layout_graph, **layoutkw)
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
    return layout_dict


def nx_agraph_layout(graph, inplace=False, **kwargs):
    r"""
    References:
        http://www.graphviz.org/doc/info/attrs.html

    """
    import networkx as nx
    import pygraphviz

    if not inplace:
        graph = graph.copy()

    node_pos = {}
    node_size = {}
    edge_pos = {}

    kwargs = kwargs.copy()
    factor = kwargs.pop('factor', 1.0)
    prog = kwargs.pop('prog', 'dot')

    if True:
        #kwargs['ratio'] = 'compress'
        #kwargs['ratio'] = .7
        #kwargs['size'] = '10,2'
        #kwargs['pagedir'] = 'LB'
        #kwargs['landscape'] = 'true'
        #kwargs['imagescale'] = kwargs.get('imagescale', 'true')
        #kwargs['fixedsize'] = kwargs.get('fixedsize', 'true')
        kwargs['splines'] = kwargs.get('splines', 'spline')
        #kwargs['splines'] = kwargs.get('splines', 'polyline')
        kwargs['pack'] = kwargs.get('pack', 'true')
        kwargs['packmode'] = kwargs.get('packmode', 'cluster')
    if prog == 'dot':
        #kwargs['ranksep'] = kwargs.get('ranksep', 2.5 * factor)
        #kwargs['nodesep'] = kwargs.get('nodesep', 2 * factor)
        kwargs['ranksep'] = kwargs.get('ranksep', 1.5 * factor)
        #kwargs['rankdir'] = kwargs.get('rankdir', 'LR')
        kwargs['nodesep'] = kwargs.get('nodesep', 1 * factor)
        kwargs['clusterrank'] = kwargs.get('clusterrank', 'local')
    if prog != 'dot':
        #kwargs['overlap_scaling'] = .01
        #kwargs['overlap'] = kwargs.get('overlap', 'false')
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
    # print(agraph)
    agraph.layout(prog=prog, args=args)
    agraph.draw('test_graphviz_draw.png')
    # print('AFTER LAYOUT')
    # print(agraph)
    #ratio_scale = 2.0
    ratio_scale = 1.0

    for node in graph_.nodes():
        anode = pygraphviz.Node(agraph, node)
        try:
            xx, yy = anode.attr['pos'].split(',')
            xy = np.array((float(xx), float(yy))) / factor
            xy[0] *= ratio_scale
            node_pos[node] = xy
        except:
            node_pos[node] = (0.0, 0.0)
        height = float(anode.attr['height']) * 72.0 / factor
        width = float(anode.attr['width']) * 72.0 / factor
        node_size[node] = width, height

    for edge in graph_.edges():
        u, v = edge[0:2]
        aedge = pygraphviz.Edge(agraph, u, v)
        #apos = aedge.attr['pos'][2:]
        apos = aedge.attr['pos']
        strpos_list = apos.split(' ')
        strtup_list = [ea.split(',') for ea in strpos_list]
        try:
            # FIXME: not sure I'm parsing this correctly
            edge_ctrlpts = [tuple([float(f) for f in ea if f not in 'es'])
                            for ea in strtup_list]
            edge_ctrlpts = np.array(edge_ctrlpts)
            edge_ctrlpts /= factor
            edge_ctrlpts[:, 0] *= ratio_scale
            edge_pos[edge] = edge_ctrlpts
        except Exception:
            raise

    layout_info = dict(
        node_pos=node_pos,
        splines=splines,
        edge_pos=edge_pos,
        node_size=node_size,
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
                  as_directed=None, edge_pos=None, node_size=None, use_arc=True):
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
        label = nattrs.get('label', None)
        alpha = nattrs.get('alpha', .5)
        node_color = nattrs.get('color', pt.NEUTRAL_BLUE)[0:3]
        xy = node_pos[node]
        if 'image' in nattrs:
            alpha_ = 0.0
        else:
            alpha_ = alpha
        patch_kw = dict(alpha=alpha_, color=node_color)
        node_shape = nattrs.get('shape', 'circle')
        if node_shape == 'circle':
            radius = _get_node_size(graph, node, node_size)[0]
            patch = mpl.patches.Circle(xy, radius=radius, **patch_kw)
        elif node_shape in ['rect', 'rhombus']:
            width, height = _get_node_size(graph, node, node_size)
            angle = 0 if node_shape == 'rect' else 45
            xy_bl = (xy[0] - width // 2, xy[1] - height // 2)
            patch = mpl.patches.Rectangle(
                xy_bl, width, height, angle=angle, **patch_kw)
            patch.center = xy
        patches[node] = patch
        x, y = node_pos[node]
        text = node
        if label is not None:
            text += ': ' + str(label)
        if not hacknonode and 'image' not in nattrs:
            pt.ax_absolute_text(x, y, text, ha='center', va='center')
        node_patch_list.append(patch)
    ###
    # Draw Edges
    if edge_pos is None:
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
    elif edge_pos is not None:
        # NEW WAY OF DRAWING EDGEES
        if as_directed is None:
            as_directed = graph.is_directed()
        for edge, pts in edge_pos.items():
            data = graph.get_edge_data(*edge)
            color = data.get('color', pt.BLACK)[0:3]

            offset = 1 if graph.is_directed() else 0
            #color = data.get('color', color)[0:3]
            start_point = pts[offset]
            other_points = pts[offset + 1:].tolist()  # [0:3]
            end_point = np.array(other_points[-1])
            verts = [start_point] + other_points

            xy1 = node_pos[edge[0]]
            xy2 = node_pos[edge[1]]
            wh1 = _get_node_size(graph, edge[0], node_size)
            wh2 = _get_node_size(graph, edge[0], node_size)

            bbox1 = vt.bbox_from_xywh(xy1, wh1, [.5, .5])
            bbox2 = vt.bbox_from_xywh(xy2, wh2, [.5, .5])

            #bbox1_verts = np.array(vt.verts_from_bbox(bbox1, close=True))
            #pt.plt.plot(bbox1_verts.T[0], bbox1_verts.T[1], 'b-')
            #bbox2_verts = np.array(vt.verts_from_bbox(bbox2, close=True))
            #pt.plt.plot(bbox2_verts.T[0], bbox2_verts.T[1], 'b-')

            close_point1 = vt.closest_point_on_bbox(start_point, bbox1)
            close_point2 = vt.closest_point_on_bbox(end_point, bbox2)
            #print('edge = %r' % (edge,))
            #print('pts = %r' % (pts,))
            #print('close_point1 = %r' % (close_point1,))
            #print('close_point2 = %r' % (close_point2,))

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

            #print('CODE = %r' % (CODE,))
            force_touch_bbox = False
            if force_touch_bbox:
                astart_code = LINETO
            else:
                astart_code = MOVETO

            # Force edge to touch node.
            #pt.plt.plot(close_point1[0], close_point1[1], 'go')
            #pt.plt.plot(close_point2[0], close_point2[1], 'gx')
            #pt.plt.plot(start_point[0], start_point[1], 'rx')
            #pt.plt.plot(other_points[0][0], other_points[0][1], 'b-x')

            verts = [start_point] + other_points
            codes = [astart_code] + [CODE] * len(other_points)
            if force_touch_bbox:
                verts = [close_point1] + verts
                codes = [MOVETO] + codes
                if not as_directed:
                    verts = verts + [close_point2]
                    codes = codes + [LINETO]
            #verts = [start_point] + other_points
            #codes = [MOVETO] + [LINETO] * len(other_points)

            path = mpl.path.Path(verts, codes)
            patch = mpl.patches.PathPatch(path, facecolor='none', lw=5,
                                          edgecolor=color,
                                          joinstyle='bevel')
            if as_directed:
                dxy = (np.array(other_points[-1]) - other_points[-2])
                dxy = (dxy / np.sqrt(np.sum(dxy ** 2))) * .1
                dx, dy = dxy
                rx, ry = other_points[-1][0], other_points[-1][1]
                patch1 = mpl.patches.FancyArrow(rx, ry, dx, dy, width=.9,
                                                length_includes_head=True,
                                                color=color,
                                                head_starts_at_zero=True)
                ax.add_patch(patch1)
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


def netx_draw_images_at_positions(img_list, pos_list, node_size, frameon=True):
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
    # from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    print('[viz_graph] drawing %d images' % len(img_list))
    # Thumb stackartist
    import plottool as pt
    ax  = pt.gca()
    artist_list = []
    offset_img_list = []

    # bboxkw = dict(
    #     xycoords='data',
    #     boxcoords='offset points',
    #     #boxcoords='data',
    #     pad=0.25, frameon=frameon,
    #     # frameon=False, bboxprops=dict(fc="cyan"),
    #     # arrowprops=dict(arrowstyle="->"))
    # )
    for pos, img, size in zip(pos_list, img_list, node_size):
        x, y = pos
        if isinstance(img, six.string_types):
            img = cv2.cvtColor(vt.imread(img), cv2.COLOR_BGR2RGB)
        if size is not None:
            width, height = size
        else:
            width, height = vt.get_size(img)
        #print('height = %r' % (height,))
        #print('width = %r' % (width,))
        if False:
            # THIS DOES NOT DO WHAT I WANT
            # Scales the image with data coords
            # offset_img = OffsetImage(img, zoom=zoom)
            # artist = AnnotationBbox(offset_img, (x, y), xybox=(-0., 0.), **bboxkw)
            # offset_img_list.append(offset_img)
            # artist_list.append(artist)
            pass

        #offset_img = None

        # THIS DOES EXACTLY WHAT I WANT
        # Ties the image to data coords
        pt.plt.imshow(img, extent=[x - width // 2, x + width // 2,
                                   y - height // 2, y + height // 2])
        #, aspect='auto')

    for artist in artist_list:
        ax.add_artist(artist)

    imgdat = {
        'offset_img_list': offset_img_list,
        'artist_list': artist_list,
    }
    return imgdat


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
