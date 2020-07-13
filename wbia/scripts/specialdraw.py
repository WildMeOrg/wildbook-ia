# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import utool as ut
import numpy as np

(print, rrr, profile) = ut.inject2(__name__)


def nx_makenode(graph, name, **attrkw):
    if 'size' in attrkw:
        attrkw['width'], attrkw['height'] = attrkw.pop('size')
    graph.add_node(name, **attrkw)
    return name


def multidb_montage():
    r"""
    CommandLine:
        python -m wbia.scripts.specialdraw multidb_montage --save montage.jpg --dpath ~/slides --diskshow --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.scripts.specialdraw import *  # NOQA
        >>> multidb_montage()
    """
    import wbia
    import wbia.plottool as pt
    import vtool as vt
    import numpy as np

    pt.ensureqt()
    dbnames = [
        'PZ_Master1',
        'GZ_Master1',
        'humpbacks_fb',
        'GIRM_Master1',
    ]
    ibs_list = [wbia.opendb(dbname) for dbname in dbnames]

    target_num = 1000
    sample_size = target_num // len(ibs_list)

    aids_list = []
    for ibs in ibs_list:
        aids = ibs.sample_annots_general(minqual='good', sample_size=sample_size)
        aids_list.append(aids)

    print(ut.depth_profile(aids_list))

    chip_lists = []
    for ibs, aids in zip(ibs_list, aids_list):
        annots = ibs.annots(aids)
        chip_lists.append(annots.chips)

    chips = ut.flatten(chip_lists)
    np.random.shuffle(chips)

    widescreen_ratio = 16 / 9
    ratio = ut.PHI
    ratio = widescreen_ratio

    fpath = ut.get_argval('--save', type_=str, default='montage.jpg')

    # height = 6000
    width = 6000
    # width = int(height * ratio)
    height = int(width / ratio)
    dsize = (width, height)
    dst = vt.montage(chips, dsize)
    vt.imwrite(fpath, dst)
    if ut.get_argflag('--show'):
        pt.imshow(dst)


def featweight_fig():
    r"""
    CommandLine:
        python -m wbia.scripts.specialdraw featweight_fig --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.scripts.specialdraw import *  # NOQA
        >>> featweight_fig()
        >>> ut.show_if_requested()
    """
    # ENABLE_DOCTEST
    import wbia

    # import wbia.plottool as pt
    import matplotlib as mpl
    from wbia.scripts.thesis import TMP_RC

    mpl.rcParams.update(TMP_RC)
    from wbia.core_annots import gen_featweight_worker

    # test_featweight_worker()

    # ibs = wbia.opendb(defaultdb='GZ_Master1')
    # aid = ut.get_argval('--aid', type_=list, default=2810)
    ibs = wbia.opendb(defaultdb='PZ_MTEST')
    aid = ut.get_argval('--aid', type_=int, default=1)
    depc = ibs.depc
    aids = [aid]

    assert all(ibs.db.rows_exist('annotations', aids))

    config = {
        'dim_size': 450,
        'resize_dim': 'area',
        'smooth_thresh': 30,
        'smooth_ksize': 30,
    }
    probchip = depc.get('probchip', aids, 'img', config=config, recompute=True)[0]
    chipsize = depc.get('chips', aids, ('width', 'height'), config=config)[0]
    kpts = depc.get('feat', aids, 'kpts', config=config)[0]
    tup = (kpts, probchip, chipsize)
    weights = gen_featweight_worker(tup)
    assert np.all(weights <= 1.0), 'weights cannot be greater than 1'
    chip = depc.get('chips', aids, 'img', config=config)[0]
    ut.quit_if_noshow()
    import wbia.plottool as pt

    fnum = 1
    pnum_ = pt.make_pnum_nextgen(1, 3)
    pt.figure(fnum=fnum, doclf=True)
    pt.imshow(chip, pnum=pnum_(0), fnum=fnum)
    pt.imshow(probchip, pnum=pnum_(2), fnum=fnum)
    pt.imshow(chip, pnum=pnum_(1), fnum=fnum)
    color_list = pt.draw_kpts2(kpts, weights=weights, ell_alpha=0.3)
    color_list
    # cb = pt.colorbar(weights, color_list)
    # cb.set_label('featweights')


def simple_vsone_matches():
    r"""
    CommandLine:
        python -m wbia.scripts.specialdraw simple_vsone_matches --show \
            --db GZ_Master1 --aids=2811,2810

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.scripts.specialdraw import *  # NOQA
        >>> simple_vsone_matches()
        >>> ut.show_if_requested()
    """
    import wbia

    # import wbia.plottool as pt
    import matplotlib as mpl
    from wbia.scripts.thesis import TMP_RC

    mpl.rcParams.update(TMP_RC)

    ibs = wbia.opendb(defaultdb='GZ_Master1')
    aids = ut.get_argval('--aids', type_=list, default=[2811, 2810])
    assert len(aids) == 2
    assert all(ibs.db.rows_exist('annotations', aids))
    aid1, aid2 = aids

    infr = wbia.AnnotInference(ibs=ibs, aids=aids)
    edges = [(aid1, aid2)]
    match = infr._exec_pairwise_match(edges)[0]

    ut.quit_if_noshow()
    import wbia.plottool as pt

    pt.figure(fnum=1, doclf=True)
    match.show(
        heatmask=True,
        vert=False,
        modifysize=True,
        show_ell=False,
        show_lines=False,
        show_ori=False,
    )


def double_depcache_graph():
    r"""
    CommandLine:
        python -m wbia.scripts.specialdraw double_depcache_graph --show --testmode

        python -m wbia.scripts.specialdraw double_depcache_graph --save=figures5/doubledepc.png --dpath ~/latex/cand/  --diskshow  --figsize=8,20 --dpi=220 --testmode --show --clipwhite
        python -m wbia.scripts.specialdraw double_depcache_graph --save=figures5/doubledepc.png --dpath ~/latex/cand/  --diskshow  --figsize=8,20 --dpi=220 --testmode --show --clipwhite --arrow-width=.5

        python -m wbia.scripts.specialdraw double_depcache_graph --save=figures5/doubledepc.png --dpath ~/latex/cand/  --diskshow  --figsize=8,20 --dpi=220 --testmode --show --clipwhite --arrow-width=5

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.scripts.specialdraw import *  # NOQA
        >>> result = double_depcache_graph()
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> ut.show_if_requested()
    """
    import wbia
    import networkx as nx
    import wbia.plottool as pt

    pt.ensureqt()
    # pt.plt.xkcd()
    ibs = wbia.opendb('testdb1')
    reduced = True
    implicit = True
    annot_graph = ibs.depc_annot.make_graph(reduced=reduced, implicit=implicit)
    image_graph = ibs.depc_image.make_graph(reduced=reduced, implicit=implicit)
    to_rename = ut.isect(image_graph.nodes(), annot_graph.nodes())
    nx.relabel_nodes(annot_graph, {x: 'annot_' + x for x in to_rename}, copy=False)
    nx.relabel_nodes(image_graph, {x: 'image_' + x for x in to_rename}, copy=False)
    graph = nx.compose_all([image_graph, annot_graph])
    # graph = nx.union_all([image_graph, annot_graph], rename=('image', 'annot'))
    # userdecision = nx_makenode(graph, 'user decision', shape='rect', color=pt.DARK_YELLOW, style='diagonals')
    # userdecision = nx_makenode(graph, 'user decision', shape='circle', color=pt.DARK_YELLOW)
    userdecision = nx_makenode(
        graph,
        'User decision',
        shape='rect',
        # width=100, height=100,
        color=pt.YELLOW,
        style='diagonals',
    )
    # longcat = True
    longcat = False

    # edge = ('feat', 'neighbor_index')
    # data = graph.get_edge_data(*edge)[0]
    # print('data = %r' % (data,))
    # graph.remove_edge(*edge)
    # # hack
    # graph.add_edge('featweight', 'neighbor_index', **data)

    graph.add_edge('detections', userdecision, constraint=longcat, color=pt.PINK)
    graph.add_edge(userdecision, 'annotations', constraint=longcat, color=pt.PINK)
    # graph.add_edge(userdecision, 'annotations', implicit=True, color=[0, 0, 0])
    if not longcat:
        pass
        # graph.add_edge('images', 'annotations', style='invis')
        # graph.add_edge('thumbnails', 'annotations', style='invis')
        # graph.add_edge('thumbnails', userdecision, style='invis')
    graph.remove_node('Has_Notch')
    graph.remove_node('annotmask')
    layoutkw = {
        'ranksep': 5,
        'nodesep': 5,
        'dpi': 96,
        # 'nodesep': 1,
    }
    ns = 1000

    ut.nx_set_default_node_attributes(graph, 'fontsize', 72)
    ut.nx_set_default_node_attributes(graph, 'fontname', 'Ubuntu')
    ut.nx_set_default_node_attributes(graph, 'style', 'filled')

    ut.nx_set_default_node_attributes(graph, 'width', ns * ut.PHI)
    ut.nx_set_default_node_attributes(graph, 'height', ns * (1 / ut.PHI))

    # for u, v, d in graph.edge(data=True):
    for u, vkd in graph.edge.items():
        for v, dk in vkd.items():
            for k, d in dk.items():
                localid = d.get('local_input_id')
                if localid:
                    # d['headlabel'] = localid
                    if localid not in ['1']:
                        d['taillabel'] = localid
                    # d['label'] = localid
                if d.get('taillabel') in {'1'}:
                    del d['taillabel']

    node_alias = {
        'chips': 'Chip',
        'images': 'Image',
        'feat': 'Feat',
        'featweight': 'Feat Weights',
        'thumbnails': 'Thumbnail',
        'detections': 'Detections',
        'annotations': 'Annotation',
        'Notch_Tips': 'Notch Tips',
        'probchip': 'Prob Chip',
        'Cropped_Chips': 'Croped Chip',
        'Trailing_Edge': 'Trailing\nEdge',
        'Block_Curvature': 'Block\nCurvature',
        # 'BC_DTW': 'block curvature /\n dynamic time warp',
        'BC_DTW': 'DTW Distance',
        'CurvRank': 'CurvRank Distance',
        'CurvRankDorsal': 'CurvRank (Dorsal) Distance',
        'CurvRankFinfindrHybridDorsal': 'CurvRank + FinFindR Hybrid (Dorsal) Distance',
        'CurvRankFluke': 'CurvRank (Fluke) Distance',
        'Deepsense': 'Deepsense Distance',
        'Finfindr': 'Finfindr Distance',
        'Kaggle7': 'Kaggle7 Distance',
        'KaggleSeven': 'Kaggle7 Distance',
        'vsone': 'Hots vsone',
        'feat_neighbs': 'Nearest\nNeighbors',
        'neighbor_index': 'Neighbor\nIndex',
        'vsmany': 'Hots vsmany',
        'annot_labeler': 'Annot Labeler',
        'labeler': 'Labeler',
        'localizations': 'Localizations',
        'classifier': 'Classifier',
        'sver': 'Spatial\nVerification',
        'Classifier': 'Existence',
        'image_labeler': 'Image Labeler',
    }
    node_alias = {
        'Classifier': 'existence',
        'feat_neighbs': 'neighbors',
        'sver': 'spatial_verification',
        'Cropped_Chips': 'cropped_chip',
        'BC_DTW': 'dtw_distance',
        'CurvRank': 'curvrank_distance',
        'CurvRankDorsal': 'curvrank_distance_dorsal',
        'CurvRankFinfindrHybridDorsal': 'curvrank_finfindr_hybrid_distance_dorsal',
        'CurvRankFluke': 'curvrank_distance_fluke',
        'Deepsense': 'deepsense_distance',
        'Finfindr': 'finfindr_distance',
        'Kaggle7': 'kaggle7_distance',
        'KaggleSeven': 'kaggle7_distance',
        'Block_Curvature': 'curvature',
        'Trailing_Edge': 'trailing_edge',
        'Notch_Tips': 'notch_tips',
        'thumbnails': 'thumbnail',
        'images': 'image',
        'annotations': 'annotation',
        'chips': 'chip',
        # userdecision: 'User de'
    }
    node_alias = ut.delete_dict_keys(
        node_alias, ut.setdiff(node_alias.keys(), graph.nodes())
    )
    nx.relabel_nodes(graph, node_alias, copy=False)

    fontkw = dict(fontname='Ubuntu', fontweight='normal', fontsize=12)
    # pt.gca().set_aspect('equal')
    # pt.figure()
    pt.show_nx(graph, layoutkw=layoutkw, fontkw=fontkw)
    pt.zoom_factory()


def lighten_hex(hexcolor, amount):
    import wbia.plottool as pt
    import matplotlib.colors as colors

    return pt.color_funcs.lighten_rgb(colors.hex2color(hexcolor), amount)


def general_identify_flow():
    r"""
    CommandLine:
        python -m wbia.scripts.specialdraw general_identify_flow --show --save pairsim.png --dpi=100 --diskshow --clipwhite

        python -m wbia.scripts.specialdraw general_identify_flow --dpi=200 --diskshow --clipwhite --dpath ~/latex/cand/ --figsize=20,10  --save figures4/pairprob.png --arrow-width=2.0


    Example:
        >>> # SCRIPT
        >>> from wbia.scripts.specialdraw import *  # NOQA
        >>> general_identify_flow()
        >>> ut.quit_if_noshow()
        >>> ut.show_if_requested()
    """
    import networkx as nx
    import wbia.plottool as pt

    pt.ensureqt()
    # pt.plt.xkcd()

    graph = nx.DiGraph()

    def makecluster(name, num, **attrkw):
        return [nx_makenode(name + str(n), **attrkw) for n in range(num)]

    def add_edge2(u, v, *args, **kwargs):
        v = ut.ensure_iterable(v)
        u = ut.ensure_iterable(u)
        for _u, _v in ut.product(u, v):
            graph.add_edge(_u, _v, *args, **kwargs)

    # *** Primary color:
    p_shade2 = '#41629A'
    # *** Secondary color
    s1_shade2 = '#E88B53'
    # *** Secondary color
    s2_shade2 = '#36977F'
    # *** Complement color
    c_shade2 = '#E8B353'

    ns = 512

    ut.inject_func_as_method(graph, nx_makenode)

    annot1_color = p_shade2
    annot2_color = s1_shade2
    # annot1_color2 = pt.color_funcs.lighten_rgb(colors.hex2color(annot1_color), .01)

    annot1 = graph.nx_makenode(
        'Annotation X', width=ns, height=ns, groupid='annot', color=annot1_color
    )
    annot2 = graph.nx_makenode(
        'Annotation Y', width=ns, height=ns, groupid='annot', color=annot2_color
    )

    featX = graph.nx_makenode(
        'Features X',
        size=(ns / 1.2, ns / 2),
        groupid='feats',
        color=lighten_hex(annot1_color, 0.1),
    )
    featY = graph.nx_makenode(
        'Features Y',
        size=(ns / 1.2, ns / 2),
        groupid='feats',
        color=lighten_hex(annot2_color, 0.1),
    )
    # '#4771B3')

    global_pairvec = graph.nx_makenode(
        'Global similarity\n(viewpoint, quality, ...)',
        width=ns * ut.PHI * 1.2,
        color=s2_shade2,
    )
    findnn = graph.nx_makenode(
        'Find correspondences\n(nearest neighbors)', shape='ellipse', color=c_shade2
    )
    local_pairvec = graph.nx_makenode(
        'Local similarities\n(LNBNN, spatial error, ...)',
        size=(ns * 2.2, ns),
        color=lighten_hex(c_shade2, 0.1),
    )
    agglocal = graph.nx_makenode(
        'Aggregate',
        size=(ns / 1.1, ns / 2),
        shape='ellipse',
        color=lighten_hex(c_shade2, 0.2),
    )
    catvecs = graph.nx_makenode(
        'Concatenate',
        size=(ns / 1.1, ns / 2),
        shape='ellipse',
        color=lighten_hex(s2_shade2, 0.1),
    )
    pairvec = graph.nx_makenode(
        'Vector of\npairwise similarities', color=lighten_hex(s2_shade2, 0.2)
    )
    classifier = graph.nx_makenode(
        'Classifier\n(SVM/RF/DNN)', color=lighten_hex(s2_shade2, 0.3)
    )
    prob = graph.nx_makenode(
        'Matching Probability\n(same individual given\nsimilar viewpoint)',
        color=lighten_hex(s2_shade2, 0.4),
    )

    graph.add_edge(annot1, global_pairvec)
    graph.add_edge(annot2, global_pairvec)

    add_edge2(annot1, featX)
    add_edge2(annot2, featY)

    add_edge2(featX, findnn)
    add_edge2(featY, findnn)

    add_edge2(findnn, local_pairvec)

    graph.add_edge(local_pairvec, agglocal, constraint=True)
    graph.add_edge(agglocal, catvecs, constraint=False)
    graph.add_edge(global_pairvec, catvecs)

    graph.add_edge(catvecs, pairvec)

    # graph.add_edge(annot1, classifier, style='invis')
    # graph.add_edge(pairvec, classifier , constraint=False)
    graph.add_edge(pairvec, classifier)
    graph.add_edge(classifier, prob)

    ut.nx_set_default_node_attributes(graph, 'shape', 'rect')
    # ut.nx_set_default_node_attributes(graph, 'fillcolor', nx.get_node_attributes(graph, 'color'))
    # ut.nx_set_default_node_attributes(graph, 'style',  'rounded')
    ut.nx_set_default_node_attributes(graph, 'style', 'filled,rounded')
    ut.nx_set_default_node_attributes(graph, 'fixedsize', 'true')
    ut.nx_set_default_node_attributes(
        graph, 'xlabel', nx.get_node_attributes(graph, 'label')
    )
    ut.nx_set_default_node_attributes(graph, 'width', ns * ut.PHI)
    ut.nx_set_default_node_attributes(graph, 'height', ns)
    ut.nx_set_default_node_attributes(graph, 'regular', False)

    # font = 'MonoDyslexic'
    # font = 'Mono_Dyslexic'
    font = 'Ubuntu'
    ut.nx_set_default_node_attributes(graph, 'fontsize', 72)
    ut.nx_set_default_node_attributes(graph, 'fontname', font)

    # ut.nx_delete_node_attr(graph, 'width')
    # ut.nx_delete_node_attr(graph, 'height')
    # ut.nx_delete_node_attr(graph, 'fixedsize')
    # ut.nx_delete_node_attr(graph, 'style')
    # ut.nx_delete_node_attr(graph, 'regular')
    # ut.nx_delete_node_attr(graph, 'shape')

    # node_dict = ut.nx_node_dict(graph)
    # node_dict[annot1]['label'] = "<f0> left|<f1> mid&#92; dle|<f2> right"
    # node_dict[annot2]['label'] = ut.codeblock(
    #    '''
    #    <<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
    #      <TR><TD>left</TD><TD PORT="f1">mid dle</TD><TD PORT="f2">right</TD></TR>
    #    </TABLE>>
    #    ''')
    # node_dict[annot1]['label'] = ut.codeblock(
    #    '''
    #    <<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
    #      <TR><TD>left</TD><TD PORT="f1">mid dle</TD><TD PORT="f2">right</TD></TR>
    #    </TABLE>>
    #    ''')

    # node_dict[annot1]['shape'] = 'none'
    # node_dict[annot1]['margin'] = '0'

    layoutkw = {
        'forcelabels': True,
        'prog': 'dot',
        'rankdir': 'LR',
        # 'splines': 'curved',
        'splines': 'line',
        'samplepoints': 20,
        'showboxes': 1,
        # 'splines': 'polyline',
        # 'splines': 'spline',
        'sep': 100 / 72,
        'nodesep': 300 / 72,
        'ranksep': 300 / 72,
        # 'inputscale': 72,
        # 'inputscale': 1,
        # 'dpi': 72,
        # 'concentrate': 'true', # merges edge lines
        # 'splines': 'ortho',
        # 'aspect': 1,
        # 'ratio': 'compress',
        # 'size': '5,4000',
        # 'rank': 'max',
    }

    # fontkw = dict(fontfamilty='sans-serif', fontweight='normal', fontsize=12)
    # fontkw = dict(fontname='Ubuntu', fontweight='normal', fontsize=12)
    # fontkw = dict(fontname='Ubuntu', fontweight='light', fontsize=20)
    fontkw = dict(fontname=font, fontweight='light', fontsize=12)
    # prop = fm.FontProperties(fname='/usr/share/fonts/truetype/groovygh.ttf')

    pt.show_nx(graph, layout='agraph', layoutkw=layoutkw, **fontkw)
    pt.zoom_factory()


def graphcut_flow():
    r"""
    Returns:
        ?: name

    CommandLine:
        python -m wbia.scripts.specialdraw graphcut_flow --show
        python -m wbia.scripts.specialdraw graphcut_flow --show --save cutflow.png --diskshow --clipwhite
        python -m wbia.scripts.specialdraw graphcut_flow --save figures4/cutiden.png --diskshow --clipwhite --dpath ~/latex/crall-candidacy-2015/ --figsize=24,10 --arrow-width=2.0

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.scripts.specialdraw import *  # NOQA
        >>> graphcut_flow()
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> ut.show_if_requested()
    """
    import wbia.plottool as pt

    pt.ensureqt()
    import networkx as nx

    # pt.plt.xkcd()

    graph = nx.DiGraph()

    def makecluster(name, num, **attrkw):
        return [nx_makenode(graph, name + str(n), **attrkw) for n in range(num)]

    def add_edge2(u, v, *args, **kwargs):
        v = ut.ensure_iterable(v)
        u = ut.ensure_iterable(u)
        for _u, _v in ut.product(u, v):
            graph.add_edge(_u, _v, *args, **kwargs)

    ns = 512

    # *** Primary color:
    p_shade2 = '#41629A'
    # *** Secondary color
    s1_shade2 = '#E88B53'
    # *** Secondary color
    s2_shade2 = '#36977F'
    # *** Complement color
    c_shade2 = '#E8B353'

    annot1 = nx_makenode(
        graph,
        'Unlabeled\nannotations\n(query)',
        width=ns,
        height=ns,
        groupid='annot',
        color=p_shade2,
    )
    annot2 = nx_makenode(
        graph,
        'Labeled\nannotations\n(database)',
        width=ns,
        height=ns,
        groupid='annot',
        color=s1_shade2,
    )
    occurprob = nx_makenode(
        graph, 'Dense \nprobabilities', color=lighten_hex(p_shade2, 0.1)
    )
    cacheprob = nx_makenode(
        graph, 'Cached \nprobabilities', color=lighten_hex(s1_shade2, 0.1)
    )
    sparseprob = nx_makenode(
        graph, 'Sparse\nprobabilities', color=lighten_hex(c_shade2, 0.1)
    )

    graph.add_edge(annot1, occurprob)

    graph.add_edge(annot1, sparseprob)
    graph.add_edge(annot2, sparseprob)
    graph.add_edge(annot2, cacheprob)

    matchgraph = nx_makenode(
        graph, 'Graph of\npotential matches', color=lighten_hex(s2_shade2, 0.1)
    )
    cutalgo = nx_makenode(
        graph, 'Graph cut algorithm', color=lighten_hex(s2_shade2, 0.2), shape='ellipse'
    )
    cc_names = nx_makenode(
        graph,
        'Identifications,\n splits, and merges are\nconnected components',
        color=lighten_hex(s2_shade2, 0.3),
    )

    graph.add_edge(occurprob, matchgraph)
    graph.add_edge(sparseprob, matchgraph)
    graph.add_edge(cacheprob, matchgraph)

    graph.add_edge(matchgraph, cutalgo)
    graph.add_edge(cutalgo, cc_names)

    ut.nx_set_default_node_attributes(graph, 'shape', 'rect')
    ut.nx_set_default_node_attributes(graph, 'style', 'filled,rounded')
    ut.nx_set_default_node_attributes(graph, 'fixedsize', 'true')
    ut.nx_set_default_node_attributes(graph, 'width', ns * ut.PHI)
    ut.nx_set_default_node_attributes(graph, 'height', ns * (1 / ut.PHI))
    ut.nx_set_default_node_attributes(graph, 'regular', False)

    layoutkw = {
        'prog': 'dot',
        'rankdir': 'LR',
        'splines': 'line',
        'sep': 100 / 72,
        'nodesep': 300 / 72,
        'ranksep': 300 / 72,
    }

    fontkw = dict(fontname='Ubuntu', fontweight='light', fontsize=14)
    pt.show_nx(graph, layout='agraph', layoutkw=layoutkw, **fontkw)
    pt.zoom_factory()


def merge_viewpoint_graph():
    r"""
    CommandLine:
        python -m wbia.scripts.specialdraw merge_viewpoint_graph --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.scripts.specialdraw import *  # NOQA
        >>> result = merge_viewpoint_graph()
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> ut.show_if_requested()
    """
    import wbia.plottool as pt
    import wbia
    import networkx as nx

    defaultdb = 'PZ_Master1'
    ibs = wbia.opendb(defaultdb=defaultdb)

    # nids = None
    aids = ibs.get_name_aids(4875)
    ibs.print_annot_stats(aids)

    left_aids = ibs.filter_annots_general(aids, view='left')[0:3]
    right_aids = ibs.filter_annots_general(aids, view='right')
    right_aids = list(set(right_aids) - {14517})[0:3]
    back = ibs.filter_annots_general(aids, view='back')[0:4]
    backleft = ibs.filter_annots_general(aids, view='backleft')[0:4]
    backright = ibs.filter_annots_general(aids, view='backright')[0:4]

    right_graph = nx.DiGraph(ut.upper_diag_self_prodx(right_aids))
    left_graph = nx.DiGraph(ut.upper_diag_self_prodx(left_aids))
    back_edges = [
        tuple([back[0], backright[0]][::1]),
        tuple([back[0], backleft[0]][::1]),
    ]
    back_graph = nx.DiGraph(back_edges)

    # Let the graph be a bit smaller

    right_graph.edge[right_aids[1]][right_aids[2]]['constraint'] = ut.get_argflag(
        '--constraint'
    )
    left_graph.edge[left_aids[1]][left_aids[2]]['constraint'] = ut.get_argflag(
        '--constraint'
    )

    # right_graph = right_graph.to_undirected().to_directed()
    # left_graph = left_graph.to_undirected().to_directed()
    nx.set_node_attributes(right_graph, name='groupid', values='right')
    nx.set_node_attributes(left_graph, name='groupid', values='left')

    # nx.set_node_attributes(right_graph, name='scale', values=.2)
    # nx.set_node_attributes(left_graph, name='scale', values=.2)
    # node_dict[back[0]]['scale'] = 2.3

    nx.set_node_attributes(back_graph, name='groupid', values='back')

    view_graph = nx.compose_all([left_graph, back_graph, right_graph])
    view_graph.add_edges_from(
        [[backright[0], right_aids[0]][::-1], [backleft[0], left_aids[0]][::-1]]
    )
    pt.ensureqt()
    graph = graph = view_graph  # NOQA
    # graph = graph.to_undirected()

    nx.set_edge_attributes(graph, name='color', values=pt.DARK_ORANGE[0:3])
    # nx.set_edge_attributes(graph, name='color', values=pt.BLACK)
    nx.set_edge_attributes(
        graph, name='color', values={edge: pt.LIGHT_BLUE[0:3] for edge in back_edges}
    )

    # pt.close_all_figures();
    from wbia.viz import viz_graph

    layoutkw = {
        'nodesep': 1,
    }
    viz_graph.viz_netx_chipgraph(
        ibs, graph, with_images=1, prog='dot', augment_graph=False, layoutkw=layoutkw
    )

    if False:
        """
        #view_graph = left_graph
        pt.close_all_figures(); viz_netx_chipgraph(ibs, view_graph, with_images=0, prog='neato')
        #viz_netx_chipgraph(ibs, view_graph, layout='pydot', with_images=False)
        #back_graph = make_name_graph_interaction(ibs, aids=back, with_all=False)

        aids = left_aids + back + backleft + backright + right_aids

        for aid, chip in zip(aids, ibs.get_annot_chips(aids)):
            fpath = ut.truepath('~/slides/merge/aid_%d.jpg' % (aid,))
            vt.imwrite(fpath, vt.resize_to_maxdims(chip, (400, 400)))

        ut.copy_files_to(, )

        aids = ibs.filterannots_by_tags(ibs.get_valid_aids(),
        dict(has_any_annotmatch='splitcase'))

        aid1 = ibs.group_annots_by_name_dict(aids)[252]
        aid2 = ibs.group_annots_by_name_dict(aids)[6791]
        aids1 = ibs.get_annot_groundtruth(aid1)[0][0:4]
        aids2 = ibs.get_annot_groundtruth(aid2)[0]

        make_name_graph_interaction(ibs, aids=aids1 + aids2, with_all=False)

        ut.ensuredir(ut.truthpath('~/slides/split/))

        for aid, chip in zip(aids, ibs.get_annot_chips(aids)):
            fpath = ut.truepath('~/slides/merge/aidA_%d.jpg' % (aid,))
            vt.imwrite(fpath, vt.resize_to_maxdims(chip, (400, 400)))
        """
    pass


def setcover_example():
    """
    CommandLine:
        python -m wbia.scripts.specialdraw setcover_example --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.scripts.specialdraw import *  # NOQA
        >>> result = setcover_example()
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> ut.show_if_requested()
    """
    import wbia
    import wbia.plottool as pt
    from wbia.viz import viz_graph
    import networkx as nx

    pt.ensureqt()
    ibs = wbia.opendb(defaultdb='testdb2')

    if False:
        # Select a good set
        aids = ibs.get_name_aids(ibs.get_valid_nids())
        # wbia.testdata_aids('testdb2', a='default:mingt=2')
        aids = [a for a in aids if len(a) > 1]
        for a in aids:
            print(ut.repr3(ibs.get_annot_stats_dict(a)))
        print(aids[-2])
    # aids = [78, 79, 80, 81, 88, 91]
    aids = [78, 79, 81, 88, 91]
    qreq_ = ibs.depc.new_request('vsone', aids, aids)
    cm_list = qreq_.execute()
    from wbia.algo.hots import orig_graph_iden

    infr = orig_graph_iden.OrigAnnotInference(cm_list)
    unique_aids, prob_annots = infr.make_prob_annots()
    import numpy as np

    print(
        ut.hz_str(
            'prob_annots = ',
            ut.repr2(prob_annots, precision=2, max_line_width=140, suppress_small=True),
        )
    )
    # ut.setcover_greedy(candidate_sets_dict)
    max_weight = 3
    prob_annots[np.diag_indices(len(prob_annots))] = np.inf
    prob_annots = prob_annots
    thresh_points = np.sort(prob_annots[np.isfinite(prob_annots)])

    # probably not the best way to go about searching for these thresholds
    # but when you have a hammer...
    if False:
        quant = sorted(np.diff(thresh_points))[(len(thresh_points) - 1) // 2]
        candset = {
            point: thresh_points[np.abs(thresh_points - point) < quant]
            for point in thresh_points
        }
        check_thresholds = len(aids) * 2
        thresh_points2 = np.array(
            ut.setcover_greedy(candset, max_weight=check_thresholds).keys()
        )
        thresh_points = thresh_points2

    # pt.plot(sorted(thresh_points), 'rx')
    # pt.plot(sorted(thresh_points2), 'o')

    # prob_annots = prob_annots.T

    # thresh_start = np.mean(thresh_points)
    current_idxs = []
    current_covers = []
    current_val = np.inf
    for thresh in thresh_points:
        covering_sets = [np.where(row >= thresh)[0] for row in (prob_annots)]
        candidate_sets_dict = {ax: others for ax, others in enumerate(covering_sets)}
        soln_cover = ut.setcover_ilp(candidate_sets_dict, max_weight=max_weight)
        exemplar_idxs = list(soln_cover.keys())
        soln_weight = len(exemplar_idxs)
        val = max_weight - soln_weight
        # print('val = %r' % (val,))
        # print('soln_weight = %r' % (soln_weight,))
        if val < current_val:
            current_val = val
            current_covers = covering_sets
            current_idxs = exemplar_idxs
    exemplars = ut.take(aids, current_idxs)
    ensure_edges = [
        (aids[ax], aids[ax2])
        for ax, other_xs in enumerate(current_covers)
        for ax2 in other_xs
    ]
    graph = viz_graph.make_netx_graph_from_aid_groups(
        ibs,
        [aids],
        allow_directed=True,
        ensure_edges=ensure_edges,
        temp_nids=[1] * len(aids),
    )
    viz_graph.ensure_node_images(ibs, graph)

    nx.set_node_attributes(graph, name='framewidth', values=False)
    nx.set_node_attributes(
        graph, name='framewidth', values={aid: 4.0 for aid in exemplars}
    )
    nx.set_edge_attributes(graph, name='color', values=pt.ORANGE)
    nx.set_node_attributes(graph, name='color', values=pt.LIGHT_BLUE)
    nx.set_node_attributes(graph, name='shape', values='rect')

    layoutkw = {
        'sep': 1 / 10,
        'prog': 'neato',
        'overlap': 'false',
        # 'splines': 'ortho',
        'splines': 'spline',
    }
    pt.show_nx(graph, layout='agraph', layoutkw=layoutkw)
    pt.zoom_factory()


def k_redun_demo():
    r"""

    python -m wbia.scripts.specialdraw k_redun_demo --save=kredun.png
    python -m wbia.scripts.specialdraw k_redun_demo --show

    Example:
        >>> # SCRIPT
        >>> from wbia.scripts.specialdraw import *  # NOQA
        >>> k_redun_demo()
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> ut.show_if_requested()
    """
    import wbia
    import wbia.plottool as pt
    from wbia.viz import viz_graph
    from wbia.algo.graph.state import POSTV, NEGTV, INCMP

    # import networkx as nx
    pt.ensureqt()
    ibs = wbia.opendb(defaultdb='PZ_Master1')
    nid2_aid = {
        6612: [7664, 7462, 7522],
        6625: [7746, 7383, 7390, 7477, 7376, 7579],
        6630: [7586, 7377, 7464, 7478],
    }
    aids = ut.flatten(nid2_aid.values())
    infr = wbia.AnnotInference(ibs=ibs, aids=aids, autoinit=True)

    for name_aids in nid2_aid.values():
        for edge in ut.itertwo(name_aids):
            infr.add_feedback(edge, POSTV)
    infr.add_feedback((7664, 7522), POSTV)
    infr.add_feedback((7746, 7477), POSTV)
    infr.add_feedback((7383, 7376), POSTV)

    # infr.add_feedback((7664, 7383), NEGTV)
    # infr.add_feedback((7462, 7746), NEGTV)

    # infr.add_feedback((7464, 7376), NEGTV)

    # Adjust between new and old variable names
    infr.set_edge_attrs('evidence_decision', infr.get_edge_attrs('evidence_decision'))
    infr.set_edge_attrs(
        infr.CUT_WEIGHT_KEY,
        ut.dzip(infr.get_edges_where_eq('evidence_decision', POSTV), [1.0]),
    )
    infr.set_edge_attrs(
        infr.CUT_WEIGHT_KEY,
        ut.dzip(infr.get_edges_where_eq('evidence_decision', NEGTV), [0.0]),
    )
    infr.set_edge_attrs(
        infr.CUT_WEIGHT_KEY,
        ut.dzip(infr.get_edges_where_eq('evidence_decision', INCMP), [0.5]),
    )

    infr.initialize_visual_node_attrs()
    infr.update_node_image_attribute(use_image=True)
    infr.update_visual_attrs(
        use_image=True,
        show_unreviewed_edges=True,
        groupby='name_label',
        splines='spline',
        show_cand=False,
    )
    infr.set_edge_attrs('linewidth', 2)
    # infr.set_edge_attrs('linewidth', ut.dzip(infr.get_edges_where_eq('evidence_decision', POSTV), [4]))
    # infr.set_edge_attrs('color', pt.BLACK)
    infr.set_edge_attrs('alpha', 0.7)
    viz_graph.ensure_node_images(ibs, infr.graph)
    infr.show(use_image=True, update_attrs=False)


def graph_iden_cut_demo():
    r"""
    CommandLine:
        python -m wbia.scripts.specialdraw graph_iden_cut_demo --show --precut
        python -m wbia.scripts.specialdraw graph_iden_cut_demo --show --postcut

        python -m wbia.scripts.specialdraw graph_iden_cut_demo --precut --save=precut.png --clipwhite
        python -m wbia.scripts.specialdraw graph_iden_cut_demo --postcut --save=postcut.png --clipwhite

    Example:
        >>> # SCRIPT
        >>> from wbia.scripts.specialdraw import *  # NOQA
        >>> graph_iden_cut_demo()
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> ut.show_if_requested()
    """
    import wbia
    import wbia.plottool as pt
    from wbia.viz import viz_graph

    # import networkx as nx
    pt.ensureqt()
    ibs = wbia.opendb(defaultdb='PZ_Master1')
    nid2_aid = {
        # 4880: [3690, 3696, 3703, 3706, 3712, 3721],
        4880: [3690, 3696, 3703],
        6537: [3739],
        # 6653: [7671],
        6610: [7566, 7408],
        # 6612: [7664, 7462, 7522],
        # 6624: [7465, 7360],
        # 6625: [7746, 7383, 7390, 7477, 7376, 7579],
        6630: [7586, 7377, 7464, 7478],
        # 6677: [7500]
    }

    if False:
        # Find extra example
        annots = ibs.annots(
            ibs.filter_annots_general(view='right', require_timestamp=True, min_pername=2)
        )
        unique_nids = ut.unique(annots.nids)
        nid_to_annots = ut.dzip(
            unique_nids, map(ibs.annots, ibs.get_name_aids(unique_nids))
        )
        # nid_to_annots = annots.group_items(annots.nids)
        right_nids = ut.argsort(ut.map_dict_vals(len, nid_to_annots))[::-1]
        right_annots = nid_to_annots[right_nids[1]]
        inter = pt.interact_multi_image.MultiImageInteraction(right_annots.chips)
        inter.start()

        inter = pt.interact_multi_image.MultiImageInteraction(
            ibs.annots([16228, 16257, 16273]).chips
        )
        inter.start()
        ut.take(right_annots.aids, [2, 6, 10])

    nid2_aid.update({4429: [16228, 16257, 16273]})

    aids = ut.flatten(nid2_aid.values())

    postcut = ut.get_argflag('--postcut')
    aids_list = ibs.group_annots_by_name(aids)[0]

    infr = wbia.AnnotInference(ibs=ibs, aids=ut.flatten(aids_list), autoinit=True)
    if postcut:
        infr.init_test_mode2(enable_autoreview=False)

        node_to_label = infr.get_node_attrs('orig_name_label')
        label_to_nodes = ut.group_items(node_to_label.keys(), node_to_label.values())
        # cliques
        new_edges = []
        for label, nodes in label_to_nodes.items():
            for edge in ut.combinations(nodes, 2):
                if not infr.has_edge(edge):
                    new_edges.append(infr.e_(*edge))
        # negative edges
        import random

        rng = random.Random(0)
        for aids1, aids2 in ut.combinations(nid2_aid.values(), 2):
            aid1 = rng.choice(aids1)
            aid2 = rng.choice(aids2)
            new_edges.append(infr.e_(aid1, aid2))

        infr.graph.add_edges_from(new_edges)
        infr.apply_edge_truth(new_edges)
        for edge in new_edges:
            infr.queue.push(edge, -1)

        from wbia.algo.graph.state import POSTV, NEGTV, INCMP

        try:
            while True:
                edge, priority = infr.pop()
                feedback = infr.request_user_review(edge)
                infr.add_feedback(edge=edge, **feedback)
        except StopIteration:
            pass
    else:
        infr.ensure_full()

    # Adjust between new and old variable names
    infr.set_edge_attrs('evidence_decision', infr.get_edge_attrs('evidence_decision'))
    infr.set_edge_attrs(
        infr.CUT_WEIGHT_KEY,
        ut.dzip(infr.get_edges_where_eq('evidence_decision', POSTV), [1.0]),
    )
    infr.set_edge_attrs(
        infr.CUT_WEIGHT_KEY,
        ut.dzip(infr.get_edges_where_eq('evidence_decision', NEGTV), [0.0]),
    )
    infr.set_edge_attrs(
        infr.CUT_WEIGHT_KEY,
        ut.dzip(infr.get_edges_where_eq('evidence_decision', INCMP), [0.5]),
    )

    infr.initialize_visual_node_attrs()
    infr.update_node_image_attribute(use_image=True)
    infr.update_visual_attrs(
        use_image=True,
        show_unreviewed_edges=True,
        groupby='name_label',
        splines='spline',
        show_cand=not postcut,
    )
    infr.set_edge_attrs('linewidth', 2)
    infr.set_edge_attrs(
        'linewidth', ut.dzip(infr.get_edges_where_eq('evidence_decision', POSTV), [4])
    )
    if not postcut:
        infr.set_edge_attrs('color', pt.BLACK)
    infr.set_edge_attrs('alpha', 0.7)
    if not postcut:
        infr.set_node_attrs('framewidth', 0)
    viz_graph.ensure_node_images(ibs, infr.graph)
    infr.show(use_image=True, update_attrs=False)


def show_id_graph():
    r"""
    CommandLine:
        python -m wbia.scripts.specialdraw show_id_graph --show
        python -m wbia.scripts.specialdraw show_id_graph --show

    Example:
        >>> # SCRIPT
        >>> from wbia.scripts.specialdraw import *  # NOQA
        >>> show_id_graph()
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> ut.show_if_requested()
    """
    import wbia
    import wbia.plottool as pt

    # import networkx as nx
    pt.ensureqt()
    # ibs = wbia.opendb(defaultdb='PZ_PB_RF_TRAIN')
    ibs = wbia.opendb(defaultdb='PZ_Master1')

    parent_infr = wbia.AnnotInference(ibs=ibs, aids='all')
    parent_infr.reset_feedback('staging', apply=True)

    edgecat = parent_infr.categorize_edges()

    MAX_SIZE = 6
    MAX_NUM = 6

    pccs = []
    infr = parent_infr
    if ibs.dbname == 'PZ_Master1':
        incomp_pcc = {5652, 5197, 4244}
        force_incomp_edge = [(5652, 5197)]
        pccs.append(incomp_pcc)
    else:
        pccs = []
        force_incomp_edge = []
        if len(pccs) == 0:
            for (n1, n2), es in edgecat['notcomp'].items():
                if n1 == n2:
                    cc = parent_infr.pos_graph._ccs[n1]
                    pccs.append(cc)
                    break
        if len(pccs) == 0:
            for cc in parent_infr.positive_components():
                a = ibs.annots(cc)
                if any(t is not None and 'left' not in t for t in a.yaw_texts):
                    # print(a.yaw_texts)
                    if any(t is not None and 'left' in t for t in a.yaw_texts):
                        if any(t is not None and 'right' in t for t in a.yaw_texts):
                            print(a.yaw_texts)
                            if len(cc) <= MAX_SIZE:
                                pccs.append(cc)
                        # break
        if len(pccs) == 0:
            for (n1, n2), es in edgecat['notcomp'].items():
                cc1 = parent_infr.pos_graph._ccs[n1]
                cc2 = parent_infr.pos_graph._ccs[n2]
                # s1 = len(parent_infr.pos_graph._ccs[n1])
                # s2 = len(parent_infr.pos_graph._ccs[n2])
                # if s1 in {3} and s2 in {3}:
                # print(annots1.yaw_texts)
                # print(annots2.yaw_texts)
                pccs.append(frozenset(cc1))
                pccs.append(frozenset(cc2))
                break

    MAX_SIZE += len(pccs) - 1

    for cc in parent_infr.positive_components():
        cc = frozenset(cc)
        if len(cc) < MAX_SIZE:
            if cc not in pccs:
                if len(cc) not in set(map(len, pccs)):
                    pccs.append(cc)
        if len(pccs) >= MAX_NUM:
            break

    from wbia.algo.graph.state import POSTV, NEGTV, INCMP, UNREV  # NOQA

    subinfr = parent_infr.subgraph(ut.flatten(pccs))
    subinfr._viz_image_config['thumbsize'] = 700
    subinfr._viz_image_config['grow'] = True
    infr = subinfr

    infr.apply_nondynamic_update()
    # infr.ensure_mst()
    infr.ensure_mst(label='orig_name_label')
    # infr.ensure_cliques(evidence_decision=POSTV)

    # infr.show(pickable=True, use_image=True, groupby='name_label',
    #              splines='spline')

    infr.apply_nondynamic_update()
    for edge in infr.find_neg_redun_candidate_edges(k=1):
        infr.add_feedback(edge, evidence_decision=NEGTV)

    import itertools as it

    edges = list(it.combinations(infr.aids, 2))
    n = 0
    incomp_edges = ut.compress(edges, [not f for f in infr.is_comparable(edges)])
    for e in ut.shuffle(incomp_edges, rng=3545115929):
        infr.add_feedback(e, evidence_decision=INCMP)
        n += 1
        if n > 3:
            break
    for e in force_incomp_edge:
        infr.add_feedback(e, evidence_decision=INCMP)

    for edge in infr.find_neg_redun_candidate_edges(k=1):
        infr.add_feedback(edge, evidence_decision=NEGTV)

    savekw = dict(dpi=300, transparent=True, edgecolor='none')
    showkw = dict(
        pickable=True, use_image=True, groupby='name_label', splines='spline', fnum=1
    )

    infr.show(
        show_positive_edges=False,
        show_negative_edges=False,
        show_incomparable_edges=False,
        **showkw,
    )
    fig = pt.gcf()
    fig.savefig(
        'id_graph1.png', bbox_inches=pt.extract_axes_extents(fig, combine=True), **savekw,
    )

    infr.show(
        show_positive_edges=True,
        show_negative_edges=False,
        show_incomparable_edges=False,
        **showkw,
    )
    fig = pt.gcf()
    fig.savefig(
        'id_graph2.png', bbox_inches=pt.extract_axes_extents(fig, combine=True), **savekw,
    )

    infr.show(
        show_positive_edges=False,
        show_negative_edges=True,
        show_incomparable_edges=False,
        **showkw,
    )
    fig = pt.gcf()
    fig.savefig(
        'id_graph3.png', bbox_inches=pt.extract_axes_extents(fig, combine=True), **savekw,
    )

    infr.show(
        show_positive_edges=False,
        show_negative_edges=False,
        show_incomparable_edges=True,
        **showkw,
    )
    fig = pt.gcf()
    fig.savefig(
        'id_graph4.png', bbox_inches=pt.extract_axes_extents(fig, combine=True), **savekw,
    )
    import networkx as nx

    infr.show(pin=True, **showkw)
    nx.set_node_attributes(infr.graph, name='pin', values='true')
    fig = pt.gcf()
    fig.savefig(
        'id_graph5.png', bbox_inches=pt.extract_axes_extents(fig, combine=True), **savekw,
    )

    infr2 = infr.copy()
    for edge in infr2.find_pos_redun_candidate_edges(k=2):
        infr2.add_feedback(edge, evidence_decision=POSTV)
    infr2.show(
        pickable=True, use_image=True, groupby='name_label', fnum=1, splines='spline'
    )
    fig = pt.gcf()
    fig.savefig(
        'id_graph6.png', bbox_inches=pt.extract_axes_extents(fig, combine=True), **savekw,
    )
    for edge in infr2.find_neg_redun_candidate_edges(k=2):
        infr2.add_feedback(edge, evidence_decision=NEGTV)
    infr2.show(
        pickable=True, use_image=True, groupby='name_label', fnum=1, splines='spline'
    )
    fig = pt.gcf()
    fig.savefig(
        'id_graph7.png', bbox_inches=pt.extract_axes_extents(fig, combine=True), **savekw,
    )

    infr3 = infr.copy()
    for edge in infr3.find_pos_redun_candidate_edges(k=2):
        infr3.add_feedback(edge, evidence_decision=POSTV)
    for cc in infr3.non_pos_redundant_pccs(k=3):
        for edge in infr3.find_pos_augment_edges(cc, k=3):
            infr3.add_feedback(edge, evidence_decision=NEGTV)
            break
    infr3.show(
        pickable=True,
        use_image=True,
        show_between=False,
        show_inconsistency=True,
        groupby='name_label',
        fnum=1,
        splines='spline',
    )
    fig = pt.gcf()
    fig.savefig(
        'id_graph8.png', bbox_inches=pt.extract_axes_extents(fig, combine=True), **savekw,
    )

    infr4 = infr.copy()
    for edge in infr4.edges():
        infr4.add_feedback(edge, evidence_decision=UNREV)
    infr4.refresh_candidate_edges()
    infr4.show(show_cand=True, **showkw)
    fig = pt.gcf()
    fig.savefig(
        'id_graph9.png', bbox_inches=pt.extract_axes_extents(fig, combine=True), **savekw,
    )


def intraoccurrence_connected():
    r"""
    CommandLine:
        python -m wbia.scripts.specialdraw intraoccurrence_connected --show
        python -m wbia.scripts.specialdraw intraoccurrence_connected --show --smaller

        python -m wbia.scripts.specialdraw intraoccurrence_connected --precut --save=precut.jpg
        python -m wbia.scripts.specialdraw intraoccurrence_connected --postcut --save=postcut.jpg

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.scripts.specialdraw import *  # NOQA
        >>> result = intraoccurrence_connected()
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> ut.show_if_requested()
    """
    import wbia
    import wbia.plottool as pt
    from wbia.viz import viz_graph
    import networkx as nx

    pt.ensureqt()
    ibs = wbia.opendb(defaultdb='PZ_Master1')
    nid2_aid = {
        # 4880: [3690, 3696, 3703, 3706, 3712, 3721],
        4880: [3690, 3696, 3703],
        6537: [3739],
        6653: [7671],
        6610: [7566, 7408],
        # 6612: [7664, 7462, 7522],
        # 6624: [7465, 7360],
        # 6625: [7746, 7383, 7390, 7477, 7376, 7579],
        6630: [7586, 7377, 7464, 7478],
        # 6677: [7500]
    }
    nid2_dbaids = {4880: [33, 6120, 7164], 6537: [7017, 7206], 6653: [7660]}
    if ut.get_argflag('--small') or ut.get_argflag('--smaller'):
        del nid2_aid[6630]
        del nid2_aid[6537]
        del nid2_dbaids[6537]
        if ut.get_argflag('--smaller'):
            nid2_dbaids[4880].remove(33)
            nid2_aid[4880].remove(3690)
            nid2_aid[6610].remove(7408)
        # del nid2_aid[4880]
        # del nid2_dbaids[4880]

    aids = ut.flatten(nid2_aid.values())

    temp_nids = [1] * len(aids)
    postcut = ut.get_argflag('--postcut')
    aids_list = ibs.group_annots_by_name(aids)[0]

    ensure_edges = 'all' if True or not postcut else None
    # unlabeled_graph = infr.graph
    unlabeled_graph = viz_graph.make_netx_graph_from_aid_groups(
        ibs,
        aids_list,
        # invis_edges=invis_edges,
        ensure_edges=ensure_edges,
        temp_nids=temp_nids,
    )
    viz_graph.color_by_nids(
        unlabeled_graph, unique_nids=[1] * len(list(unlabeled_graph.nodes()))
    )
    viz_graph.ensure_node_images(ibs, unlabeled_graph)
    nx.set_node_attributes(unlabeled_graph, name='shape', values='rect')
    # unlabeled_graph = unlabeled_graph.to_undirected()

    # Find the "database exemplars for these annots"
    if False:
        gt_aids = ibs.get_annot_groundtruth(aids)
        gt_aids = [ut.setdiff(s, aids) for s in gt_aids]
        dbaids = ut.unique(ut.flatten(gt_aids))
        dbaids = ibs.filter_annots_general(dbaids, minqual='good')
        ibs.get_annot_quality_texts(dbaids)
    else:
        dbaids = ut.flatten(nid2_dbaids.values())
    exemplars = nx.DiGraph()
    # graph = exemplars  # NOQA
    exemplars.add_nodes_from(dbaids)

    def add_clique(graph, nodes, edgeattrs={}, nodeattrs={}):
        edge_list = ut.upper_diag_self_prodx(nodes)
        graph.add_edges_from(edge_list, **edgeattrs)
        return edge_list

    for aids_, nid in zip(*ibs.group_annots_by_name(dbaids)):
        add_clique(exemplars, aids_)
    viz_graph.ensure_node_images(ibs, exemplars)
    viz_graph.color_by_nids(exemplars, ibs=ibs)

    nx.set_node_attributes(unlabeled_graph, name='framewidth', values=False)
    nx.set_node_attributes(exemplars, name='framewidth', values=4.0)

    nx.set_node_attributes(unlabeled_graph, name='group', values='unlab')
    nx.set_node_attributes(exemplars, name='group', values='exemp')

    # big_graph = nx.compose_all([unlabeled_graph])
    big_graph = nx.compose_all([exemplars, unlabeled_graph])

    # add sparse connections from unlabeled to exemplars
    import numpy as np

    rng = np.random.RandomState(0)
    if True or not postcut:
        for aid_ in unlabeled_graph.nodes():
            flags = rng.rand(len(exemplars)) > 0.5
            nid_ = ibs.get_annot_nids(aid_)
            exnids = np.array(ibs.get_annot_nids(list(exemplars.nodes())))
            flags = np.logical_or(exnids == nid_, flags)
            exmatches = ut.compress(list(exemplars.nodes()), flags)
            big_graph.add_edges_from(
                list(ut.product([aid_], exmatches)), color=pt.ORANGE, implicit=True
            )
    else:
        for aid_ in unlabeled_graph.nodes():
            flags = rng.rand(len(exemplars)) > 0.5
            exmatches = ut.compress(list(exemplars.nodes()), flags)
            nid_ = ibs.get_annot_nids(aid_)
            exnids = np.array(ibs.get_annot_nids(exmatches))
            exmatches = ut.compress(exmatches, exnids == nid_)
            big_graph.add_edges_from(list(ut.product([aid_], exmatches)))
        pass

    nx.set_node_attributes(big_graph, name='shape', values='rect')
    # if False and postcut:
    #    ut.nx_delete_node_attr(big_graph, 'nid')
    #    ut.nx_delete_edge_attr(big_graph, 'color')
    #    viz_graph.ensure_graph_nid_labels(big_graph, ibs=ibs)
    #    viz_graph.color_by_nids(big_graph, ibs=ibs)
    #    big_graph = big_graph.to_undirected()

    layoutkw = {
        'sep': 1 / 5,
        'prog': 'neato',
        'overlap': 'false',
        # 'splines': 'ortho',
        'splines': 'spline',
    }

    as_directed = False
    # as_directed = True
    # hacknode = True
    hacknode = 0

    graph = big_graph
    ut.nx_ensure_agraph_color(graph)
    if hacknode:
        nx.set_edge_attributes(
            graph, name='taillabel', values={e: str(e[0]) for e in graph.edges()}
        )
        nx.set_edge_attributes(
            graph, name='headlabel', values={e: str(e[1]) for e in graph.edges()}
        )

    _, layout_info = pt.nx_agraph_layout(graph, inplace=True, **layoutkw)

    node_dict = ut.nx_node_dict(graph)
    if ut.get_argflag('--smaller'):
        node_dict[7660]['pos'] = np.array([550, 350])
        node_dict[6120]['pos'] = np.array([200, 600]) + np.array([350, -400])
        node_dict[7164]['pos'] = np.array([200, 480]) + np.array([350, -400])
        nx.set_node_attributes(graph, name='pin', values='true')
        _, layout_info = pt.nx_agraph_layout(graph, inplace=True, **layoutkw)
    elif ut.get_argflag('--small'):
        node_dict[7660]['pos'] = np.array([750, 350])
        node_dict[33]['pos'] = np.array([300, 600]) + np.array([350, -400])
        node_dict[6120]['pos'] = np.array([500, 600]) + np.array([350, -400])
        node_dict[7164]['pos'] = np.array([410, 480]) + np.array([350, -400])
        nx.set_node_attributes(graph, name='pin', values='true')
        _, layout_info = pt.nx_agraph_layout(graph, inplace=True, **layoutkw)

    if not postcut:
        # pt.show_nx(graph.to_undirected(), layout='agraph', layoutkw=layoutkw,
        #           as_directed=False)
        # pt.show_nx(graph, layout='agraph', layoutkw=layoutkw,
        #           as_directed=as_directed, hacknode=hacknode)

        pt.show_nx(
            graph,
            layout='custom',
            layoutkw=layoutkw,
            as_directed=as_directed,
            hacknode=hacknode,
        )
    else:
        # explicit_graph = pt.get_explicit_graph(graph)
        # _, layout_info = pt.nx_agraph_layout(explicit_graph, orig_graph=graph,
        #                                     **layoutkw)

        # layout_info['edge']['alpha'] = .8
        # pt.apply_graph_layout_attrs(graph, layout_info)

        # graph_layout_attrs = layout_info['graph']
        # # edge_layout_attrs  = layout_info['edge']
        # # node_layout_attrs  = layout_info['node']

        # for key, vals in layout_info['node'].items():
        #    #print('[special] key = %r' % (key,))
        #    nx.set_node_attributes(graph, name=key, values=vals)

        # for key, vals in layout_info['edge'].items():
        #    #print('[special] key = %r' % (key,))
        #    nx.set_edge_attributes(graph, name=key, values=vals)

        # nx.set_edge_attributes(graph, name='alpha', values=.8)
        # graph.graph['splines'] = graph_layout_attrs.get('splines', 'line')
        # graph.graph['splines'] = 'polyline'   # graph_layout_attrs.get('splines', 'line')
        # graph.graph['splines'] = 'line'

        cut_graph = graph.copy()
        edge_list = list(cut_graph.edges())
        edge_nids = np.array(ibs.unflat_map(ibs.get_annot_nids, edge_list))
        cut_flags = edge_nids.T[0] != edge_nids.T[1]
        cut_edges = ut.compress(edge_list, cut_flags)
        cut_graph.remove_edges_from(cut_edges)
        ut.nx_delete_node_attr(cut_graph, 'nid')
        viz_graph.ensure_graph_nid_labels(cut_graph, ibs=ibs)

        # ut.nx_get_default_node_attributes(exemplars, 'color', None)
        ut.nx_delete_node_attr(cut_graph, 'color', nodes=unlabeled_graph.nodes())
        aid2_color = ut.nx_get_default_node_attributes(cut_graph, 'color', None)
        nid2_colors = ut.group_items(
            aid2_color.values(), ibs.get_annot_nids(aid2_color.keys())
        )
        nid2_colors = ut.map_dict_vals(ut.filter_Nones, nid2_colors)
        nid2_colors = ut.map_dict_vals(ut.unique, nid2_colors)
        # for val in nid2_colors.values():
        #    assert len(val) <= 1
        # Get initial colors
        nid2_color_ = {
            nid: colors_[0] for nid, colors_ in nid2_colors.items() if len(colors_) == 1
        }

        graph = cut_graph
        viz_graph.color_by_nids(cut_graph, ibs=ibs, nid2_color_=nid2_color_)
        nx.set_node_attributes(cut_graph, name='framewidth', values=4)

        pt.show_nx(
            cut_graph,
            layout='custom',
            layoutkw=layoutkw,
            as_directed=as_directed,
            hacknode=hacknode,
        )

    pt.zoom_factory()

    # The database exemplars
    # TODO: match these along with the intra encounter set
    # interact = viz_graph.make_name_graph_interaction(
    #    ibs, aids=dbaids, with_all=False, prog='neato', framewidth=True)
    # print(interact)

    # Groupid only works for dot
    # nx.set_node_attributes(unlabeled_graph, name='groupid', values='unlabeled')
    # nx.set_node_attributes(exemplars, name='groupid', values='exemplars')
    # exemplars = exemplars.to_undirected()
    # add_clique(exemplars, aids_, edgeattrs=dict(constraint=False))
    # layoutkw = {}
    # pt.show_nx(exemplars, layout='agraph', layoutkw=layoutkw,
    #           as_directed=False, framewidth=True,)


def scalespace():
    r"""
    THIS DOES NOT SHOW A REAL SCALE SPACE PYRAMID YET. FIXME.

    Returns:
        ?: imgBGRA_warped

    CommandLine:
        python -m wbia.scripts.specialdraw scalespace --show

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.scripts.specialdraw import *  # NOQA
        >>> imgBGRA_warped = scalespace()
        >>> result = ('imgBGRA_warped = %s' % (ut.repr2(imgBGRA_warped),))
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> ut.show_if_requested()
    """
    import numpy as np

    # import matplotlib.pyplot as plt
    import cv2
    import vtool as vt
    import wbia.plottool as pt

    pt.qt4ensure()

    # imgBGR = vt.imread(ut.grab_test_imgpath('lena.png'))
    imgBGR = vt.imread(ut.grab_test_imgpath('zebra.png'))
    # imgBGR = vt.imread(ut.grab_test_imgpath('carl.jpg'))

    # Convert to colored intensity image
    imgGray = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)
    imgBGR = cv2.cvtColor(imgGray, cv2.COLOR_GRAY2BGR)
    imgRaw = imgBGR

    # TODO: # stack images in pyramid # boarder?
    initial_sigma = 1.6
    num_intervals = 4

    def makepyramid_octave(imgRaw, level, num_intervals):
        # Downsample image to take sigma to a power of level
        step = 2 ** (level)
        img_level = imgRaw[::step, ::step]
        # Compute interval relative scales
        interval = np.array(list(range(num_intervals)))
        relative_scales = 2 ** ((interval / num_intervals))
        sigma_intervals = initial_sigma * relative_scales
        octave_intervals = []
        for sigma in sigma_intervals:
            sizex = int(6.0 * sigma + 1.0) + int(1 - (int(6.0 * sigma + 1.0) % 2))
            ksize = (sizex, sizex)
            img_blur = cv2.GaussianBlur(
                img_level,
                ksize,
                sigmaX=sigma,
                sigmaY=sigma,
                borderType=cv2.BORDER_REPLICATE,
            )
            octave_intervals.append(img_blur)
        return octave_intervals

    pyramid = []
    num_octaves = 4
    for level in range(num_octaves):
        octave = makepyramid_octave(imgRaw, level, num_intervals)
        pyramid.append(octave)

    def makewarp(imgBGR):
        # hack a projection matrix using dummy homogrpahy
        imgBGRA = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2BGRA)
        imgBGRA[:, :, 3] = 0.87 * 255  # hack alpha
        imgBGRA = vt.pad_image(imgBGRA, 2, value=[0, 0, 255, 255])
        size = np.array(vt.get_size(imgBGRA))
        pts1 = np.array([(0, 0), (0, 1), (1, 1), (1, 0)]) * size
        x_adjust = 0.15
        y_adjust = 0.5
        pts2 = (
            np.array(
                [(x_adjust, 0), (0, 1 - y_adjust), (1, 1 - y_adjust), (1 - x_adjust, 0)]
            )
            * size
        )
        H = cv2.findHomography(pts1, pts2)[0]

        dsize = np.array(vt.bbox_from_verts(pts2)[2:4]).astype(np.int)
        warpkw = dict(flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
        imgBGRA_warped = cv2.warpPerspective(imgBGRA, H, tuple(dsize), **warpkw)
        return imgBGRA_warped

    framesize = (700, 500)
    steps = np.array([0.04, 0.03, 0.02, 0.01]) * 1.3

    numintervals = 4
    octave_ty_starts = [1.0]
    for i in range(1, 4):
        prev_ty = octave_ty_starts[-1]
        prev_base = pyramid[i - 1][0]
        next_ty = prev_ty - (
            (prev_base.shape[0] / framesize[1]) / 2 + (numintervals - 1) * (steps[i - 1])
        )
        octave_ty_starts.append(next_ty)

    def temprange(stop, step, num):
        return [stop - (x * step) for x in range(num)]

    layers = []
    for i in range(0, 4):
        ty_start = octave_ty_starts[i]
        step = steps[i]
        intervals = pyramid[i]
        ty_range = temprange(ty_start, step, numintervals)
        nextpart = [
            vt.embed_in_square_image(
                makewarp(interval),
                framesize,
                img_origin=(0.5, 0.5),
                target_origin=(0.5, ty / 2),
            )
            for ty, interval in zip(ty_range, intervals)
        ]
        layers += nextpart

    for layer in layers:
        pt.imshow(layer)

    pt.plt.grid(False)


def event_space():
    """
    pip install matplotlib-venn
    """
    from matplotlib import pyplot as plt

    # import numpy as np
    from matplotlib_venn import venn3, venn2, venn3_circles

    plt.figure(figsize=(4, 4))
    v = venn3(subsets=(1, 1, 1, 1, 1, 1, 1), set_labels=('A', 'B', 'C'))
    v.get_patch_by_id('100').set_alpha(1.0)
    v.get_patch_by_id('100').set_color('white')
    v.get_label_by_id('100').set_text('Unknown')
    v.get_label_by_id('A').set_text('Set "A"')
    c = venn3_circles(subsets=(1, 1, 1, 1, 1, 1, 1), linestyle='dashed')
    c[0].set_lw(1.0)
    c[0].set_ls('dotted')
    plt.show()

    same = set(['comparable', 'incomparable', 'same'])
    diff = set(['comparable', 'incomparable', 'diff'])
    # comparable = set(['comparable', 'same', 'diff'])
    # incomparable = set(['incomparable', 'same', 'diff'])
    subsets = [same, diff]  # , comparable, incomparable]
    set_labels = ('same', 'diff')  # , 'comparable', 'incomparable')
    venn3(subsets=subsets, set_labels=set_labels)
    plt.show()

    import wbia.plottool as pt

    pt.ensureqt()
    from matplotlib_subsets import treesets_rectangles

    tree = (
        (120, 'Same', None),
        [((50, 'comparable', None), []), ((50, 'incomparable', None), [])](
            120, 'Diff', None
        ),
        [((50, 'comparable', None), []), ((50, 'incomparable', None), [])],
    )

    treesets_rectangles(tree)
    plt.show()

    from matplotlib import pyplot as plt
    from matplotlib_venn import venn2, venn2_circles  # NOQA

    # Subset sizes
    s = (
        2,  # Ab
        3,  # aB
        1,  # AB
    )

    v = venn2(subsets=s, set_labels=('A', 'B'))

    # Subset labels
    v.get_label_by_id('10').set_text('A but not B')
    v.get_label_by_id('01').set_text('B but not A')
    v.get_label_by_id('11').set_text('A and B')

    # Subset colors
    v.get_patch_by_id('10').set_color('c')
    v.get_patch_by_id('01').set_color('#993333')
    v.get_patch_by_id('11').set_color('blue')

    # Subset alphas
    v.get_patch_by_id('10').set_alpha(0.4)
    v.get_patch_by_id('01').set_alpha(1.0)
    v.get_patch_by_id('11').set_alpha(0.7)

    # Border styles
    c = venn2_circles(subsets=s, linestyle='solid')
    c[0].set_ls('dashed')  # Line style
    c[0].set_lw(2.0)  # Line width

    plt.show()
    # plt.savefig('example_tree.pdf', bbox_inches='tight')
    # plt.close()

    # venn2(subsets=(25, 231+65, 8+15))

    # # Find out the location of the two circles
    # # (you can look up how its done in the first lines
    # # of the venn2 function)

    # from matplotlib_venn._venn2 import compute_venn2_areas, solve_venn2_circles
    # subsets = (25, 231+65, 8+15)
    # areas = compute_venn2_areas(subsets, normalize_to=1.0)
    # centers, radii = solve_venn2_circles(areas)

    # # Now draw the third circle.
    # # Its area is (15+65)/(25+8+15) times
    # # that of the first circle,
    # # hence its radius must be

    # r3 = radii[0]*sqrt((15+65.0)/(25+8+15))

    # # Its position must be such that the intersection
    # # area  with C1 is  15/(15+8+25) of C1's area.
    # # The way to compute the distance between
    # # the circles by area can be looked up in
    # # solve_venn2_circles

    # from matplotlib_venn._math import find_distance_by_area
    # distance = find_distance_by_area(radii[0], r3,
    #             15.0/(15+8+25)*np.pi*radii[0]*radii[0])
    # ax = gca()
    # ax.add_patch(Circle(centers[0] + np.array([distance, 0]),
    #              r3, alpha=0.5, edgecolor=None,
    #              facecolor='red', linestyle=None,
    #              linewidth=0))


def draw_inconsistent_pcc():
    """
    CommandLine:
        python -m wbia.scripts.specialdraw draw_inconsistent_pcc --show
    """
    from wbia.algo.graph import demo
    import wbia.plottool as pt
    import matplotlib as mpl
    from wbia.scripts.thesis import TMP_RC

    mpl.rcParams.update(TMP_RC)
    kwargs = dict(num_pccs=1, n_incon=1, p_incon=1, size=4)
    infr = demo.demodata_infr(**kwargs)
    infr.set_node_attrs('pos', {1: (30, 40), 3: (70, 40), 4: (0, 0), 2: (100, 0)})
    fnum = 1
    infr.set_node_attrs('pin', True)
    # infr.set_node_attrs('fixed_size', False)
    # infr.set_node_attrs('scale', .1)
    # infr.set_node_attrs('width', 16)
    infr.show(
        show_inconsistency=False,
        simple_labels=True,
        pickable=True,
        pnum=(1, 2, 1),
        fnum=fnum,
    )
    ax = pt.gca()
    truth_colors = infr._get_truth_colors()
    from wbia.algo.graph.state import POSTV, NEGTV

    pt.append_phantom_legend_label('positive', truth_colors[POSTV], ax=ax)
    pt.append_phantom_legend_label('negative', truth_colors[NEGTV], ax=ax)
    # pt.append_phantom_legend_label('incomparble', truth_colors[INCMP], ax=ax)
    pt.show_phantom_legend_labels(size=infr.graph.graph['fontsize'])
    ax.set_aspect('equal')

    infr.show(
        show_inconsistency=True,
        simple_labels=True,
        pickable=True,
        pnum=(1, 2, 2),
        fnum=fnum,
    )
    ax = pt.gca()
    truth_colors = infr._get_truth_colors()
    from wbia.algo.graph.state import POSTV, NEGTV

    pt.append_phantom_legend_label('positive', truth_colors[POSTV], ax=ax)
    pt.append_phantom_legend_label('negative', truth_colors[NEGTV], ax=ax)
    pt.append_phantom_legend_label('hypothesis', infr._error_color, ax=ax)
    # pt.append_phantom_legend_label('incomparble', truth_colors[INCMP], ax=ax)
    pt.show_phantom_legend_labels(size=infr.graph.graph['fontsize'])
    # ax.set_aspect('equal')
    ax.set_aspect('equal')
    ut.show_if_requested()


def draw_graph_id():
    """
    CommandLine:
        python -m wbia.scripts.specialdraw draw_graph_id --show
    """
    from wbia.algo.graph import demo
    import wbia.plottool as pt
    import matplotlib as mpl
    from wbia.scripts.thesis import TMP_RC

    mpl.rcParams.update(TMP_RC)
    kwargs = dict(
        num_pccs=5,
        p_incon=0,
        size=4,
        size_std=1,
        p_incomp=0.2,
        p_pair_neg=0.5,
        p_pair_incmp=0.4,
    )
    infr = demo.demodata_infr(**kwargs)
    infr.graph.graph['hpad'] = 50
    infr.graph.graph['vpad'] = 10
    infr.graph.graph['group_grid'] = True
    infr.show(
        show_inconsistency=False,
        simple_labels=True,
        wavy=False,
        groupby='name_label',
        pickable=True,
    )
    ax = pt.gca()
    truth_colors = infr._get_truth_colors()
    from wbia.algo.graph.state import POSTV, NEGTV, INCMP

    pt.append_phantom_legend_label('positive', truth_colors[POSTV], ax=ax)
    pt.append_phantom_legend_label('negative', truth_colors[NEGTV], ax=ax)
    pt.append_phantom_legend_label('incomparble', truth_colors[INCMP], ax=ax)
    pt.show_phantom_legend_labels(size=infr.graph.graph['fontsize'])
    ax.set_aspect('equal')
    ut.show_if_requested()


def redun_demo2():
    r"""
    python -m wbia.scripts.specialdraw redun_demo2 --show
    """
    from wbia.algo.graph.state import POSTV, NEGTV, INCMP  # NOQA
    from wbia.algo.graph import demo

    # from wbia.algo.graph import nx_utils
    import wbia.plottool as pt

    # import networkx as nx
    pt.ensureqt()
    import matplotlib as mpl
    from wbia.scripts.thesis import TMP_RC

    mpl.rcParams.update(TMP_RC)

    fnum = 1
    showkw = dict(
        show_inconsistency=False,
        show_labels=True,
        simple_labels=True,
        show_recent_review=False,
        wavy=False,
        groupby='name_label',
        splines='spline',
        pickable=True,
        fnum=fnum,
    )

    graphkw = dict(hpad=50, vpad=50, group_grid=True)
    pnum_ = pt.make_pnum_nextgen(2, 3)

    def show_redun(infr):
        infr.graph.graph.update(graphkw)
        infr.show(pnum=pnum_(), **showkw)
        ax = pt.gca()
        ax.set_aspect('equal')
        ccs = list(infr.positive_components())
        if len(ccs) == 1:
            cc = ccs[0]
            ax.set_xlabel(str(infr.pos_redundancy(cc)) + '-positive-redundant')
        else:
            cc1, cc2 = ccs
            ax.set_xlabel(str(infr.neg_redundancy(cc1, cc2)) + '-negative-redundant')

    infr = demo.make_demo_infr(ccs=[(1, 2, 3, 5, 4), (6,)])
    infr.add_feedback((5, 6), evidence_decision=POSTV)
    # infr.add_feedback((3, 4), evidence_decision='unreviewed')
    show_redun(infr)

    infr = infr.copy()
    for u, v in infr.find_pos_augment_edges(set(infr.graph.nodes()), k=2):
        infr.add_feedback((u, v), evidence_decision=POSTV)
    show_redun(infr)

    infr = infr.copy()
    for u, v in infr.find_pos_augment_edges(set(infr.graph.nodes()), k=3):
        infr.add_feedback((u, v), evidence_decision=POSTV)
    show_redun(infr)

    infr = demo.make_demo_infr(ccs=[(1, 2, 3, 4), (11, 12, 13, 14, 15)])
    infr.add_feedback((2, 11), evidence_decision=NEGTV)
    show_redun(infr)

    infr = demo.make_demo_infr(ccs=[(1, 2, 3, 4), (11, 12, 13, 14, 15)])
    infr.add_feedback((2, 11), evidence_decision=NEGTV)
    infr.add_feedback((4, 14), evidence_decision=NEGTV)
    show_redun(infr)

    infr = demo.make_demo_infr(ccs=[(1, 2, 3, 4), (11, 12, 13, 14, 15)])
    infr.add_feedback((2, 11), evidence_decision=NEGTV)
    infr.add_feedback((4, 14), evidence_decision=NEGTV)
    infr.add_feedback((2, 14), evidence_decision=NEGTV)
    show_redun(infr)

    fig = pt.gcf()
    fig.set_size_inches(10, 5)

    ut.show_if_requested()


def redun_demo3():
    r"""
    python -m wbia.scripts.specialdraw redun_demo3 --show
    python -m wbia.scripts.specialdraw redun_demo3 --saveparts=~/slides/incon_redun.jpg --dpi=300
    """
    from wbia.algo.graph.state import POSTV, NEGTV, INCMP  # NOQA
    from wbia.algo.graph import demo
    from wbia.algo.graph import nx_utils as nxu
    import wbia.plottool as pt

    # import networkx as nx
    pt.ensureqt()
    import matplotlib as mpl
    from wbia.scripts.thesis import TMP_RC

    mpl.rcParams.update(TMP_RC)

    fnum = 1
    showkw = dict(
        show_inconsistency=False,
        show_labels=True,
        simple_labels=True,
        show_recent_review=False,
        wavy=False,
        groupby='name_label',
        splines='spline',
        show_all=True,
        pickable=True,
        fnum=fnum,
    )

    graphkw = dict(hpad=50, vpad=50, group_grid=True)
    pnum_ = pt.make_pnum_nextgen(2, 1)

    infr = demo.make_demo_infr(ccs=[(1, 2, 3, 5, 4), (6,)])
    infr.add_feedback((5, 6), evidence_decision=POSTV)
    for e in nxu.complement_edges(infr.graph):
        infr.add_feedback(e, evidence_decision=INCMP)

    infr.graph.graph.update(graphkw)
    infr.show(pnum=pnum_(), **showkw)
    ax = pt.gca()
    ax.set_aspect('equal')

    ccs = [(1, 2, 3, 4), (11, 12, 13, 14, 15)]
    infr = demo.make_demo_infr(ccs=ccs)
    infr.add_feedback((4, 14), evidence_decision=NEGTV)
    import networkx as nx

    for e in nxu.edges_between(nx.complement(infr.graph), ccs[0], ccs[1]):
        print('e = %r' % (e,))
        infr.add_feedback(e, evidence_decision=INCMP)
    infr.graph.graph.update(graphkw)
    infr.show(pnum=pnum_(), **showkw)
    ax = pt.gca()
    ax.set_aspect('equal')

    fig = pt.gcf()
    fig.set_size_inches(10 / 3, 5)

    ut.show_if_requested()


def system_diagram():
    """
    CommandLine:
        python -m wbia.scripts.specialdraw system_diagram --show

    """
    from wbia.algo.graph.state import POSTV, NEGTV, INCMP, UNREV  # NOQA
    from wbia.algo.graph import demo
    from wbia.algo.graph import nx_utils as nxu  # NOQA
    import wbia.plottool as pt

    # import networkx as nx
    pt.ensureqt()
    import matplotlib as mpl
    from wbia.scripts.thesis import TMP_RC

    mpl.rcParams.update(TMP_RC)

    # fnum = 1
    # showkw = dict(show_inconsistency=False, show_labels=True,
    #               simple_labels=True,
    #               show_recent_review=False, wavy=False,
    #               groupby='name_label',
    #               splines='spline',
    #               show_all=True,
    #               pickable=True, fnum=fnum)

    # graphkw = dict(hpad=50, vpad=50, group_grid=True)
    # pnum_ = pt.make_pnum_nextgen(2, 1)

    infr = demo.demodata_infr(ccs=[(1, 2, 3, 4), (5, 6, 7), (8, 9,), (10,)])
    showkw = dict(
        show_unreviewed_edges=True,
        show_inferred_same=False,
        show_inferred_diff=False,
        show_labels=True,
        simple_labels=True,
        show_recent_review=False,
        reposition=False,
        pickable=True,
        outof=(len(infr.aids)),  # hack for colors
    )
    infr.clear_edges()

    # ----------------------
    # Step1: Find candidates
    # ----------------------
    infr.params['ranking.ntop'] = 4
    infr.refresh_candidate_edges()

    infr.update_visual_attrs(groupby='name_label')
    infr.set_node_attrs('pin', 'true')
    infr.set_node_attrs('shape', 'circle')

    infr.clear_feedback()
    infr.clear_name_labels()

    # infr.ensure_edges_from([(10, 5), (10, 6)])
    infr.ensure_prioritized(list(infr.edges()))
    edge_overrides = {}
    # edge_overrides = {
    #     # 'linestyle': {e: 'dashed' for e in infr.edges()},
    #     'linestyle': {e: 'dashed' for e in infr.get_edges_where_eq('decision', UNREV)},
    # }
    infr.show(edge_overrides=edge_overrides, fnum=1, pnum=(1, 4, 1), **showkw)
    pt.gca().set_aspect('equal')

    # ---------------------------
    # Step 2: Automatic decisions
    # ---------------------------
    infr.task_probs.pop('photobomb_state', None)
    infr.params['autoreview.enabled'] = True
    infr.params['autoreview.prioritize_nonpos'] = True
    infr.task_thresh['match_state'][POSTV] = 0.8
    infr.task_thresh['match_state'][NEGTV] = 0.54
    infr.task_thresh['match_state'][INCMP] = 0.5

    # infr.add_feedback((1, 2), POSTV)  # hack

    infr.ensure_prioritized(infr.get_edges_where_eq('decision', UNREV))
    gen = infr._inner_priority_gen()
    next(gen)

    # edge_overrides = {
    #     # 'linestyle': {e: 'dashed' for e in infr.edges()},
    #     'linestyle': {e: 'dashed' for e in infr.get_edges_where_eq('decision', UNREV)},
    # }
    infr.apply_nondynamic_update()
    infr.update_visual_attrs(groupby='name_label')
    infr.show(edge_overrides=edge_overrides, fnum=1, pnum=(1, 4, 2), **showkw)
    pt.gca().set_aspect('equal')

    # --------------
    # Error recovery
    # --------------
    possible = list(infr.find_pos_redun_candidate_edges())
    edge = possible[min(1, len(possible) - 1)]
    infr.add_feedback(edge, NEGTV)

    node_overrides = {
        'label': {n: '{}!'.format(n) for n in ut.flatten(infr.inconsistent_components())}
    }
    # edge_overrides = {
    #     'linestyle': {e: 'dashed' for e in infr.get_edges_where_eq('decision', UNREV)},
    # }
    infr.update_visual_attrs(groupby='name_label')
    infr.show(
        edge_overrides=edge_overrides,
        node_overrides=node_overrides,
        fnum=1,
        pnum=(1, 4, 3),
        **showkw,
    )
    pt.gca().set_aspect('equal')

    # Manual Decisions
    infr.init_simulation(oracle_accuracy=1.0)
    infr.params['redun.neg.only_auto'] = False
    infr.main_loop()

    # ISSUE:
    # For some reason a incomparable edge (3, 10) is being manually reviewed
    # again in the main loop even though it was already reviewed.
    # Quick Fix: add feedback specifically for this example.
    infr.add_feedback((3, 10), INCMP)

    # ISSUE:
    # When candidate edges are added within pos-redun CCs, the inferred state
    # should be set (but currently it is not).
    # EG: edge (1, 2) is added, but the CC is already pos-redun, but the
    # inferred state on the edge is never set.
    # Quick Fix: inference between newly added edges that were already
    # pos-redun
    infr.apply_nondynamic_update()

    # edge_overrides = {
    #     # 'linestyle': {e: 'dashed' for e in infr.edges()},
    #     'linestyle': {e: 'dashed' for e in infr.get_edges_where_eq('decision', UNREV)},
    # }
    infr.update_visual_attrs(groupby='name_label')
    infr.show(edge_overrides=edge_overrides, fnum=1, pnum=(1, 4, 4), **showkw)
    pt.gca().set_aspect('equal')

    ut.show_if_requested()


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.scripts.specialdraw
        python -m wbia.scripts.specialdraw --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
