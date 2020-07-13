# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import wbia.plottool as pt
from wbia.plottool import plot_helpers as ph
from wbia.viz import viz_helpers as vh
from wbia.viz import viz_image

(print, rrr, profile) = ut.inject2(__name__, '[viz_chip]')


def HARDCODE_SHOW_PB_PAIR():
    """
    python -m wbia.viz.viz_chip HARDCODE_SHOW_PB_PAIR --show

    Example:
        >>> # SCRIPT
        >>> from wbia.viz.viz_chip import *  # NOQA
        >>> import wbia.plottool as pt
        >>> HARDCODE_SHOW_PB_PAIR()
        >>> pt.show_if_requested()
    """
    # TODO: generalize into testdata_annotmatches which filters ams propertly
    # Then a function to show these ams
    import wbia
    import wbia.viz

    has_any = ut.get_argval('--has_any', default=['photobomb'])
    index = ut.get_argval('--index', default=0)

    ibs = wbia.opendb(defaultdb='PZ_Master1')
    ams = ibs._get_all_annotmatch_rowids()
    tags = ibs.get_annotmatch_case_tags(ams)
    flags = ut.filterflags_general_tags(tags, has_any=has_any)
    selected_ams = ut.compress(ams, flags)
    aid_pairs = ibs.get_annotmatch_aids(selected_ams)
    aid1, aid2 = aid_pairs[index]
    import wbia.plottool as pt

    fnum = 1
    if ut.get_argflag('--match'):
        request = ibs.depc_annot.new_request('vsone', [aid1], [aid2])
        res_list2 = request.execute()
        match = res_list2[0]
        match.show_single_annotmatch(
            qreq_=request,
            vert=False,
            colorbar_=False,
            notitle=True,
            draw_lbl=False,
            draw_border=False,
        )
    else:
        chip1, chip2 = ibs.get_annot_chips([aid1, aid2])
        pt.imshow(chip1, pnum=(1, 2, 1), fnum=fnum)
        pt.imshow(chip2, pnum=(1, 2, 2), fnum=fnum)


def testdata_showchip():
    import wbia

    ibs = wbia.opendb(defaultdb='PZ_MTEST')
    aid_list = ut.get_argval(('--aids', '--aid'), type_=list, default=None)
    if aid_list is None:
        aid_list = ibs.get_valid_aids()[0:4]
    weight_label = ut.get_argval('--weight_label', type_=str, default='fg_weights')
    annote = not ut.get_argflag('--no-annote')
    kwargs = dict(ori=ut.get_argflag('--ori'), weight_label=weight_label, annote=annote)
    kwargs['notitle'] = ut.get_argflag('--notitle')
    kwargs['pts'] = ut.get_argflag('--drawpts')
    kwargs['ell'] = True or ut.get_argflag('--drawell')
    kwargs['ell_alpha'] = ut.get_argval('--ellalpha', default=0.4)
    kwargs['ell_linewidth'] = ut.get_argval('--ell_linewidth', default=2)
    kwargs['draw_lbls'] = ut.get_argval('--draw_lbls', default=True)
    print('kwargs = ' + ut.repr4(kwargs, nl=True))
    default_config = dict(wbia.algo.Config.FeatureWeightConfig().parse_items())
    cfgdict = ut.argparse_dict(default_config)
    print('[viz_chip.testdata] cfgdict = %r' % (cfgdict,))
    config2_ = cfgdict
    print('[viz_chip.testdata] aid_list = %r' % (aid_list,))
    return ibs, aid_list, kwargs, config2_


def show_many_chips(ibs, aid_list, config2_=None, fnum=None, pnum=None, vert=True):
    r"""
    CommandLine:
        python -m wbia.viz.viz_chip --test-show_many_chips
        python -m wbia.viz.viz_chip --test-show_many_chips --show
        python -m wbia.viz.viz_chip --test-show_many_chips --show --db NNP_Master3 --aids=13276,14047,14489,14906,10194,10201,12656,10150,11002,15315,7191,13127,15591,12838,13970,14123,14167 --no-annote --dpath figures --save ~/latex/crall-candidacy-2015/figures/challengechips.jpg '--caption=challenging images'

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.viz.viz_chip import *  # NOQA
        >>> import numpy as np
        >>> in_image = False
        >>> ibs, aid_list, kwargs, config2_ = testdata_showchip()
        >>> # execute function
        >>> show_many_chips(ibs, aid_list, config2_)
        >>> ut.show_if_requested()
    """
    if ut.VERBOSE:
        print('[viz] show_many_chips')
    in_image = False
    chip_list = vh.get_chips(ibs, aid_list, in_image=in_image, config2_=config2_)
    import vtool as vt

    stacked_chips = vt.stack_image_recurse(chip_list, modifysize=True, vert=vert)
    pt.imshow(stacked_chips, fnum=None, pnum=None)


# @ut.indent_func
def show_chip(
    ibs,
    aid,
    in_image=False,
    annote=True,
    title_suffix='',
    weight_label=None,
    weights=None,
    config2_=None,
    **kwargs
):
    r""" Driver function to show chips

    Args:
        ibs (wbia.IBEISController):
        aid (int): annotation rowid
        in_image (bool): displays annotation with the context of its source image
        annote (bool): enables overlay annoations
        title_suffix (str):
        weight_label (None): (default = None)
        weights (None): (default = None)
        config2_ (dict): (default = None)

    Kwargs:
        enable_chip_title_prefix, nokpts, kpts_subset, kpts, text_color,
        notitle, draw_lbls, show_aidstr, show_gname, show_name, show_nid,
        show_exemplar, show_num_gt, show_quality_text, show_viewcode, fnum,
        title, figtitle, pnum, interpolation, cmap, heatmap, data_colorbar,
        darken, update, xlabel, redraw_image, ax, alpha, docla, doclf,
        projection, pts, ell
        color (3/4-tuple, ndarray, or str): colors for keypoints

    CommandLine:
        python -m wbia.viz.viz_chip show_chip --show --ecc
        python -c "import utool as ut; ut.print_auto_docstr('wbia.viz.viz_chip', 'show_chip')"
        python -m wbia.viz.viz_chip show_chip --show --db NNP_Master3 --aids 14047 --no-annote
        python -m wbia.viz.viz_chip show_chip --show --db NNP_Master3 --aids 14047 --no-annote

        python -m wbia.viz.viz_chip show_chip --show --db PZ_MTEST --aid 1 --bgmethod=cnn
        python -m wbia.viz.viz_chip show_chip --show --db PZ_MTEST --aid 1 --bgmethod=cnn --scale_max=30

        python -m wbia.viz.viz_chip show_chip --show --db PZ_MTEST --aid 1 --ecc --draw_lbls=False --notitle --save=~/slides/lnbnn_query.jpg --dpi=300

    Example:
        >>> # VIZ_TEST
        >>> from wbia.viz.viz_chip import *  # NOQA
        >>> import numpy as np
        >>> import vtool as vt
        >>> in_image = False
        >>> ibs, aid_list, kwargs, config2_ = testdata_showchip()
        >>> aid = aid_list[0]
        >>> if True:
        >>>     import matplotlib as mpl
        >>>     from wbia.scripts.thesis import TMP_RC
        >>>     mpl.rcParams.update(TMP_RC)
        >>> if ut.get_argflag('--ecc'):
        >>>     kpts = ibs.get_annot_kpts(aid, config2_=config2_)
        >>>     weights = ibs.get_annot_fgweights([aid], ensure=True, config2_=config2_)[0]
        >>>     kpts = ut.random_sample(kpts[weights > .9], 200, seed=0)
        >>>     ecc = vt.get_kpts_eccentricity(kpts)
        >>>     scale = 1 / vt.get_scales(kpts)
        >>>     #s = ecc if config2_.affine_invariance else scale
        >>>     s = scale
        >>>     colors = pt.scores_to_color(s, cmap_='jet')
        >>>     kwargs['color'] = colors
        >>>     kwargs['kpts'] = kpts
        >>>     kwargs['ell_linewidth'] = 3
        >>>     kwargs['ell_alpha'] = .7
        >>> show_chip(ibs, aid, in_image=in_image, config2_=config2_, **kwargs)
        >>> pt.show_if_requested()
    """
    if ut.VERBOSE:
        print('[viz] show_chip(aid=%r)' % (aid,))
    # ibs.assert_valid_aids((aid,))
    # Get chip
    # print('in_image = %r' % (in_image,))
    chip = vh.get_chips(ibs, aid, in_image=in_image, config2_=config2_)
    # Create chip title
    chip_text = vh.get_annot_texts(ibs, [aid], **kwargs)[0]
    if kwargs.get('enable_chip_title_prefix', True):
        chip_title_text = chip_text + title_suffix
    else:
        chip_title_text = title_suffix
    chip_title_text = chip_title_text.strip('\n')
    # Draw chip
    fig, ax = pt.imshow(chip, **kwargs)
    # Populate axis user data
    vh.set_ibsdat(ax, 'viztype', 'chip')
    vh.set_ibsdat(ax, 'aid', aid)
    if annote and not kwargs.get('nokpts', False):
        # Get and draw keypoints
        if 'color' not in kwargs:
            if weight_label == 'fg_weights':
                if weights is None and ibs.has_species_detector(
                    ibs.get_annot_species_texts(aid)
                ):
                    weight_label = 'fg_weights'
                    weights = ibs.get_annot_fgweights(
                        [aid], ensure=True, config2_=config2_
                    )[0]
            if weights is not None:
                cmap_ = 'hot'
                # if weight_label == 'dstncvs':
                #    cmap_ = 'rainbow'
                color = pt.scores_to_color(weights, cmap_=cmap_, reverse_cmap=False)
                kwargs['color'] = color
                kwargs['ell_color'] = color
                kwargs['pts_color'] = color

        kpts_ = vh.get_kpts(
            ibs,
            aid,
            in_image,
            config2_=config2_,
            kpts_subset=kwargs.get('kpts_subset', None),
            kpts=kwargs.pop('kpts', None),
        )
        pt.viz_keypoints._annotate_kpts(kpts_, **kwargs)
        if kwargs.get('draw_lbls', True):
            pt.upperleft_text(chip_text, color=kwargs.get('text_color', None))
    use_title = not kwargs.get('notitle', False)
    if use_title:
        pt.set_title(chip_title_text)
    if in_image:
        gid = ibs.get_annot_gids(aid)
        aid_list = ibs.get_image_aids(gid)
        annotekw = viz_image.get_annot_annotations(
            ibs, aid_list, sel_aids=[aid], draw_lbls=kwargs.get('draw_lbls', True)
        )
        # Put annotation centers in the axis
        ph.set_plotdat(ax, 'annotation_bbox_list', annotekw['bbox_list'])
        ph.set_plotdat(ax, 'aid_list', aid_list)
        pt.viz_image2.draw_image_overlay(ax, **annotekw)

        zoom_ = ut.get_argval('--zoom', type_=float, default=None)
        if zoom_ is not None:
            import vtool as vt

            # Zoom into the chip for some image context
            rotated_verts = ibs.get_annot_rotated_verts(aid)
            bbox = ibs.get_annot_bboxes(aid)
            # print(bbox)
            # print(rotated_verts)
            rotated_bbox = vt.bbox_from_verts(rotated_verts)
            imgw, imgh = ibs.get_image_sizes(gid)

            pad_factor = zoom_
            pad_length = min(bbox[2], bbox[3]) * pad_factor
            minx = max(rotated_bbox[0] - pad_length, 0)
            miny = max(rotated_bbox[1] - pad_length, 0)
            maxx = min((rotated_bbox[0] + rotated_bbox[2]) + pad_length, imgw)
            maxy = min((rotated_bbox[1] + rotated_bbox[3]) + pad_length, imgh)

            # maxy = imgh - maxy
            # miny = imgh - miny

            ax = pt.gca()
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)
            ax.invert_yaxis()
    else:
        ph.set_plotdat(ax, 'chipshape', chip.shape)

    # if 'featweights' in vars() and 'color' in kwargs:
    if weights is not None and weight_label is not None:
        # # HACK HACK HACK
        if len(weights) > 0:
            cb = pt.colorbar(weights, kwargs['color'])
            cb.set_label(weight_label)
    return fig, ax


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.viz.viz_chip
        python -m wbia.viz.viz_chip --allexamples
        python -m wbia.viz.viz_chip --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
