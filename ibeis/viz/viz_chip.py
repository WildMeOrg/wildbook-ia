from __future__ import absolute_import, division, print_function
import utool as ut
import plottool as pt
from plottool import plot_helpers as ph
from ibeis.viz import viz_helpers as vh
from ibeis.viz import viz_image
(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[viz_chip]',
                                                       DEBUG=False)


def testdata_showchip():
    import ibeis
    ibs = ibeis.opendb(defaultdb='PZ_MTEST')
    aid_list = ut.get_argval('--aids', type_=list, default=None)
    if aid_list is None:
        aid_list = ibs.get_valid_aids()[0:4]
    weight_label = ut.get_argval('--weight_label', type_=str, default='fg_weights')
    annote = not ut.get_argflag('--no-annote')
    kwargs = dict(ori=False, weight_label=weight_label, annote=annote)
    ut.print_dict(kwargs)
    print(aid_list)
    return ibs, aid_list, kwargs


def show_many_chips(ibs, aid_list):
    r"""
    CommandLine:
        python -m ibeis.viz.viz_chip --test-show_many_chips
        python -m ibeis.viz.viz_chip --test-show_many_chips --show --db NNP_Master3 --aids=13276,14047,14489,14906,10194,10201,12656,10150,11002,15315,7191,13127,15591,12838,13970,14123,14167 --no-annote --dpath figures --save ~/latex/crall-candidacy-2015/figures/challengechips.jpg '--caption=challenging images'

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.viz.viz_chip import *  # NOQA
        >>> import numpy as np
        >>> in_image = False
        >>> ibs, aid_list, kwargs = testdata_showchip()
        >>> # execute function
        >>> show_many_chips(ibs, aid_list)
        >>> ut.show_if_requested()
    """
    if ut.VERBOSE:
        print('[viz] show_many_chips')
    config2_ = None
    in_image = False
    chip_list = vh.get_chips(ibs, aid_list, in_image=in_image, config2_=config2_)
    stacked_chips = pt.stack_image_recurse(chip_list, modifysize=True)
    pt.imshow(stacked_chips)


#@ut.indent_func
def show_chip(ibs, aid, in_image=False, annote=True, title_suffix='',
                weight_label=None, weights=None, config2_=None, **kwargs):
    r""" Driver function to show chips

    Args:
        ibs (IBEISController):
        aid (int): annotation rowid
        in_image (bool): displays annotation with the context of its source image
        annote (bool): enables overlay annoations
        title_suffix (str):

    Keywords:
        color (3/4-tuple, ndarray, or str): colors for keypoints

    CommandLine:
        python -m ibeis.viz.viz_chip --test-show_chip --show
        python -c "import utool as ut; ut.print_auto_docstr('ibeis.viz.viz_chip', 'show_chip')"
        python -m ibeis.viz.viz_chip --test-show_chip --show --db NNP_Master3 --aids 14047 --no-annote

    Example:
        >>> # VIZ_TEST
        >>> from ibeis.viz.viz_chip import *  # NOQA
        >>> import numpy as np
        >>> in_image = False
        >>> ibs, aid_list, kwargs = testdata_showchip()
        >>> aid = aid_list[0]
        >>> config2_ = None
        >>> show_chip(ibs, aid, in_image=in_image, **kwargs)
        >>> pt.show_if_requested()
    """
    if ut.VERBOSE:
        print('[viz] show_chip(aid=%r)' % (aid,))
    #ibs.assert_valid_aids((aid,))
    # Get chip
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
            #from ibeis.model.preproc import preproc_featweight
            #featweights = preproc_featweight.compute_fgweights(ibs, [aid])[0]
            if weight_label == 'fg_weights':
                if weights is None and ibs.has_species_detector(ibs.get_annot_species_texts(aid)):
                    weight_label = 'fg_weights'
                    weights = ibs.get_annot_fgweights([aid], ensure=True, config2_=config2_)[0]
            if weights is not None:
                cmap_ = 'hot'
                #if weight_label == 'dstncvs':
                #    cmap_ = 'rainbow'
                color = pt.scores_to_color(weights, cmap_=cmap_, reverse_cmap=False)
                kwargs['color'] = color
                kwargs['ell_color'] = color
                kwargs['pts_color'] = color

        kpts_ = vh.get_kpts(ibs, aid, in_image, config2_=config2_,
                            kpts_subset=kwargs.get('kpts_subset', None),
                            kpts=kwargs.get('kpts', None))
        try:
            del kwargs['kpts']
        except KeyError:
            pass
        pt.viz_keypoints._annotate_kpts(kpts_, **kwargs)
        pt.upperleft_text(chip_text, color=kwargs.get('text_color', None))
    use_title = not kwargs.get('notitle', False)
    if use_title:
        pt.set_title(chip_title_text)
    if in_image:
        gid = ibs.get_annot_gids(aid)
        aid_list = ibs.get_image_aids(gid)
        annotekw = viz_image.get_annot_annotations(ibs, aid_list, sel_aids=[aid], draw_lbls=kwargs.get('draw_lbls', True))
        # Put annotation centers in the axis
        ph.set_plotdat(ax, 'annotation_bbox_list', annotekw['bbox_list'])
        ph.set_plotdat(ax, 'aid_list', aid_list)
        pt.viz_image2.draw_image_overlay(ax, **annotekw)

        zoom_ = ut.get_argval('--zoom', type_=float, default=None)
        if zoom_ is not None:
            # Zoom into the chip for some image context
            rotated_verts = ibs.get_annot_rotated_verts(aid)
            bbox = ibs.get_annot_bboxes(aid)
            print(bbox)
            print(rotated_verts)
            import vtool as vt
            rotated_bbox = vt.bbox_from_verts(rotated_verts)
            imgw, imgh = ibs.get_image_sizes(gid)

            pad_factor = zoom_
            pad_length = min(bbox[2], bbox[3]) * pad_factor
            minx = max(rotated_bbox[0] - pad_length, 0)
            miny = max(rotated_bbox[1] - pad_length, 0)
            maxx = min((rotated_bbox[0] + rotated_bbox[2]) + pad_length, imgw)
            maxy = min((rotated_bbox[1] + rotated_bbox[3]) + pad_length, imgh)

            #maxy = imgh - maxy
            #miny = imgh - miny

            ax = pt.gca()
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)
            ax.invert_yaxis()
    else:
        ph.set_plotdat(ax, 'chipshape', chip.shape)

    #if 'featweights' in vars() and 'color' in kwargs:
    if weights is not None and weight_label is not None:
        ## HACK HACK HACK
        if len(weights) > 0:
            cb = pt.colorbar(weights, kwargs['color'])
            cb.set_label(weight_label)
    return fig, ax


#if __name__ == '__main__':
#    """
#    CommandLine:
#         python ibeis/viz/viz_chip.py
#    """
#    #from plottool.viz_keypoints import _annotate_kpts
#    from ibeis.viz.viz_chip import *  # NOQA
#    import ibeis
#    ibs = ibeis.opendb('PZ_MTEST')
#    aid = ibs.get_valid_aids()[0]
#    in_image = False
#    annote = True
#    kpts = ibs.get_annot_kpts(aid)
#    kwargs = {}
#    #from ibeis.model.preproc import preproc_featweight
#    #featweights = preproc_featweight.compute_fgweights(ibs, [aid])[-1]
#    #color = featweights
#    #import numpy as np
#    # plot rf feature weights
#    #detect_cfgstr = ibs.cfg.detect_cfg.get_cfgstr()
#    #color = np.array([pt.ORANGE] * len(kpts))
#    #color = np.array(np.random.rand(len(kpts), 3))
#    #kwargs = {'kpt1s': [kpts], 'color': color}
#    show_chip(ibs, aid, in_image=in_image, annote=annote, **kwargs)
#    if not ut.get_argflag('--noshow'):
#        execstr = pt.present()
#        exec(execstr)
if __name__ == '__main__':
    """
    CommandLine:
        python -m ibeis.viz.viz_chip
        python -m ibeis.viz.viz_chip --allexamples
        python -m ibeis.viz.viz_chip --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
