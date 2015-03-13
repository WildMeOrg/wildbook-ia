from __future__ import absolute_import, division, print_function
import utool as ut
import plottool as pt
from ibeis.viz import viz_helpers as vh
from ibeis.viz import viz_image
(print, print_, printDBG, rrr, profile) = ut.inject(__name__, '[viz_chip]',
                                                       DEBUG=False)


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

    Example:
        >>> # VIZ_TEST
        >>> from ibeis.viz.viz_chip import *  # NOQA
        >>> import ibeis
        >>> import numpy as np
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> aid = ibs.get_valid_aids()[0]
        >>> in_image = False
        >>> annote = True
        >>> config2_ = None
        >>> #kpts = ibs.get_annot_kpts(aid, config2_=config2_)[::100]
        >>> #kpts = None
        >>> #color = np.array([pt.ORANGE] * len(kpts))
        >>> #kwargs = dict(kpts=kpts, color=color)
        >>> kwargs = dict(ori=True)
        >>> show_chip(ibs, aid, in_image=in_image, annote=annote, **kwargs)
        >>> pt.show_if_requested()
    """
    if ut.VERBOSE:
        print('[viz] show_chip(aid=%r)' % (aid,))
    ibs.assert_valid_aids((aid,))
    #ut.embed()
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
    if not kwargs.get('ntitle', False):
        pt.set_title(chip_title_text)
    if in_image:
        gid = ibs.get_annot_gids(aid)
        aid_list = ibs.get_image_aids(gid)
        annotekw = viz_image.get_annot_annotations(ibs, aid_list, sel_aids=[aid])
        pt.viz_image2.draw_image_overlay(ax, **annotekw)

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
