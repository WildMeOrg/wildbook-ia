from __future__ import absolute_import, division, print_function
import utool
import plottool as pt
import plottool.draw_func2 as df2
from plottool.viz_keypoints import _annotate_kpts
from plottool import viz_image2
from ibeis.viz import viz_helpers as vh
from ibeis.viz import viz_image
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[viz_chip]',
                                                       DEBUG=False)


#@utool.indent_func
def show_chip(ibs, aid, in_image=False, annote=True, title_suffix='',
                weight_label=None, weights=None, qreq_=None, **kwargs):
    """ Driver function to show chips

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

    Example:
        >>> from plottool.viz_keypoints import _annotate_kpts
        >>> from ibeis.viz.viz_chip import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb('PZ_Mothers')
        >>> aid = ibs.get_valid_aids()[0]
        >>> in_image = False
        >>> annote = True
        >>> kpts = ibs.get_annot_kpts(aid)
        >>> color = np.array([df2.ORANGE] * len(kpts))
        >>> kwargs = {'kpts': kpts, 'color': color}
        >>> show_chip(ibs, aid, in_image=in_image, annote=annote, **kwargs)
        >>> pt.show_if_requested()

    """
    # python -c "import utool; utool.print_auto_docstr('ibeis.viz.viz_chip', 'show_chip')
    printDBG('[viz] show_chip()')
    ibs.assert_valid_aids((aid,))
    #utool.embed()
    # Get chip
    chip = vh.get_chips(ibs, aid, in_image, qreq_=qreq_, **kwargs)
    # Create chip title
    chip_text = vh.get_annot_texts(ibs, [aid], **kwargs)[0]
    if kwargs.get('enable_chip_title_prefix', True):
        chip_title_text = chip_text + title_suffix
    else:
        chip_title_text = title_suffix
    chip_title_text = chip_title_text.strip('\n')
    # Draw chip
    fig, ax = df2.imshow(chip, **kwargs)
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
                    weights = ibs.get_annot_fgweights([aid], ensure=True, qreq_=qreq_)[0]
            if weights is not None:
                cmap_ = 'hot'
                #if weight_label == 'dstncvs':
                #    cmap_ = 'rainbow'
                color = df2.scores_to_color(weights, cmap_=cmap_, reverse_cmap=False)
                kwargs['color'] = color
                kwargs['ell_color'] = color
                kwargs['pts_color'] = color

        kpts_ = vh.get_kpts(ibs, aid, in_image, qreq_=qreq_, **kwargs)
        try:
            del kwargs['kpts']
        except KeyError:
            pass
        _annotate_kpts(kpts_, **kwargs)
        df2.upperleft_text(chip_text, color=kwargs.get('text_color', None))
    if not kwargs.get('ntitle', False):
        pt.set_title(chip_title_text)
    if in_image:
        gid = ibs.get_annot_gids(aid)
        aid_list = ibs.get_image_aids(gid)
        annotekw = viz_image.get_annot_annotations(ibs, aid_list, sel_aids=[aid])
        viz_image2.draw_image_overlay(ax, **annotekw)

    #if 'featweights' in vars() and 'color' in kwargs:
    if weights is not None and weight_label is not None:
        ## HACK HACK HACK
        if len(weights) > 0:
            cb = df2.colorbar(weights, kwargs['color'])
            cb.set_label(weight_label)
    return fig, ax


if __name__ == '__main__':
    """
    CommandLine:
         python ibeis/viz/viz_chip.py
    """
    #from plottool.viz_keypoints import _annotate_kpts
    from ibeis.viz.viz_chip import *  # NOQA
    import ibeis
    ibs = ibeis.opendb('PZ_MTEST')
    aid = ibs.get_valid_aids()[0]
    in_image = False
    annote = True
    kpts = ibs.get_annot_kpts(aid)
    kwargs = {}
    #from ibeis.model.preproc import preproc_featweight
    #featweights = preproc_featweight.compute_fgweights(ibs, [aid])[-1]
    #color = featweights
    #import numpy as np
    # plot rf feature weights
    #detect_cfgstr = ibs.cfg.detect_cfg.get_cfgstr()
    #color = np.array([df2.ORANGE] * len(kpts))
    #color = np.array(np.random.rand(len(kpts), 3))
    #kwargs = {'kpt1s': [kpts], 'color': color}
    show_chip(ibs, aid, in_image=in_image, annote=annote, **kwargs)
    if not utool.get_argflag('--noshow'):
        execstr = df2.present()
        exec(execstr)
