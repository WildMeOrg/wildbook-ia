from __future__ import absolute_import, division, print_function
import utool
import plottool.draw_func2 as df2
from plottool.viz_keypoints import _annotate_kpts
from plottool import viz_image2
from ibeis.viz import viz_helpers as vh
from ibeis.viz import viz_image
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[viz_chip]',
                                                       DEBUG=False)


@utool.indent_func
def show_chip(ibs, aid, in_image=False, annote=True, title_suffix='', **kwargs):
    """ Driver function to show chips

        Args:
            ibs (IBEISController):
            aid (int): annotation rowid
            in_image (bool): displays annotation with the context of its source image
            annote (bool): enables overlay annoations
            title_suffix (str):

        Keywords:
            color (3/4-tuple, ndarray, or str): colors for keypoints

        Example:
            >>> from plottool.viz_keypoints import _annotate_kpts
            >>> from ibeis.viz.viz_chip import *  # NOQA
            >>> import ibeis
            >>> ibs = ibeis.opendb('PZ_Mothers')
            >>> aid = ibs.get_valid_aids()[0]
            >>> in_image = False
            >>> annote = True
            >>> kpts = ibs.get_annot_kpts(aid)
            >>> colors = np.array([df2.ORANGE] * len(kpts))
            >>> kwargs = {'kpts': kpts, 'color': colors}
            >>> show_chip(ibs, aid, in_image=in_image, annote=annote, **kwargs)

    """
    # python -c "import utool; utool.print_auto_docstr('ibeis.viz.viz_chip', 'show_chip')
    printDBG('[viz] show_chip()')
    vh.ibsfuncs.assert_valid_aids(ibs, (aid,))
    #utool.embed()
    # Get chip
    chip = vh.get_chips(ibs, aid, in_image, **kwargs)
    # Create chip title
    chip_text = vh.get_annot_texts(ibs, [aid], **kwargs)[0]
    chip_title_text = chip_text + title_suffix
    # Draw chip
    fig, ax = df2.imshow(chip, **kwargs)
    # Populate axis user data
    vh.set_ibsdat(ax, 'viztype', 'chip')
    vh.set_ibsdat(ax, 'aid', aid)
    if annote and not kwargs.get('nokpts', False):
        # Get and draw keypoints
        if 'colors' not in kwargs:
            from ibeis.model.preproc import preproc_featweight
            featweights = preproc_featweight.compute_fg_weights(ibs, [aid])[0]
            kwargs['colors'] = featweights
        kpts_ = vh.get_kpts(ibs, aid, in_image, **kwargs)
        try:
            del kwargs['kpts']
        except KeyError:
            pass
        _annotate_kpts(kpts_, **kwargs)
    df2.upperleft_text(chip_text, color=kwargs.get('text_color', None))
    if not kwargs.get('ntitle', False):
        ax.set_title(chip_title_text)
    if in_image:
        gid = ibs.get_annot_gids(aid)
        aid_list = ibs.get_image_aids(gid)
        annotekw = viz_image.get_annot_annotations(ibs, aid_list, sel_aids=[aid])
        viz_image2.draw_image_overlay(ax, **annotekw)


if __name__ == '__main__':
    #from plottool.viz_keypoints import _annotate_kpts
    from ibeis.viz.viz_chip import *  # NOQA
    import ibeis
    ibs = ibeis.opendb('PZ_Mothers')
    aid = ibs.get_valid_aids()[0]
    in_image = False
    annote = True
    kpts = ibs.get_annot_kpts(aid)

    from ibeis.model.preproc import preproc_featweight
    featweights = preproc_featweight.compute_fg_weights(ibs, [aid])[0]
    colors = featweights
    #import numpy as np
    # plot rf feature weights
    #detect_cfgstr = ibs.cfg.detect_cfg.get_cfgstr()
    #colors = np.array([df2.ORANGE] * len(kpts))
    #colors = np.array(np.random.rand(len(kpts), 3))
    kwargs = {'kpt1s': [kpts], 'color': colors}
    show_chip(ibs, aid, in_image=in_image, annote=annote, **kwargs)
    if not utool.get_argflag('--noshow'):
        execstr = df2.present()
        exec(execstr)
