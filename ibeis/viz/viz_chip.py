from __future__ import absolute_import, division, print_function
import utool
import plottool.draw_func2 as df2
from plottool.viz_keypoints import _annotate_kpts
from plottool import viz_image2
from . import viz_helpers as vh
from . import viz_image
(print, print_, printDBG, rrr, profile) = utool.inject(__name__, '[viz_chip]',
                                                       DEBUG=False)


@utool.indent_func
def show_chip(ibs, aid, in_image=False, annote=True, **kwargs):
    """ Driver function to show chips """
    printDBG('[viz] show_chip()')
    vh.ibsfuncs.assert_valid_aids(ibs, (aid,))
    #utool.embed()
    # Get chip
    chip = vh.get_chips(ibs, aid, in_image, **kwargs)
    # Create chip title
    chip_text = vh.get_annotation_texts(ibs, [aid], **kwargs)[0]
    # Draw chip
    fig, ax = df2.imshow(chip, **kwargs)
    # Populate axis user data
    vh.set_ibsdat(ax, 'viztype', 'chip')
    vh.set_ibsdat(ax, 'aid', aid)
    if annote and not kwargs.get('nokpts', False):
        # Get and draw keypoints
        kpts = vh.get_kpts(ibs, aid, in_image, **kwargs)
        _annotate_kpts(kpts, **kwargs)
    df2.upperleft_text(chip_text)
    if not kwargs.get('notitle', False):
        ax.set_title(chip_text)
    if in_image:
        gid = ibs.get_annotation_gids(aid)
        aid_list = ibs.get_image_aids(gid)
        annotekw = viz_image.get_annotation_annotations(ibs, aid_list, sel_aids=[aid])
        viz_image2.annotate_image(ax, **annotekw)
