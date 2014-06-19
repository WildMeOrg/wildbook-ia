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
def show_chip(ibs, rid, in_image=False, annote=True, **kwargs):
    """ Driver function to show chips """
    printDBG('[viz] show_chip()')
    vh.ibsfuncs.assert_valid_rids(ibs, (rid,))
    #utool.embed()
    # Get chip
    chip = vh.get_chips(ibs, rid, in_image, **kwargs)
    # Create chip title
    title_str = vh.get_chip_labels(ibs, rid, **kwargs)
    # Draw chip
    fig, ax = df2.imshow(chip, title=title_str, **kwargs)
    # Populate axis user data
    vh.set_ibsdat(ax, 'viztype', 'chip')
    vh.set_ibsdat(ax, 'rid', rid)
    if annote and not kwargs.get('nokpts', False):
        # Get and draw keypoints
        kpts = vh.get_kpts(ibs, rid, in_image, **kwargs)
        _annotate_kpts(kpts, **kwargs)
    if in_image:
        gid = ibs.get_roi_gids(rid)
        rid_list = ibs.get_image_rids(gid)
        annotekw = viz_image.get_roi_annotations(ibs, rid_list, sel_rids=[rid])
        viz_image2.annotate_image(ax, **annotekw)
