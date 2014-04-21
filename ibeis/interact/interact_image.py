from __future__ import absolute_import, division, print_function
import utool
from ibeis import viz
from ibeis.viz import viz_helpers as vh
from plottool import draw_func2 as df2
from . import interact_helpers as ih
(print, print_, printDBG, rrr, profile) = utool.inject(__name__,
                                                       '[interact_img]',
                                                       DEBUG=False)


@utool.indent_decor('[interact_img]')
def interact_image(ibs, gid, sel_rids=[], fnum=1, select_callback=None,
                   **kwargs):
    fig = ih.begin_interaction('image', fnum)
    #printDBG(utool.func_str(interact_image, [], locals()))
    kwargs['draw_lbls'] = kwargs.get('draw_lbls', True)

    def _image_view(sel_rids=sel_rids, **_kwargs):
        viz.show_image(ibs, gid, sel_rids, **_kwargs)
        df2.set_figtitle('Image View')

    # Create callback wrapper
    def _on_image_click(event):
        printDBG('[inter] clicked image')
        if ih.clicked_outside_axis(event):
            # Toggle draw lbls
            kwargs['draw_lbls'] = not kwargs.get('draw_lbls', True)
            _image_view(ibs, **kwargs)
        else:
            ax          = event.inaxes
            viztype     = vh.get_ibsdat(ax, 'viztype')
            roi_centers = vh.get_ibsdat(ax, 'roi_centers', default=[])
            printDBG(' roi_centers=%r' % roi_centers)
            printDBG(' viztype=%r' % viztype)
            if len(roi_centers) == 0:
                print(' ...no chips exist to click')
                return
            x, y = event.xdata, event.ydata
            # Find ROI center nearest to the clicked point
            rid_list = vh.get_ibsdat(ax, 'rid_list', default=[])
            centx, _dist = utool.nearest_point(x, y, roi_centers)
            rid = rid_list[centx]
            print(' ...clicked rid=%r' % rid)
            if select_callback is not None:
                select_callback(gid, [rid])
            else:
                _image_view(sel_rids=[rid])

        viz.draw()

    _image_view(**kwargs)
    viz.draw()
    ih.connect_callback(fig, 'button_press_event', _on_image_click)
