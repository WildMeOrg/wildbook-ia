from __future__ import absolute_import, division, print_function
import utool
from ibeis.view import viz
from ibeis.view.viz import viz_helpers as vh
from plottool import draw_func2 as df2
from . import interact_helpers
(print, print_, printDBG, rrr, profile) = utool.inject(__name__,
                                                       '[interact_img]',
                                                       DEBUG=False)


@utool.indent_func
def interact_image(ibs, gid, sel_rids=[], fnum=1,
                   select_rid_callback=None,
                   **kwargs):
    fig = interact_helpers.begin_interaction('image', fnum)
    kwargs['draw_lbls'] = kwargs.get('draw_lbls', True)

    def _image_view(**_kwargs):
        viz.show_image(ibs, gid, sel_rids, **_kwargs)
        df2.set_figtitle('Image View')

    # Create callback wrapper
    def _on_image_click(event):
        print('[inter] clicked image')
        if interact_helpers.clicked_outside_axis(event):
            # Toggle draw lbls
            kwargs['draw_lbls'] = not kwargs.get('draw_lbls', True)
            _image_view(ibs, **kwargs)
        else:
            ax          = event.inaxes
            viztype     = vh.get_ibsdat(ax, 'viztype')
            roi_centers = vh.get_ibsdat(ax, 'roi_centers', default=[])
            print(' roi_centers=%r' % roi_centers)
            print(' viztype=%r' % viztype)
            if len(roi_centers) == 0:
                print(' ...no chips exist to click')
                return
            x, y = event.xdata, event.ydata
            # Find ROI center nearest to the clicked point
            rid_list = vh.get_ibsdat(ax, 'rid_list', default=[])
            centx, _dist = utool.nearest_point(x, y, roi_centers)
            rid = rid_list[centx]
            print(' ...clicked rid=%r' % rid)
            if select_rid_callback is not None:
                select_rid_callback(gid, sel_rids=[rid])
        viz.draw()

    _image_view(**kwargs)
    viz.draw()
    interact_helpers.connect_callback(fig, 'button_press_event', _on_image_click)
