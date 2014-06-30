from __future__ import absolute_import, division, print_function
import utool
from ibeis import viz
from ibeis.viz import viz_helpers as vh
from plottool import draw_func2 as df2
from plottool import interact_helpers as ih
(print, print_, printDBG, rrr, profile) = utool.inject(__name__,
                                                       '[interact_img]',
                                                       DEBUG=False)


@utool.indent_func
def ishow_image(ibs, gid, sel_aids=[], fnum=1, select_callback=None,
                **kwargs):
    fig = ih.begin_interaction('image', fnum)
    #printDBG(utool.func_str(interact_image, [], locals()))
    kwargs['draw_lbls'] = kwargs.get('draw_lbls', True)

    def _image_view(sel_aids=sel_aids, **_kwargs):
        try:
            viz.show_image(ibs, gid, sel_aids=sel_aids, fnum=fnum, **_kwargs)
            df2.set_figtitle('Image View')
        except TypeError as ex:
            utool.printex(ex, utool.dict_str(_kwargs))
            raise

    # Create callback wrapper
    def _on_image_click(event):
        printDBG('[inter] clicked image')
        if ih.clicked_outside_axis(event):
            # Toggle draw lbls
            kwargs['draw_lbls'] = not kwargs.get('draw_lbls', True)
            _image_view(**kwargs)
        else:
            ax          = event.inaxes
            viztype     = vh.get_ibsdat(ax, 'viztype')
            annotion_centers = vh.get_ibsdat(ax, 'annotion_centers', default=[])
            printDBG(' annotion_centers=%r' % annotion_centers)
            printDBG(' viztype=%r' % viztype)
            if len(annotion_centers) == 0:
                print(' ...no chips exist to click')
                return
            x, y = event.xdata, event.ydata
            # Find ANNOTATION center nearest to the clicked point
            aid_list = vh.get_ibsdat(ax, 'aid_list', default=[])
            centx, _dist = utool.nearest_point(x, y, annotion_centers)
            aid = aid_list[centx]
            print(' ...clicked aid=%r' % aid)
            if select_callback is not None:
                select_callback(gid, sel_aids=[aid])
            else:
                _image_view(sel_aids=[aid])

        viz.draw()

    _image_view(**kwargs)
    viz.draw()
    ih.connect_callback(fig, 'button_press_event', _on_image_click)
