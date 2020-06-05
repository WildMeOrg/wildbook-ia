# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import utool as ut
from wbia import viz
from wbia.viz import viz_helpers as vh
from wbia.plottool import draw_func2 as df2
from wbia.plottool import interact_helpers as ih

(print, print_, printDBG, rrr, profile) = ut.inject(
    __name__, '[interact_img]', DEBUG=False
)


# @ut.indent_func
def ishow_image(ibs, gid, sel_aids=[], fnum=None, select_callback=None, **kwargs):
    if ut.VERBOSE:
        print(ut.get_caller_name(range(9)))
        print('[interact_image] gid=%r fnum=%r' % (gid, fnum,))
    if fnum is None:
        fnum = df2.next_fnum()
    # TODO: change to class based structure
    self = ut.DynStruct()
    self.fnum = fnum

    fig = ih.begin_interaction('image', fnum)
    # printDBG(utool.func_str(interact_image, [], locals()))
    kwargs['draw_lbls'] = kwargs.get('draw_lbls', True)

    def _image_view(sel_aids=sel_aids, **_kwargs):
        try:
            viz.show_image(ibs, gid, sel_aids=sel_aids, fnum=self.fnum, **_kwargs)
            df2.set_figtitle('Image View')
        except TypeError as ex:
            ut.printex(ex, ut.repr2(_kwargs))
            raise

    # Create callback wrapper
    def _on_image_click(event):
        print('[inter] clicked image')
        if ih.clicked_outside_axis(event):
            # Toggle draw lbls
            kwargs['draw_lbls'] = not kwargs.get('draw_lbls', True)
            _image_view(**kwargs)
        else:
            ax = event.inaxes
            viztype = vh.get_ibsdat(ax, 'viztype')
            annotation_centers = vh.get_ibsdat(ax, 'annotation_centers', default=[])
            print(' annotation_centers=%r' % annotation_centers)
            print(' viztype=%r' % viztype)
            if len(annotation_centers) == 0:
                print(' ...no chips exist to click')
                return
            x, y = event.xdata, event.ydata
            # Find ANNOTATION center nearest to the clicked point
            aid_list = vh.get_ibsdat(ax, 'aid_list', default=[])
            import vtool as vt

            centx, _dist = vt.nearest_point(x, y, annotation_centers)
            aid = aid_list[centx]
            print(' ...clicked aid=%r' % aid)
            if select_callback is not None:
                # HACK, should just implement this correctly here
                select_callback(gid, sel_aids=[aid], fnum=self.fnum)
            else:
                _image_view(sel_aids=[aid])

        viz.draw()

    _image_view(**kwargs)
    viz.draw()
    ih.connect_callback(fig, 'button_press_event', _on_image_click)
