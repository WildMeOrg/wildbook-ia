# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from plottool import interact_helpers as ih
from ibeis import viz
from plottool import draw_func2 as df2
from plottool import fig_presenter
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(__name__,
                                                       '[interact_bbox]', DEBUG=False)


#================
def iselect_bbox(ibs, gid, fnum=1,
                 figtitle='Image View - Select ANNOTATION (click two points)',
                 **kwargs):
    #from matplotlib.backend_bases import mplDeprecation
    print('[*interact] select_bbox(gid=%r, fnum=%r)' % (gid, fnum))
    print('[*interact] Define a Rectanglular ANNOTATION by clicking two points.')
    # Show the image
    fig = ih.begin_interaction('select_bbox', fnum)
    fig_presenter.bring_to_front(fig)
    viz.show_image(ibs, gid, **kwargs)
    try:
        viz.draw()
        fig = df2.gcf()
        pts = fig.ginput(2)
        print('[*guitools] ginput(2) = %r' % (pts,))
        [(x1, y1), (x2, y2)] = pts
        xm = min(x1, x2)
        xM = max(x1, x2)
        ym = min(y1, y2)
        yM = max(y1, y2)
        bbox = tuple(map(int, map(round, (xm, ym, xM - xm, yM - ym))))
        # Reconnect the old button press events
        print('[*interact] bbox = %r ' % (bbox,))
        return bbox
    except Exception as ex:
        print('<!!!>')
        print('[*interact] Caught: %s %s' % (type(ex), ex))
        print('[*interact] ANNOTATION selection Failed:')
        print('</!!!>')
        raise
