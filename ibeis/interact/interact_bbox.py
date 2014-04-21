from __future__ import absolute_import, division, print_function
from . import interact_helpers as ih
from ibeis import viz
from plottool import draw_func2 as df2
import utool
(print, print_, printDBG, rrr, profile) = utool.inject(__name__,
                                                       '[interact_bbox]', DEBUG=False)


#================
def select_bbox(ibs, gid, fnum=1,
                figtitle='Image View - Select ROI (click two points)',
                **kwargs):
    #from matplotlib.backend_bases import mplDeprecation
    print('[*interact] select_bbox(gid=%r, fnum=%r)' % (gid, fnum))
    print('[*interact] Define a Rectanglular ROI by clicking two points.')
    # Show the image
    fig = ih.begin_interaction('select_bbox', fnum)
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
        bbox = map(int, map(round, (xm, ym, xM - xm, yM - ym)))
        # Reconnect the old button press events
        print('[*interact] bbox = %r ' % (bbox,))
        return bbox
    except Exception as ex:
        print('<!!!>')
        print('[*interact] Caught: %s %s' % (type(ex), ex))
        print('[*interact] ROI selection Failed:')
        print('</!!!>')
        raise
