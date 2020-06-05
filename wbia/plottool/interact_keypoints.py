# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import six
from . import draw_func2 as df2
from wbia.plottool import plot_helpers as ph
from wbia.plottool import interact_helpers as ih
from wbia.plottool.viz_featrow import draw_feat_row
from wbia.plottool.viz_keypoints import show_keypoints
from wbia.plottool import abstract_interaction

(print, rrr, profile) = ut.inject2(__name__)


class KeypointInteraction(abstract_interaction.AbstractInteraction):
    r"""
    CommandLine:
        python -m wbia.plottool.interact_keypoints --exec-KeypointInteraction --show
        python -m wbia.plottool.interact_keypoints --exec-KeypointInteraction --show --fname=lena.png

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.plottool.interact_keypoints import *  # NOQA
        >>> import numpy as np
        >>> import wbia.plottool as pt
        >>> import utool as ut
        >>> import pyhesaff
        >>> import vtool as vt
        >>> kpts, vecs, imgBGR = pt.viz_keypoints.testdata_kpts()
        >>> ut.quit_if_noshow()
        >>> #pt.interact_keypoints.ishow_keypoints(imgBGR, kpts, vecs, ori=True, ell_alpha=.4, color='distinct')
        >>> pt.interact_keypoints.KeypointInteraction(imgBGR, kpts, vecs, ori=True, ell_alpha=.4, autostart=True)
        >>> pt.show_if_requested()
    """

    def __init__(self, chip, kpts, vecs, fnum=0, figtitle=None, **kwargs):
        self.chip = chip
        self.kpts = kpts
        self.vecs = vecs
        self.figtitle = figtitle
        self.mode = 0
        super(KeypointInteraction, self).__init__(**kwargs)

    def plot(self, fnum=None, pnum=(1, 1, 1), **kwargs):
        import wbia.plottool as pt

        fnum = pt.ensure_fnum(fnum)
        pt.figure(fnum=fnum, docla=True, doclf=True)
        show_keypoints(self.chip, self.kpts, fnum=fnum, pnum=pnum, **kwargs)
        if self.figtitle is not None:
            pt.set_figtitle(self.figtitle)

    def _select_ith_kpt(self, fx):
        print('[interact] viewing ith=%r keypoint' % fx)
        # Get the fx-th keypiont
        kp, sift = self.kpts[fx], self.vecs[fx]
        # Draw the image with keypoint fx highlighted
        self.plot(self.fnum, (2, 1, 1), sel_fx=fx)
        # Draw the selected feature
        nRows, nCols, px = (2, 3, 3)
        draw_feat_row(self.chip, fx, kp, sift, self.fnum, nRows, nCols, px, None)

    def on_click_outside(self, event):
        self.mode = (self.mode + 1) % 3
        ell = self.mode == 1
        pts = self.mode == 2
        print('... default kpts view mode=%r' % self.mode)
        self.plot(self.fnum, ell=ell, pts=pts)
        self.draw()

    def on_click_inside(self, event, ax):
        import wbia.plottool as pt

        viztype = ph.get_plotdat(ax, 'viztype', None)
        print('[ik] viztype=%r' % viztype)
        if viztype is None:
            pass
        elif viztype == 'keypoints':
            kpts = ph.get_plotdat(ax, 'kpts', [])
            if len(kpts) == 0:
                print('...nokpts')
            else:
                print('...nearest')
                x, y = event.xdata, event.ydata
                import vtool as vt

                fx = vt.nearest_point(x, y, kpts)[0]
                self._select_ith_kpt(fx)
        elif viztype == 'warped':
            hs_fx = ph.get_plotdat(ax, 'fx', None)
            if hs_fx is not None:
                kp = self.kpts[hs_fx]  # FIXME
                sift = self.vecs[hs_fx]
                df2.draw_keypoint_gradient_orientations(
                    self.chip, kp, sift=sift, mode='vec', fnum=pt.next_fnum()
                )
                pt.draw()
        elif viztype.startswith('colorbar'):
            pass
        else:
            print('...unhandled')
        self.draw()


def ishow_keypoints(chip, kpts, desc, fnum=0, figtitle=None, nodraw=False, **kwargs):
    """
    TODO: Depricate in favor of the class

    CommandLine:
        python -m wbia.plottool.interact_keypoints --test-ishow_keypoints --show
        python -m wbia.plottool.interact_keypoints --test-ishow_keypoints --show --fname zebra.png

    Example:
        >>> # DISABLE_DOCTEST
        >>> from wbia.plottool.interact_keypoints import *  # NOQA
        >>> import numpy as np
        >>> import wbia.plottool as pt
        >>> import utool as ut
        >>> import pyhesaff
        >>> import vtool as vt
        >>> kpts, vecs, imgBGR = pt.viz_keypoints.testdata_kpts()
        >>> ut.quit_if_noshow()
        >>> #pt.interact_keypoints.ishow_keypoints(imgBGR, kpts, vecs, ori=True, ell_alpha=.4, color='distinct')
        >>> pt.interact_keypoints.ishow_keypoints(imgBGR, kpts, vecs, ori=True, ell_alpha=.4)
        >>> pt.show_if_requested()
    """
    if isinstance(chip, six.string_types):
        import vtool as vt

        chip = vt.imread(chip)
    fig = ih.begin_interaction('keypoint', fnum)
    annote_ptr = [1]

    self = ut.DynStruct()  # MOVE TO A CLASS INTERACTION
    self.kpts = kpts
    vecs = desc
    self.vecs = vecs

    def _select_ith_kpt(fx):
        print('[interact] viewing ith=%r keypoint' % fx)
        # Get the fx-th keypiont
        kp, sift = kpts[fx], vecs[fx]
        # Draw the image with keypoint fx highlighted
        _viz_keypoints(fnum, (2, 1, 1), sel_fx=fx, **kwargs)  # MAYBE: remove kwargs
        # Draw the selected feature
        nRows, nCols, px = (2, 3, 3)
        draw_feat_row(chip, fx, kp, sift, fnum, nRows, nCols, px, None)

    def _viz_keypoints(fnum, pnum=(1, 1, 1), **kwargs):
        df2.figure(fnum=fnum, docla=True, doclf=True)
        show_keypoints(chip, kpts, fnum=fnum, pnum=pnum, **kwargs)
        if figtitle is not None:
            df2.set_figtitle(figtitle)

    def _on_keypoints_click(event):
        print('[viz] clicked keypoint view')
        if event is None or event.xdata is None or event.inaxes is None:
            annote_ptr[0] = (annote_ptr[0] + 1) % 3
            mode = annote_ptr[0]
            ell = mode == 1
            pts = mode == 2
            print('... default kpts view mode=%r' % mode)
            _viz_keypoints(fnum, ell=ell, pts=pts, **kwargs)  # MAYBE: remove kwargs
        else:
            ax = event.inaxes
            viztype = ph.get_plotdat(ax, 'viztype', None)
            print('[ik] viztype=%r' % viztype)
            if viztype == 'keypoints':
                kpts = ph.get_plotdat(ax, 'kpts', [])
                if len(kpts) == 0:
                    print('...nokpts')
                else:
                    print('...nearest')
                    x, y = event.xdata, event.ydata
                    import vtool as vt

                    fx = vt.nearest_point(x, y, kpts)[0]
                    _select_ith_kpt(fx)
            elif viztype == 'warped':
                hs_fx = ph.get_plotdat(ax, 'fx', None)
                # kpts = ph.get_plotdat(ax, 'kpts', [])
                if hs_fx is not None:
                    # Ugly. Interactions should be changed to classes.
                    kp = self.kpts[hs_fx]  # FIXME
                    sift = self.vecs[hs_fx]
                    df2.draw_keypoint_gradient_orientations(
                        chip, kp, sift=sift, mode='vec', fnum=df2.next_fnum()
                    )
            elif viztype.startswith('colorbar'):
                pass
                # Hack to get a specific scoring feature
                # sortx = self.fs.argsort()
                # idx = np.clip(int(np.round(y * len(sortx))), 0, len(sortx) - 1)
                # mx = sortx[idx]
                # (fx1, fx2) = self.fm[mx]
                # (fx1, fx2) = self.fm[mx]
                # print('... selected score at rank idx=%r' % (idx,))
                # print('... selected score with fs=%r' % (self.fs[mx],))
                # print('... resolved to mx=%r' % mx)
                # print('... fx1, fx2 = %r, %r' % (fx1, fx2,))
                # self.select_ith_match(mx)
            else:
                print('...unhandled')
        ph.draw()

    # Draw without keypoints the first time
    _viz_keypoints(fnum, **kwargs)  # MAYBE: remove kwargs
    ih.connect_callback(fig, 'button_press_event', _on_keypoints_click)
    if not nodraw:
        ph.draw()


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.plottool.interact_keypoints
        python -m wbia.plottool.interact_keypoints --allexamples
        python -m wbia.plottool.interact_keypoints --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
