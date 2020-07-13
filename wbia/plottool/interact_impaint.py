# -*- coding: utf-8 -*-
"""
helpers for painting on top of images for groundtruthing

References:
    http://stackoverflow.com/questions/22232812/drawing-on-image-with-matplotlib-and-opencv-update-image
    http://stackoverflow.com/questions/34933254/force-matplotlib-to-block-in-a-pyqt-thread-process
    http://matplotlib.org/examples/user_interfaces/embedding_in_qt4.html
    http://stackoverflow.com/questions/22410663/block-qmainwindow-while-child-widget-is-alive-pyqt
    http://stackoverflow.com/questions/20289939/pause-execution-until-button-press
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import matplotlib.pyplot as plt
import numpy as np

try:
    import vtool as vt
except ImportError:
    pass
from wbia.plottool import abstract_interaction
import math
from six.moves import range, zip, input  # NOQA

ut.noinject('impaint')


PAINTER_BASE = abstract_interaction.AbstractInteraction


class PaintInteraction(PAINTER_BASE):
    """
    References:
        http://stackoverflow.com/questions/22232812/drawing-on-image-with-mpl

    CommandLine:
        python -m wbia.plottool.interact_impaint --exec-draw_demo --show
    """

    def __init__(self, img, **kwargs):
        super(PaintInteraction, self).__init__(**kwargs)
        init_mask = kwargs.get('init_mask', None)
        if init_mask is None:
            mask = np.full(img.shape, 255, dtype=np.uint8)
        else:
            mask = init_mask
        self.mask = mask
        self.img = img
        self.brush_size = 75
        import wbia.plottool as pt

        self.valid_colors1 = ut.odict(
            [
                # ('background', (255 * pt.BLACK).tolist()),
                ('scenery', (255 * pt.BLACK).tolist()),
                ('photobomb', (255 * pt.RED).tolist()),
            ]
        )
        self.valid_colors2 = ut.odict([('foreground', (255 * pt.WHITE).tolist())])
        self.color1_idx = 0
        self.color1 = self.valid_colors1['scenery']
        self.color2 = self.valid_colors2['foreground']
        self.background = None
        self.last_stroke = None
        self.finished_callback = None
        self._imshow_running = True

    def update_title(self):
        import wbia.plottool as pt

        key = (self.valid_colors1.keys())[self.color1_idx]
        pt.plt.title(
            'Click on the image to draw. exit to finish.\n'
            'Right click erases, scroll wheel resizes.'
            't changes current_color=%r' % (key,)
        )

    def static_plot(self, fnum=None, pnum=(1, 1, 1)):
        import wbia.plottool as pt

        self.ax = pt.gca()
        # self.ax.imshow(img, interpolation='nearest', alpha=1)
        # self.ax.imshow(mask, interpolation='nearest', alpha=0.6)
        pt.imshow(self.img, ax=self.ax, interpolation='nearest', alpha=1)
        pt.imshow(self.mask, ax=self.ax, interpolation='nearest', alpha=0.6)
        self.update_title()
        self.ax.grid(False)

    def update_image(self):
        import wbia.plottool as pt

        # print('update_image')
        self.ax.images.pop()
        # self.ax.imshow(self.mask, interpolation='nearest', alpha=0.6)
        pt.imshow(self.mask, ax=self.ax, interpolation='nearest', alpha=0.6)
        self.draw()
        # self.do_blit()
        # self.update()
        # self.ax.imshow(vt.blend_images_multiply(self.img, self.mask))
        # self.ax.grid(False)
        # self.ax.set_xticks([])
        # self.ax.set_yticks([])

    def on_close(self, event=None):
        if self.finished_callback is not None:
            self.finished_callback(self.mask)
        super(PaintInteraction, self).on_close(event)

    def do_blit(self):
        if self.debug > 3:
            print('[pt.impaint] do_blit')
        if self.background is None:
            self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        else:
            self.fig.canvas.restore_region(self.background)
            pass
        self.fig.canvas.blit(self.ax.bbox)

    def on_draw(self, event):
        # print('on draw')
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

    def apply_stroke(self, x, y, color):
        import cv2

        if self.debug > 3:
            print('[pt.impaint] apply stroke')
        center = (x, y)
        radius = int(self.brush_size / 2)
        thickness = -1
        color_bgr = color[0:3][::-1]
        cv2.circle(self.mask, center, radius, color_bgr, thickness)
        if self.last_stroke is not None:
            if self.last_stroke[0] == color:
                old_center = self.last_stroke[1]
                line_thickness = int(self.brush_size)
                cv2.line(self.mask, center, old_center, color_bgr, line_thickness)
        self.last_stroke = (color, (x, y))

    def on_click_inside(self, event, ax):
        x = int(math.floor(event.xdata))
        y = int(math.floor(event.ydata))
        if event.button == self.LEFT_BUTTON:
            self.apply_stroke(x, y, self.color1)
        if event.button == self.RIGHT_BUTTON:
            self.apply_stroke(x, y, self.color2)
        self.update_image()
        # self.draw()
        # self.print_status()

    def on_scroll(self, event):
        self.brush_size = max(self.brush_size + event.step, 1)
        print('self.brush_size = %r' % (self.brush_size,))

    def on_key_press(self, event):
        if event.key == 't':
            print('toggle color')
            self.color1_idx = (self.color1_idx + 1) % len(self.valid_colors1)
            key = (self.valid_colors1.keys())[self.color1_idx]
            self.color1 = self.valid_colors1[key]
            print('self.color1_idx = %r' % (self.color1_idx,))
            print('key = %r' % (key,))
            self.update_title()
            self.draw()

    def on_drag_stop(self, event):
        self.last_stroke = None

    def on_drag_inside(self, event):
        # self.print_status()
        x = int(math.floor(event.xdata))
        y = int(math.floor(event.ydata))
        if event.button == self.LEFT_BUTTON:
            self.apply_stroke(x, y, self.color1)
        elif event.button == self.RIGHT_BUTTON:
            self.apply_stroke(x, y, self.color2)
        self.update_image()
        # self.do_blit()
        # self.draw()


def impaint_mask2(img, init_mask=None):
    """
    python -m wbia.plottool.interact_impaint --exec-draw_demo --show
    """
    if False:
        QT = False  # NOQA
        # if QT:
        #    from wbia.guitool import mpl_embed
        #    import wbia.guitool
        #    guitool.ensure_qapp()  # must be ensured before any embeding
        #    wgt = mpl_embed.QtAbstractMplInteraction()
        #    fig = wgt.fig
        #    ax = wgt.axes
        # else:
        #    fig = plt.figure(1)
        #    ax = plt.subplot(111)
        # if init_mask is None:
        #    mask = np.zeros(img.shape, np.uint8) + 255
        # else:
        #    mask = init_mask
        # ax.imshow(img, interpolation='nearest', alpha=1)
        # ax.imshow(mask, interpolation='nearest', alpha=0.6)
        # ax.grid(False)
        # ax.set_xticks([])
        # ax.set_yticks([])

        # pstartntr = _OldPainter(fig, ax, mask)
        # ax.set_title('Click on the image to draw. exit to finish')
        # print('Starting interaction')
        # if not QT:
        #    plt.show(block=True)
        # else:
        #    guitool.qtapp_loop(wgt, frequency=100, init_signals=True)
        #    wgt.show()
        # # input('hack to block... press enter when done')
    else:
        pntr = PaintInteraction(img, init_mask=init_mask)
        # pntr.show_page()
        # print('Starting interaction')
        pntr.start()
        pntr.show()

        # Hacky code to block until the interaction is actually done
        # pntr.show()
        import time
        from wbia.guitool.__PYQT__ import QtGui as QtWidgets

        while pntr.is_running:
            QtWidgets.qApp.processEvents()
            time.sleep(0.05)
        # plt.show()
    print('Finished interaction')
    return pntr.mask


def draw_demo():
    r"""
    CommandLine:
        python -m wbia.plottool.interact_impaint --exec-draw_demo --show

    Example:
        >>> # SCRIPT
        >>> from wbia.plottool.interact_impaint import *  # NOQA
        >>> result = draw_demo()
        >>> print(result)
        >>> import wbia.plottool as pt
        >>> pt.show_if_requested()
    """
    fpath = ut.grab_test_imgpath('zebra.png')
    img = vt.imread(fpath)
    mask = impaint_mask2(img)
    print('mask = %r' % (mask,))
    print('mask.sum() = %r' % (mask.sum(),))
    if False:
        plt.imshow(vt.blend_images_multiply(img, mask))
        ax = plt.gca()
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])


if __name__ == '__main__':
    """
    CommandLine:
        python -m wbia.plottool.interact_impaint
        python -m wbia.plottool.interact_impaint --allexamples
        python -m wbia.plottool.interact_impaint --allexamples --noface --nosrc
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
