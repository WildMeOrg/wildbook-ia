# -*- coding: utf-8 -*-
"""
helpers for painting on top of images for groundtruthing

References:
    http://stackoverflow.com/questions/22232812/drawing-on-image-with-matplotlib-and-opencv2-update-image
    http://stackoverflow.com/questions/34933254/force-matplotlib-to-block-in-a-pyqt-thread-process
    http://matplotlib.org/examples/user_interfaces/embedding_in_qt4.html
    http://stackoverflow.com/questions/22410663/block-qmainwindow-while-child-widget-is-alive-pyqt
    http://stackoverflow.com/questions/20289939/pause-execution-until-button-press
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import numpy as np
import vtool as vt
from plottool import abstract_interaction
import math
from six.moves import range, zip, input  # NOQA
ut.noinject('impaint')


PAINTER_BASE = abstract_interaction.AbstractInteraction


class PaintInteraction(PAINTER_BASE):
    """
    References:
        http://stackoverflow.com/questions/22232812/drawing-on-image-with-mpl

    CommandLine:
        python -m plottool.interact_impaint --exec-draw_demo --show
    """
    def __init__(self, img, **kwargs):
        super(PaintInteraction, self).__init__(**kwargs)
        import plottool as pt

        init_mask = kwargs.get('init_mask', None)

        if init_mask is None:
            mask = np.full(img.shape, 255, dtype=np.uint8)
        else:
            mask = init_mask

        self.ax = pt.gca()
        #self.ax.imshow(img, interpolation='nearest', alpha=1)
        #self.ax.imshow(mask, interpolation='nearest', alpha=0.6)
        pt.imshow(img, ax=self.ax, interpolation='nearest', alpha=1)
        pt.imshow(mask, ax=self.ax, interpolation='nearest', alpha=0.6)
        self.ax.grid(False)

        self.mask = mask
        self.img = img
        self.brush_size = 100
        self.bg_color = (255, 255, 255)
        self.fg_color = (0, 0, 0)
        self.background = None
        self._running = True

        self.last_stroke = None

        self.connect_callbacks()
        self.update_image()
        self.finished_callback = None

    def update_image(self):
        import plottool as pt
        #print('update_image')
        self.ax.images.pop()
        #self.ax.imshow(self.mask, interpolation='nearest', alpha=0.6)
        pt.imshow(self.mask, ax=self.ax, interpolation='nearest', alpha=0.6)
        self.draw()
        #self.do_blit()
        #self.update()
        #self.ax.imshow(vt.blend_images_multiply(self.img, self.mask))
        #self.ax.grid(False)
        #self.ax.set_xticks([])
        #self.ax.set_yticks([])

    def on_close(self, event=None):
        if self.finished_callback is not None:
            self.finished_callback(self.mask)
        super(PaintInteraction, self).on_close(event)
        self._running = False

    def do_blit(self):
        if self.debug > 3:
            print('[pt.impaint] do_blit')
        if self.background is  None:
            self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        else:
            self.fig.canvas.restore_region(self.background)
            pass
        self.fig.canvas.blit(self.ax.bbox)

    def on_draw(self, event):
        #print('on draw')
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

    def apply_stroke(self, x, y, color):
        if self.debug > 3:
            print('[pt.impaint] apply stroke')
        import cv2
        center = (x, y)
        radius = int(self.brush_size / 2)
        thickness = -1
        cv2.circle(self.mask, center, radius, color, thickness)
        if self.last_stroke is not None:
            if self.last_stroke[0] == color:
                old_center = self.last_stroke[1]
                line_thickness = int(self.brush_size)
                cv2.line(self.mask, center, old_center, color, line_thickness)
        self.last_stroke = (color, (x, y))

    def on_click_inside(self, event, ax):
        x = int(math.floor(event.xdata))
        y = int(math.floor(event.ydata))
        if(event.button == self.LEFT_BUTTON):
            self.apply_stroke(x, y, self.fg_color)
        if(event.button == self.RIGHT_BUTTON):
            self.apply_stroke(x, y, self.bg_color)
        self.update_image()
        #self.draw()
        #self.print_status()
        #update the image

    def on_scroll(self, event):
        self.brush_size = max(self.brush_size + event.step, 1)
        print('self.brush_size = %r' % (self.brush_size,))

    def on_drag_stop(self, event):
        self.last_stroke = None

    def on_drag_inside(self, event):
        #self.print_status()
        x = int(math.floor(event.xdata))
        y = int(math.floor(event.ydata))
        if(event.button == self.LEFT_BUTTON):
            self.apply_stroke(x, y, self.fg_color)
        elif(event.button == self.RIGHT_BUTTON):
            self.apply_stroke(x, y, self.bg_color)
        self.update_image()
        #self.do_blit()
        #self.draw()


def impaint_mask2(img, init_mask=None):
    """
        python -m plottool.interact_impaint --exec-draw_demo --show
    """
    if False:
        QT = False  # NOQA
        #if QT:
        #    from guitool import mpl_embed
        #    import guitool
        #    guitool.ensure_qapp()  # must be ensured before any embeding
        #    wgt = mpl_embed.QtAbstractMplInteraction()
        #    fig = wgt.fig
        #    ax = wgt.axes
        #else:
        #    fig = plt.figure(1)
        #    ax = plt.subplot(111)
        #if init_mask is None:
        #    mask = np.zeros(img.shape, np.uint8) + 255
        #else:
        #    mask = init_mask
        #ax.imshow(img, interpolation='nearest', alpha=1)
        #ax.imshow(mask, interpolation='nearest', alpha=0.6)
        #ax.grid(False)
        #ax.set_xticks([])
        #ax.set_yticks([])

        #pntr = _OldPainter(fig, ax, mask)
        #ax.set_title('Click on the image to draw. exit to finish')
        #print('Starting interaction')
        #if not QT:
        #    plt.show(block=True)
        #else:
        #    guitool.qtapp_loop(wgt, frequency=100, init_signals=True)
        #    wgt.show()
        ##input('hack to block... press enter when done')
    else:
        pntr = PaintInteraction(img, init_mask=init_mask)
        #pntr.show_page()
        plt.title('Click on the image to draw. exit to finish')
        print('Starting interaction')

        # Hacky code to block until the interaction is actually done
        pntr.show()
        import time
        from guitool.__PYQT__ import QtGui
        while pntr._running:
            QtGui.qApp.processEvents()
            time.sleep(0.05)
        #plt.show()
    print('Finished interaction')
    return pntr.mask


def draw_demo():
    r"""
    CommandLine:
        python -m plottool.interact_impaint --exec-draw_demo --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from plottool.interact_impaint import *  # NOQA
        >>> result = draw_demo()
        >>> print(result)
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> ut.show_if_requested()
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
        python -m plottool.interact_impaint
        python -m plottool.interact_impaint --allexamples
        python -m plottool.interact_impaint --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
