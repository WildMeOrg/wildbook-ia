# -*- coding: utf-8 -*-
import math


class _OldPainter(object):
    """
    References:
        http://stackoverflow.com/questions/22232812/drawing-on-image-with-matplotlib-and-opencv2-update-image
    """

    def __init__(self, fig, ax, mask):
        self.button_pressed = False
        self.mask = mask
        self.brush_size = 50
        self.ax = ax
        self.fig = fig
        self.color = 0
        self.background = None

        canvas = self.fig.canvas
        canvas.mpl_connect('button_press_event', self.button_press_callback)
        canvas.mpl_connect('button_release_event', self.button_release_callback)
        canvas.mpl_connect('motion_notify_event', self.on_move)
        canvas.mpl_connect('draw_event', self.draw_callback)

    def draw(self):
        self.fig.canvas.draw()
        # print('draw')

    def do_blit(self):
        if self.background is None:
            self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        else:
            self.fig.canvas.restore_region(self.background)
            pass
        self.fig.canvas.blit(self.ax.bbox)

    def draw_callback(self, event):
        self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

    def button_press_callback(self, event):
        import cv2

        if event.button == 1:
            self.button_pressed = True
            try:
                x = int(math.floor(event.xdata))
                y = int(math.floor(event.ydata))
                cv2.circle(
                    self.mask,
                    (x, y),
                    int(self.brush_size / 2),
                    (self.color, self.color, self.color),
                    -1,
                )
            except TypeError:
                pass
        self.update_image()
        # update the image
        self.do_blit()

    def update_image(self):
        self.ax.images.pop()
        self.ax.imshow(self.mask, interpolation='nearest', alpha=0.6)

    def button_release_callback(self, event):
        self.button_pressed = False
        self.update_image()
        self.draw()
        # cv2.imwrite('test.png', self.mask)
        self.do_blit()

    def on_move(self, event):
        import cv2

        if self.button_pressed:
            try:
                x = int(math.floor(event.xdata))
                y = int(math.floor(event.ydata))
                cv2.circle(
                    self.mask,
                    (x, y),
                    int(self.brush_size / 2),
                    (self.color, self.color, self.color),
                    -1,
                )
            except TypeError:
                pass
            self.update_image()
        self.draw()
        self.do_blit()
