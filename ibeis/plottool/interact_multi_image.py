from __future__ import absolute_import, division, print_function
from matplotlib.widgets import Button
import matplotlib.pyplot as plt
from plottool import viz_image2
from plottool import draw_func2 as df2
from plottool import plot_helpers as ph
from plottool import interact_rois as irs
#import utool


class MultiImageInteraction(object):
    def __init__(self, img_list, nImgs=None, max_per_page=10, fnum=None):
        print("test")
        if nImgs is None:
            nImgs = len(img_list)
        if fnum is None:
            self.fnum = df2.next_fnum()
        self.nImgs = nImgs
        self.max_per_page = min(max_per_page, nImgs)
        self.current_index = 0
        self.img_iter = iter(img_list)
        self.display_next_page()



    def display_next_page(self, event=None):
        start_index = self.current_index
        nLeft    = self.nImgs - self.current_index
        if nLeft == 0:
            raise AssertionError('no more images to display')
        nDisplay = min(nLeft, self.max_per_page)
        nRows, nCols = ph.get_square_row_cols(nDisplay)
        print('[viz*] r=%r, c=%r' % (nRows, nCols))
        pnum_ = df2.get_pnum_func(nRows, nCols)
        fig = df2.figure(fnum=self.fnum, pnum=pnum_(0))
        fig.clf()
        fig.canvas.mpl_connect('pick_event', self.onpick)
        px = -1
        for px, img in enumerate(self.img_iter):
            print(px)
            _vizkw = {
                'fnum': self.fnum,
                'pnum': pnum_(px),
                'title': '',
                'bbox_list'  : [],
                'theta_list' : [],
                'sel_list'   : [],
                'label_list' : [],
            }
            #print(utool.dict_str(_vizkw))
            viz_image2.show_image(img, **_vizkw)
            #print('components: ', fig.get_children())
            if px + 1 >= nDisplay:
                break
        finish_index = start_index + px + 1
        self.current_index += px + 1
        df2.set_figtitle('Displaying (%d - %d) / %d' % (start_index + 1, finish_index, self.nImgs))
        figlist = fig.get_children()
        print('figlist size: ', len(figlist))

        """would rather do something like "for fig in figlist, if fig.name = "Axes"", but am unable to find the proper syntax.
        this sets all the images in the frame to respond to a mouseclick."""
        for x in range(1, len(figlist)-1):
            figlist[x].set_picker(True)
        # Define buttons


        self.next_ax = plt.axes([0.7, 0.05, 0.15, 0.075])
        self.next_but = Button(self.next_ax, 'next')
        self.next_but.on_clicked(self.display_next_page)
        fig.show()
        print('next')


    def onpick(self, event):
        print('test2')
        """Need to add ROI code"""
        verts_of_image_selected = None
        """Need to figure out how to get the img from the code above"""
        img = None
        irs.ROI_creator(img, verts_of_image_selected)
