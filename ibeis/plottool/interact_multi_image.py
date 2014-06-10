from __future__ import absolute_import, division, print_function
from matplotlib.widgets import Button
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
from plottool import viz_image2
from plottool import interact_rois
from plottool import draw_func2 as df2
from plottool import plot_helpers as ph
from plottool import interact_helpers as ih
import utool
from vtool import image as gtool
#import utool


class MultiImageInteraction(object):
    def __init__(self, gpath_list, max_per_page=4, bboxes_list=None, thetas_list=None, verts_list=None, gid_list=None,
                 nImgs=None, fnum=None):
        print('Creating multi-image interaction')

    #def __init__(self, img_list, nImgs=None, gid_list=None, rids_list=None, bboxes_list=None, max_per_page=10,fnum=None):
        print('maX ', max_per_page)
        if nImgs is None:
            nImgs = len(gpath_list)
        if fnum is None:
            self.fnum = df2.next_fnum()
        if bboxes_list is None:
            bboxes_list = [[]] * nImgs
        if thetas_list is None:
            thetas_list = [[0] * len(bboxes) for bboxes in bboxes_list]
        # How many images we are showing and per page
        self.thetas_list = thetas_list
        self.bboxes_list = bboxes_list
        if gid_list is None:
            self.gid_list = None
        else:
            self.gid_list = gid_list

        self.nImgs = nImgs
        self.max_per_page = min(max_per_page, nImgs)
        self.current_index = 0
        self.page_number = -1
        # Initialize iterator over the image paths
        self.gpath_list = gpath_list
        # Display the first page
        self.first_load = True
        self.display_next_page()

    def display_next_page(self, event=None):

        nLeft = self.nImgs - self.current_index
        if(nLeft <= 0):
            print('No more images to display')
            return
        self.page_number = self.page_number + 1

        if nLeft == 0:
            fig = df2.figure(fnum=self.fnum, pnum=(1, 1, 1))
            fig.clf()
            return False
            raise AssertionError('no more images to display')
        nDisplay = min(nLeft, self.max_per_page)
        self.nDisplay = nDisplay
        nRows, nCols = ph.get_square_row_cols(nDisplay)
        print('[viz*] r=%r, c=%r' % (nRows, nCols))
        pnum_ = df2.get_pnum_func(nRows, nCols)
        # Clear the figure for the new page of data
        fig = df2.figure(fnum=self.fnum, pnum=pnum_(0))
        fig.clf()
        # Draw the new page of data
        px = -1  # plot-index
        start_index = self.current_index
        end_index   = start_index + nDisplay
        """this is so, on reaching final page, if you hit prev_button you
        return to the same images as were on the last page before"""
        self.current_index = self.current_index + self.max_per_page
        for px, index in enumerate(xrange(start_index, end_index)):
            gpath      = self.gpath_list[index]
            bbox_list  = self.bboxes_list[index]
            #print('bbox_list=%r in display for px=%r' % (bbox_list, px))
            theta_list = self.thetas_list[index]
            img = gtool.imread(gpath)
            label_list = [ix + 1 for ix in xrange(len(bbox_list))]
            sel_list = [True for ix in xrange(len(bbox_list))]
            #Add true values for every bbox to display
            for i in range(0, len(bbox_list)):
                sel_list.append(True)
            _vizkw = {
                'fnum': self.fnum,
                'pnum': pnum_(px),
                'title': str(index),
                'bbox_list'  : bbox_list,
                'theta_list' : theta_list,
                'sel_list'   : sel_list,
                'label_list' : label_list,
            }
            #print(utool.dict_str(_vizkw))
            #print('vizkw = ' + utool.dict_str(_vizkw))
            _, ax = viz_image2.show_image(img, **_vizkw)
            ph.set_plotdat(ax, 'px', str(px))
            ph.set_plotdat(ax, 'bbox_list', bbox_list)
            ph.set_plotdat(ax, 'theta_list', theta_list)
            ph.set_plotdat(ax, 'title', str(index))
            ph.set_plotdat(ax, 'gpath', gpath)
            #print('components: ', fig.get_children())
            if px + 1 >= nDisplay:
                break

        # Set the figure title
        df2.set_figtitle('Displaying (%d - %d) / %d' % (start_index + 1, end_index, self.nImgs))

        if self.first_load is True:
            self.first_load = False
            ih.connect_callback(fig, 'button_press_event', self.on_figure_clicked)
            ih.connect_callback(fig, 'key_press_event', self.key_press_callback)

        self.display_buttons()
        # Connect the callback whenever the figure is clicked

        # Show the changes
        fig.show()
        plt.draw()
        print('next')

    def display_prev_page(self, event=None):
        if(self.current_index <= self.max_per_page):
            print('at top of list.')
            return
        self.page_number = self.page_number - 1
        'update our current index instantly, in case of multiple button clicks at once'
        end_index   = self.current_index - self.max_per_page
        start_index = self.current_index - (2 * self.max_per_page)
        self.current_index = end_index
        # if nLeft == 0:
        #     fig = df2.figure(fnum=self.fnum, pnum=(1, 1, 1))
        #     fig.clf()
        #     return False
        #     raise AssertionError('no more images to display')
        nDisplay = self.max_per_page
        self.nDisplay = nDisplay
        nRows, nCols = ph.get_square_row_cols(nDisplay)
        print('[viz*] r=%r, c=%r' % (nRows, nCols))
        pnum_ = df2.get_pnum_func(nRows, nCols)
        # Clear the figure for the new page of data
        fig = df2.figure(fnum=self.fnum, pnum=pnum_(0))
        fig.clf()
        # Draw the new page of data
        px = -1  # plot-index

        for px, index in enumerate(xrange(start_index, end_index)):
            gpath      = self.gpath_list[index]
            bbox_list  = self.bboxes_list[index]
            print('bbox_list %r in display for px: %r ' % (bbox_list, px))
            theta_list = self.thetas_list[index]
            img = gtool.imread(gpath)
            label_list = []
            for i in range(0, len(bbox_list)):
                label_list.append(i + 1)

            sel_list = []
            #Add true values for every bbox to display
            for i in range(0, len(bbox_list)):
                sel_list.append(True)
            _vizkw = {
                'fnum': self.fnum,
                'pnum': pnum_(px),
                #title should always be the image number
                'title': str(index),
                'bbox_list'  : bbox_list,
                'theta_list' : theta_list,
                'sel_list'   : sel_list,
                'label_list' : label_list,
            }
            #print(utool.dict_str(_vizkw))
            print('vizkw = ' + utool.dict_str(_vizkw))
            _, ax = viz_image2.show_image(img, **_vizkw)
            ph.set_plotdat(ax, 'px', str(px))
            ph.set_plotdat(ax, 'title', str(index))
            ph.set_plotdat(ax, 'bbox_list', bbox_list)
            ph.set_plotdat(ax, 'gpath', gpath)
            #print('components: ', fig.get_children())
            if px + 1 >= nDisplay:
                break

        # Set the figure title
        df2.set_figtitle('Displaying (%d - %d) / %d' % (start_index + 1, end_index, self.nImgs))

        #ih.connect_callback(fig, 'button_press_event', self.on_figure_clicked)

        self.display_buttons()
        # Show the changes
        plt.draw()
        fig.show()
        print('next')

    def display_buttons(self):
        # Create the button for scrolling forwards
        self.next_ax = plt.axes([0.75, 0.025, 0.15, 0.075])
        self.next_but = Button(self.next_ax, 'next')
        self.next_but.on_clicked(self.display_next_page)

        # Create the button for scrolling backwards
        self.prev_ax = plt.axes([0.1, .025, 0.15, 0.075])
        self.prev_but = Button(self.prev_ax, 'prev')
        self.prev_but.on_clicked(self.display_prev_page)
        # Connect the callback whenever the figure is clicked

    def update_images(self, img_ind, updated_bbox_list, updated_theta_list):
        """Insert code for viz_image2 redrawing here"""
        print('update called')
        index = int(img_ind)
        print('index: %r' % index)
        print('Images bbox before: %r' % (self.bboxes_list[index],))
        self.bboxes_list[index] = updated_bbox_list
        self.thetas_list[index] = updated_theta_list
        print('Images bbox after: %r' % (self.bboxes_list[index],))

        nRows, nCols = ph.get_square_row_cols(self.nDisplay)
        pnum_ = df2.get_pnum_func(nRows, nCols)
        gpath = self.gpath_list[index]
        bbox_list  = self.bboxes_list[index]
        theta_list = self.thetas_list[index]
        px = index % self.max_per_page
        label_list = []
        for i in range(0, len(bbox_list)):
            label_list.append(i + 1)
        sel_list = []
        for i in range(0, len(bbox_list)):
                print('image has a bbox')
                sel_list.append(True)
        img = gtool.imread(gpath)
        _vizkw = {
            'fnum': self.fnum,
            'pnum': pnum_(px),
            'title': str(index),
            'bbox_list'  : bbox_list,
            'theta_list' : theta_list,
            'sel_list'   : sel_list,
            'label_list' : label_list,
        }
        #print(utool.dict_str(_vizkw))
        _, ax = viz_image2.show_image(img, **_vizkw)
        plt.draw()

    def on_figure_clicked(self, event):
        #don't do other stuff if we clicked a button
        point = (event.x, event.y)
        if self.next_ax.contains_point(point) or self.prev_ax.contains_point(point):
            print('in button click')
            return

        if ih.clicked_inside_axis(event):
            ax = event.inaxes
            image_number = int(ph.get_plotdat(ax, 'title'))
            #bbox_list  = ph.get_plotdat(ax, 'bbox_list')
            bbox_list = self.bboxes_list[image_number]
            print('Bbox of figure: %r' % (bbox_list,))
            theta_list = self.thetas_list[image_number]
            print('theta_list = %r' % (theta_list,))
            gpath      = ph.get_plotdat(ax, 'gpath')
            #img = mpimg.imread(gpath)
            img = gtool.imread(gpath)
            fnum = df2.next_fnum()
            mc = interact_rois.ROIInteraction(img, image_number, self.update_images, bbox_list=bbox_list, theta_list=theta_list, fnum=fnum)
            self.mc = mc
            # """wait for accept
            # have a flag to tell if a bbox has been changed, on the bbox list that is brought it"
            # on accept:
            # viz_image2.show_image callback
            # """
            #plt.show()
            df2.update()
            print('Clicked: ax: num=%r' % image_number)

    def key_press_callback(self, event):
        if event.key == 'n':
            self.display_next_page()
        if event.key == 'p':
            self.display_prev_page()
