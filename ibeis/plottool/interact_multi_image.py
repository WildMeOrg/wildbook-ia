from __future__ import absolute_import, division, print_function
from matplotlib.widgets import Button
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from plottool import viz_image2
from plottool import interact_rois
from plottool import draw_func2 as df2
from plottool import plot_helpers as ph
from plottool import interact_helpers as ih
import cv2
#import utool


class MultiImageInteraction(object):
    def __init__(self, gpath_list, max_per_page=10, bboxes_list=None, thetas_list=None, verts_list=None, gid_list=None,
                 nImgs=None, fnum=None):
        print('Creating multi-image interaction')

    #def __init__(self, img_list, nImgs=None, gid_list=None, rids_list=None, bboxes_list=None, max_per_page=10,fnum=None):
        print("maX ",max_per_page)
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
        self.display_next_page()

    def display_next_page(self, event=None):
        self.page_number = self.page_number + 1
        nLeft = self.nImgs - self.current_index
        if nLeft == 0:
            fig = df2.figure(fnum=self.fnum, pnum=(1, 1, 1))
            fig.clf()
            return False
            raise AssertionError('no more images to display')
        nDisplay = min(nLeft, self.max_per_page)
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
        for px, index in enumerate(xrange(start_index, end_index)):
            gpath      = self.gpath_list[index]
            bbox_list  = self.bboxes_list[index]
            theta_list = self.thetas_list[index]
            img = cv2.imread(gpath)
            _vizkw = {
                'fnum': self.fnum,
                'pnum': pnum_(px),
                'title': str(index),
                'bbox_list'  : bbox_list,
                'theta_list' : theta_list,
                'sel_list'   : [],
                'label_list' : [],
            }
            #print(utool.dict_str(_vizkw))
            _, ax = viz_image2.show_image(img, **_vizkw)
            ph.set_plotdat(ax, 'px', str(px))
            ph.set_plotdat(ax, 'bbox_list', bbox_list)
            ph.set_plotdat(ax, 'gpath', gpath)
            #print('components: ', fig.get_children())
            if px + 1 >= nDisplay:
                break
        self.current_index = end_index
        # Set the figure title
        df2.set_figtitle('Displaying (%d - %d) / %d' % (start_index + 1, end_index, self.nImgs))

        # Create the button for scrolling forwards
        self.next_ax = plt.axes([0.7, 0.05, 0.15, 0.075])
        self.next_but = Button(self.next_ax, 'next')
        self.next_but.on_clicked(self.display_next_page)

        # Connect the callback whenever the figure is clicked
        ih.connect_callback(fig, 'button_press_event', self.on_figure_clicked)

        # Show the changes
        fig.show()
        print('next')

    def update_images(self, img_ind, updated_bbox_list):
        print("update called")
        index = int (img_ind)
        print(self.bboxes_list[index])
        self.bboxes_list[index] = updated_bbox_list;
        print(self.bboxes_list[index])
        print("bbox update done")
        """Insert code for viz_image2 redrawing here"""
        gpath = self.gpath_list[index]
        bbox_list  = self.bboxes_list[index]
        theta_list = self.thetas_list[index]
        img = cv2.imread(gpath)
        _vizkw = {
            'fnum': self.fnum,
            #'pnum': pnum_(px),
            'title': str(index),
            'bbox_list'  : bbox_list,
            'theta_list' : theta_list,
            'sel_list'   : [],
            'label_list' : [],
        }
            #print(utool.dict_str(_vizkw))
        _, ax = viz_image2.show_image(img, **_vizkw)
    def on_figure_clicked(self, event):



        if ih.clicked_inside_axis(event):
            ax = event.inaxes
            px = ph.get_plotdat(ax, 'px')
            bbox_list  = ph.get_plotdat(ax, 'bbox_list')
            bbox_list = self.bboxes_list[int(px)]
            print("Bbox of figure: ",bbox_list)
            theta_list = ph.get_plotdat(ax, 'theta_list')
            gpath      = ph.get_plotdat(ax, 'gpath')
            img = mpimg.imread(gpath)
            fnum = df2.next_fnum()
            mc = interact_rois.ROIInteraction(img, px, self.update_images, bbox_list=bbox_list, fnum=fnum)
            # """wait for accept
            # have a flag to tell if a bbox has been changed, on the bbox list that is brought it"
            # on accept:
            # viz_image2.show_image callback
            # """
            plt.show()
            print('Clicked: ax: px=%r' % px)

        #img_ind = (self.figlist.index(event.artist) - 1) + (self.max_per_page * self.page_number) #print(imgs[0].make_image())
        #print(self.img_list[3])
        #"""Need to add ROI code"""
        #verts_of_image_selected = None
        #"""Need to figure out how to get the img from the code above"""
        #img = self.img_list[img_ind]
        #irs.ROI_creator(img, verts_of_image_selected)
#     def onpick(self, event):
#         img_ind = (self.figlist.index(event.artist) - 1) + (self.max_per_page * self.page_number)
        
#         """Need to add ROI code"""
#         if self.rids_list is not None:
#             verts_of_image_selected = self.rids_list[img_ind]
#         else:
#             verts_of_image_selected = None
#         """Need to figure out how to get the img from the code above"""
#         img = self.img_list[img_ind]
#         irs.ROI_creator(img, img_ind, verts_of_image_selected, self.update_lists)
# >>>>>>> Stashed changes
