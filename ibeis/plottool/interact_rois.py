"""

guiback.py  -- code/ibeis/ibeis/gui
./resetdbs.sh
./dev.py --cmd --gui --db testdb1 (--setdb) <-set default


TODO:
1. create a function call to create a rectangle at a given set of points
2.. Make 45 degree rotation - need help



3. Change bounding box and update continuously to the original image the new ROIs

2. Make new window and frames inside, double click to pull up normal window with editing
start with just taking in 6 images and ROIs

1. ROI ID number, then list of 4 touples





Interactive tool to draw mask on an image or image-like array.

Adapted from matplotlib/examples/event_handling/poly_editor.py
Jan 9 2014: taken from: https://gist.github.com/tonysyu/3090704
"""
from __future__ import absolute_import, division, print_function
import matplotlib
#matplotlib.use('Qt4Agg')
from matplotlib.patches import Polygon
from matplotlib.widgets import Button
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math as math
#from matplotlib import nxutils  # Depricated
#from matplotlib.mlab import dist_point_to_segment
# Scientific
import numpy as np

from plottool import draw_func2 as df2


def _nxutils_points_inside_poly(points, verts):
    """ nxutils is depricated """
    path = matplotlib.path.Path(verts)
    return path.contains_points(points)


def verts_to_mask(shape, verts):
    print(verts)
    h, w = shape[0:2]
    y, x = np.mgrid[:h, :w]
    points = np.transpose((x.ravel(), y.ravel()))
    #mask = nxutils.points_inside_poly(points, verts)
    mask = _nxutils_points_inside_poly(points, verts)
    return mask.reshape(h, w)


def vertices_under_cursor(event):
    """Create 5 ind on one point"""
    x1 = event.xdata
    y1 = event.ydata
    return ((x1, y1), (x1, y1), (x1, y1,), (x1, y1))


def apply_mask(img, mask):
    masked_img = img.copy()
    masked_img[~mask] = np.uint8(np.clip(masked_img[~mask] - 100., 0, 255))
    return masked_img


def bbox_to_verts(bbox):
    (x, y, w, h) = bbox
    verts = np.array([(x + 0, y + h),
                      (x + 0, y + 0),
                      (x + w, y + 0),
                      (x + w, y + h),
                      (x + 0, y + h)], dtype=np.float32)
    return verts


def bbox_to_mask(shape, bbox):
    verts = bbox_to_verts(bbox)
    mask = verts_to_mask(shape, verts)
    return mask


class ROIInteraction(object):
    """
    An interactive polygon editor.

    Parameters
    ----------
    verts_list : list of lists of (float, float)
        List of (x, y) coordinates used as vertices of the polygon.
    max_ds : float
        Max pixel distance to count as a vertex hit.

    Key-bindings
    ------------
    't' : toggle vertex markers on and off.  When vertex markers are on,
          you can move them, delete them
    'd' : delete the vertex under point
    'i' : insert a vertex at point.  You must be within max_ds of the
          line connecting two existing vertices
    """
    def __init__(self,
                 img,
                 verts_list=None,
                 bbox_list=None,
                 max_ds=10,
                 line_width=4,
                 line_color=(1, 1, 1),
                 face_color=(0, 0, 0),
                 fnum=None,
                 do_mask=False):

        if fnum is None:
            fnum = df2.next_fnum()

        self.img = img
        self.do_mask = do_mask
        plt.figure(fnum)
        ax = plt.subplot(111)
        self.ax = ax

        ax.imshow(img)

        ax.set_clip_on(False)
        ax.set_title(('Click and drag a point to move it; or click once, then '
                      'click again.\nClick an ROI and press \"r\" to remove it\nPress \"t\" to add an ROI'))

        self.showverts = True
        self.max_ds = max_ds
        self.fc_default = face_color
        #mouse coordinates
        self.mouseX = None
        self.mouseY = None
        #if a polygon is currently active
        self._polyHeld = False
        #the polygon that is currently active
        self._thisPoly = None
        #used in small case to determine if polygon should be highlighted or not
        self.press1 = False
        #boolean to tell if the polygon SHOULD be active
        self.canUncolor = False
        #number of polygons in the image
        self._autoinc_polynum = 0

        #Something Jon added
        self.background = None
        #the method to call to update ROIs in the other program
        self.callback = None

        def new_polygon(verts):
            """ verts - list of (x, y) tuples """
            # create new polygon from verts
            poly = Polygon(verts, animated=True, fc=face_color, ec='none', alpha=0, picker=True)
            # register this polygon
            poly.num = self.next_polynum()
            return poly

        def new_line(poly):
            """ verts - list of (x, y) tuples """
            _xs, _ys = zip(*poly.xy)
            color = np.array(line_color)
            marker_face_color = line_color
            line_kwargs = {'lw': line_width, 'color': color, 'mfc': marker_face_color}
            line = plt.Line2D(_xs, _ys, marker='o', alpha=1, animated=True, **line_kwargs)
            return line

        # Ensure that our input is in verts_list format
        assert verts_list is None or bbox_list is None, 'only one can be specified'
        if bbox_list is not None:
            verts_list = [bbox_to_verts(bbox) for bbox in bbox_list]

        # Create the list of polygons
        self.polyList = [new_polygon(verts) for verts in verts_list]
        # Create the list of lines
        self.line = [new_line(poly) for poly in self.polyList]
        self._update_line()

        # Add polygons and lines to the axis
        for poly in self.polyList:
            ax.add_patch(poly)
        for line in self.line:
            self.ax.add_line(line)

        # Connect callbacks
        for poly in self.polyList:
            poly.add_callback(self.poly_changed)
        self._ind = None  # the active vert

        canvas = ax.figure.canvas
        canvas.mpl_connect('draw_event', self.draw_callback)
        canvas.mpl_connect('button_press_event', self.button_press_callback)
        canvas.mpl_connect('button_release_event', self.button_release_callback)
        canvas.mpl_connect('key_press_event', self.key_press_callback)
        canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)
        canvas.mpl_connect('pick_event', self.onpick)
        # canvas.mpl_connect('figure_enter_event', self.mouse_enter)
        # canvas.mpl_connect('figure_leave_event', self.mouse_leave)
        self.canvas = canvas

        # Define buttons
        self.accept_ax  = plt.axes([0.7, 0.05, 0.15, 0.075])
        self.accept_but = Button(self.accept_ax, 'Accept New ROIs')
        self.accept_but.on_clicked(self.accept_new_rois)

    def next_polynum(self):
        num = self._autoinc_polynum
        self._autoinc_polynum += 1
        return num

    def rotate45(self, poly):
        """
        starting to figure out rotation
        called when a certain button is clicked (currently when a key is clicked)
        move the points clockwise
        """
        # How I might do something like this:
        sin, cos, array = np.sin, np.cos, np.array

        # get a vector of the points I want to rotate around (0, 0):
        # e.g. pts = array([(1, 1), (2, 1), (1, 0), (0, 0)])
        pts = poly.xy

        # Because tau is twice as good as pi
        tau = 2 * np.pi  # tauday.com

        # Convert degrees to radians
        theta = 45 * tau / 360.0

        # Define rotation matrix (relative to the origin (0, 0))
        rot_mat = array(
            [(cos(theta), -sin(theta)),
             (sin(theta),  cos(theta))]
        )

        # LINEAR ALGEBRA TIME!
        # Given:
        #   rot_mat = A (2 x 2) rotation matrix
        #   pts = a (2 x 1) vector representing a point
        # rot_mat dotted with pts results in that point rotated theta radians around the origin
        #
        # FURTHERMORE:
        # Let pts = a (2 x N) matrix representing a list of N points
        # rot_mat dotted with this matrix results in all a a new (2 x N) matrix
        # representing all of the rotated points.
        #
        # Notes the .T is a numpy commmand for transpose
        # ie: change a (M x N) to an (N x M)
        # we do this because poly.xy is (N x 2), and we want a (2 x N)
        # then we do a second transpose to get us back to the original format
        new_pts = rot_mat.dot(pts.T).T

        poly.xy = new_pts
        pass

    def update_UI(self):
        self._update_line()
        self.canvas.restore_region(self.background)
        for n, poly in enumerate(self.polyList):
            self.ax.draw_artist(poly)
            self.ax.draw_artist(self.line[n])
        self.canvas.blit(self.ax.bbox)

    def poly_changed(self, poly):
        """ this method is called whenever the polygon object is called """
        # only copy the artist props to the line (except visibility)
        num = self.polyList.index(poly)
        vis = self.line[num].get_visible()
        #Artist.update_from(self.line, poly)
        self.line[num].set_visible(vis)
        #self.line[poly.num].set_visible(vis)  # don't use the poly visibility state

    def draw_callback(self, event):
        #print('[mask] draw_callback(event=%r)' % event)
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        for n, poly in enumerate(self.polyList):
            self.ax.draw_artist(poly)
            self.ax.draw_artist(self.line[n])
        self.canvas.blit(self.ax.bbox)

    def button_press_callback(self, event):
        """ whenever a mouse button is pressed """
        if self._ind is not None:
            self._ind = None
            return
        ignore = not self.showverts or event.inaxes is None or event.button != 1
        if ignore:
            return
        if self._thisPoly is None or self.line is None:
            print('WARNING: Polygon unknown. Using default.')
            if len(self.polyList) == 0:
                print('No polygons on screen')
                return
            else:
                self._thisPoly = self.polyList[0]

        polyind, self._ind = self.get_ind_under_cursor(event)

        if self._ind is not None and polyind is not None:
            self._thisPoly = self.polyList[polyind]
            self.indX, self.indY = self._thisPoly.xy[self._ind]
            self._polyHeld = True

        self.mouseX, self.mouseY = event.xdata, event.ydata

        if self._polyHeld is True or self._ind is not None:
            self._thisPoly.set_alpha(.2)
            self._thisPoly.set_facecolor('white')

        self.press1 = True
        self.canUncolor = False
        self._update_line()
        if self.background is not None:
            self.canvas.restore_region(self.background)
        else:
            print("error: self.background is none. Trying refresh.")
            self.canvas.restore_region(self.background)
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        for n, poly in enumerate(self.polyList):
            self.ax.draw_artist(poly)
            self.ax.draw_artist(self.line[n])
        self.canvas.blit(self.ax.bbox)

    def button_release_callback(self, event):
        """ whenever a mouse button is released """
        if self._polyHeld is True and (self._ind is None or self.press1 is False):
            self._polyHeld = False

        ignore = not self.showverts or event.button != 1 or self._thisPoly is None
        if ignore:
            return
        if (self._ind is None) or self._polyHeld is False or (self._ind is not None and self.press1 is True) and self._thisPoly is not None and self.canUncolor is True:
            self._thisPoly.set_alpha(0)

        self.update_UI()
        self.press1 = False

        if self._ind is None:
            return
        if self._thisPoly is None:
            print('WARNING: Polygon unknown. Using default. (2)')
            self._thisPoly = self.polyList[0]
        currX, currY = self._thisPoly.xy[self._ind]

        if math.fabs(self.indX - currX) < 3 and math.fabs(self.indY - currY) < 3:
            return

        if (self._ind is None) or self._polyHeld is False or (self._ind is not None and self.press1 is True) and self._thisPoly is not None:
            self._thisPoly = None
        self._ind = None
        self._polyHeld = False

    def draw_new_poly(self):
        coords = default_vertices(self.img)
        Poly = Polygon(coords, animated=True,
                            fc='white', ec='none', alpha=0.2, picker=True)
        self.polyList.append(Poly)
        self.ax.add_patch(Poly)
        x, y = zip(*Poly.xy)
        color = np.array((1, 1, 1))
        marker_face_color = (1, 1, 1)
        line_width = 4

        line_kwargs = {'lw': line_width, 'color': color, 'mfc': marker_face_color}
        self.line.append(plt.Line2D(x, y, marker='o', alpha=1, animated=True, **line_kwargs))
        self._update_line()
        self.ax.add_line(self.line[-1])

        Poly.add_callback(self.poly_changed)
        Poly.num = self.next_polynum()
        self._ind = None  # the active vert

    def delete_current_poly(self):
        if self._thisPoly is None:
            print('No Poly Selected to delete')
            return
        Poly = self._thisPoly
        lineNumber = self.polyList.index(Poly)
        ###print('poly list: ', len(self.polyList), 'list size ', len(self.line), 'index of poly ', lineNumber)

        #line deletion
        del self.line[lineNumber]
        #poly deletion
        self.polyList.remove(Poly)
        #remove the poly from the figure itself
        Poly.remove()
        #reset anything that has to do with current poly
        self._thisPoly = None;
        self._polyHeld = False;

    def load_points(self):
        new_verts_list = []
        for poly in self.polyList:
            new_verts_list.append(poly.xy)
        print(new_verts_list)
        return new_verts_list

    def key_press_callback(self, event):
        """ whenever a key is pressed """
        print('key_press_callback')
        if not event.inaxes:
            return
        if event.key == 't':
            self.draw_new_poly()
        # old code for adding and deleting Polygon vertices (would need to
        # rewrite for multiply polygons

        """code for deleting a polygon"""
        if event.key == 'r':
            self.delete_current_poly()

        if event.key == 'u':
            self.load_points()

        # elif event.key == 'd':
        #     ind = self.get_ind_under_cursor(event)
        #     if ind is None:
        #         return
        #     if ind == 0 or ind == self.last_vert_ind:
        #         print('[mask] Cannot delete root node')
        #         return
        #     self.poly.xy = [tup for i, tup in enumerate(self.poly.xy) if i != ind]
        #     self._update_line()
        # elif event.key == 'i':
        #     xys = self.poly.get_transform().transform(self.poly.xy)
        #     p = event.x, event.y  # cursor coords
        #     for i in range(len(xys) - 1):
        #         s0 = xys[i]
        #         s1 = xys[i + 1]
        #         d = dist_point_to_segment(p, s0, s1)
        #         if d <= self.max_ds:
        #             self.poly.xy = np.array(
        #                 list(self.poly.xy[:i + 1]) +
        #                 [(event.xdata, event.ydata)] +
        #                 list(self.poly.xy[i + 1:]))
        #             self._update_line()
        #             break
        self.canvas.draw()

    def motion_notify_callback(self, event):
        """ on mouse movement """
        #print('motion_notify_callback')
        ignore = (not self.showverts or event.inaxes is None)
        if ignore:
            return
        if self.press1 is True:
            self.canUncolor = True
        if self._ind is None and event.button == 1:
            # move all vertices
            if self._polyHeld is True:
                self.move_rectangle(event, self._thisPoly, event.xdata, event.ydata)
            self.update_UI()
            self._ind = None
            # set new mouse loc
            self.mouseX, self.mouseY = event.xdata, event.ydata

        if self._ind is None:
            return
        if self._polyHeld is True:
            self.calculate_move(event, self._thisPoly)
        else:
            print('error no poly known')
        self.update_UI()

    def onpick(self, event):
        """ Makes selected polygon translucent """
        print('onpick')
        self._thisPoly = event.artist
        #x, y = event.mouseevent.xdata, event.mouseevent.xdata
        self._polyHeld = True

    def mouse_enter(self, event):
        print('mouse_enter')
        self._thisPoly = event.artist
        self._thisPoly.set_alpha(.2)

    def mouse_leave(self, event):
        print('mouse_leave')
        self._thisPoly.set_alpha(0)
        self._thisPoly = None

    def check_dims(self, coords):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        if coords[0] >= xlim[0] and coords[0] <= xlim[1] and coords[1] >= ylim[1] and coords[1] <= ylim[0]:
            return True
        return False

    def move_rectangle(self, event, polygon, x, y):
        print('move_rectangle')
        selectedX, selectedY = (polygon.xy[1])
        beforeX, beforeY = (polygon.xy[0])
        afterX, afterY = (polygon.xy[2])
        acrossX, acrossY = (polygon.xy[3])
        # if we are not holding a rectangle, return
        if self._polyHeld is not True:
            return
        # Change selected
        new1 = (selectedX + (x - self.mouseX), selectedY + (y - self.mouseY))
        new0 = (beforeX   + (x - self.mouseX), beforeY   + (y - self.mouseY))
        new2 = (afterX    + (x - self.mouseX), afterY    + (y - self.mouseY))
        new3 = (acrossX   + (x - self.mouseX), acrossY   + (y - self.mouseY))
        if(self.check_dims(new1)) is True and self.check_dims(new0) is True and self.check_dims(new2) is True and self.check_dims(new3) is True:
            polygon.xy[1] = new1
            # Change before vert
            polygon.xy[0] = new0
            polygon.xy[self.last_vert_ind] = new0
            # Change after vert
            polygon.xy[2] = new2
            # Change across vert
            polygon.xy[3] = new3

    def calculate_move(self, event, poly):
        print('calculate_move')
        indBefore = self._ind - 1
        if(indBefore < 0):
            indBefore = len(poly.xy) - 2
        indAfter = (self._ind + 1) % 4
        selectedX, selectedY = (poly.xy[self._ind])
        beforeX, beforeY = (poly.xy[indBefore])
        afterX, afterY = (poly.xy[indAfter])

        changeBefore = -1
        keepX, changeY = -1, -1
        changeAfter = -1
        changeX, keepY = -1, -1

        if beforeX != selectedX:
            changeBefore = indBefore
            keepX, changeY = poly.xy[indBefore]
            changeAfter = indAfter
            changeX, keepY = poly.xy[indAfter]
        else:
            changeBefore = indAfter
            keepX, changeY = poly.xy[indAfter]
            changeAfter = indBefore
            changeX, keepY = poly.xy[indBefore]

        x, y = event.xdata, event.ydata

        # Change selected
        if self._ind == 0 or self._ind == self.last_vert_ind:
            poly.xy[0] = x, y
            poly.xy[self.last_vert_ind] = x, y
        else:
            poly.xy[self._ind] = x, y

        # Change vert
        if changeBefore == 0 or changeBefore == self.last_vert_ind:
            poly.xy[0] = keepX, y
            poly.xy[self.last_vert_ind] = keepX, y
        else:
            poly.xy[changeBefore] = keepX, y

        # Change horiz
        if changeAfter == 0 or changeAfter == self.last_vert_ind:
            poly.xy[0] = x, keepY
            poly.xy[self.last_vert_ind] = x, keepY
        else:
            poly.xy[changeAfter] = x, keepY

    def _update_line(self):
        # save verts because polygon gets deleted when figure is closed
        for n, poly in enumerate(self.polyList):
            #self.verts = poly.xy
            self.last_vert_ind = len(poly.xy) - 1
            self.line[n].set_data(zip(*poly.xy))

    def get_ind_under_cursor(self, event):
        'get the index of the vertex under cursor if within max_ds tolerance'

        def get_ind_and_dist(poly):
            xy = np.asarray(poly.xy)
            xyt = poly.get_transform().transform(xy)
            xt, yt = xyt[:, 0], xyt[:, 1]
            d = np.sqrt((xt - event.x) ** 2 + (yt - event.y) ** 2)
            indseq = np.nonzero(np.equal(d, np.amin(d)))[0]
            ind = indseq[0]
            mindist = d[ind]
            if mindist >= self.max_ds:
                ind = None
                mindist = None
            return (ind, mindist)
        ind_dist_list = [get_ind_and_dist(poly) for poly in self.polyList]
        min_dist = None
        min_ind  = None
        sel_polyind = None
        for polyind, (ind, dist) in enumerate(ind_dist_list):
            if ind is None:
                continue
            if min_dist is None:
                min_dist = dist
                min_ind = ind
                sel_polyind = polyind
            elif dist < min_dist:
                min_dist = dist
                min_ind = ind
                sel_polyind = polyind
        return (sel_polyind, min_ind)

    def accept_new_rois(self, event):
        print('Pressed Accept Button')


        def send_back_rois(self):
            point_list = self.load_points()
            self.callback(point_list)

        print(self.callback)
        if self.callback is not None:
            send_back_rois()
        else:
            #just print the updated points
            self.load_points()
        # Make mask from selection
        if self.do_mask is True:
            plt.clf()
            ax = plt.subplot(111)
            img = self.img
            mask = self.get_mask(img.shape)
            # User must close previous figure
            # Modify the image with the mask
            masked_img = apply_mask(img, mask)
            # show the modified image
            ax.imshow(masked_img)
            plt.title('Region outside of mask is darkened')
        
            ax.figure.show()

        print('show2')


    def get_mask(self, shape):
        """Return image mask given by mask creator"""
        mask_list = [verts_to_mask(shape, poly.xy) for poly in self.polyList]
        if len(mask_list) == 0:
            print('No polygons to make mask out of')
            return 0
        mask = mask_list[0]
        for mask_ in mask_list:
            mask = np.maximum(mask, mask_)
        return mask


def default_vertices(img):
    """Default to rectangle that has a quarter-width/height border."""
    (h, w) = img.shape[0:2]
    x1, x2 = np.array([0, w]) + (w // 4 * np.array([1, -1]))
    y1, y2 = np.array([0, h]) + (h // 4 * np.array([1, -1]))
    return ((x1, y1), (x1, y2), (x2, y2), (x2, y1))


def ROI_creator(img, verts_list):#add callback as variable
    print('*** START DEMO ***')
    

    if verts_list is None:
        verts_list = [default_vertices(img)]
    else:
        for verts in verts_list:
            if (len(verts) is not 5):
                print("verts list is not of correct length. ", len(verts))
                return


    if img is None:
        try:
            import utool
            img_url = 'http://i.imgur.com/Vq9CLok.jpg'
            img_fpath = utool.grab_file_url(img_url)
            img = mpimg.imread(img_fpath)
        except Exception as ex:
            print('cant read zebra: %r' % ex)
            img = np.random.uniform(0, 255, size=(100, 100))
    # if callback is not None:
    #     self.callback = callback
    mc = ROIInteraction(img, verts_list=verts_list, fnum=0)  # NOQA
    # Do interaction
    plt.show()
    # Make mask from selection
    #mask = mc.get_mask(img.shape)
    # User must close previous figure
    # Modify the image with the mask
    #masked_img = apply_mask(img, mask)
    # show the modified image
    #plt.imshow(masked_img)
    #plt.title('Region outside of mask is darkened')
    #print('show2')
    #plt.show()


if __name__ == '__main__':
    verts = [((0,400),(400,400),(400,0),(0,0),(0,400)), ((400,700),(700,700),(700,400),(400,400), (400,700))]
    ROI_creator(None, verts)
