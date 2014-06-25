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
from matplotlib.patches import Polygon
from matplotlib.widgets import Button
#import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math as math
#from matplotlib import nxutils  # Depricated
#from matplotlib.mlab import dist_point_to_segment
# Scientific
import numpy as np
import utool

from plottool import draw_func2 as df2
from itertools import izip

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


def verts_to_bbox(verts):
    print()
    new_bbox_list = []
    print(verts)
    for i in range(0, len(verts)):
        print('verts[i][0][0]: ', verts[i][0][0], " verts[i][0][1]: ", verts[i][0][1])
        x = verts[i][0][0]
        y = verts[i][0][1]
        w = verts[i][2][0] - x
        h = verts[i][2][1] - y
        bbox = (x, y, w, h)
        new_bbox_list.append(bbox)
    return new_bbox_list


def bbox_to_mask(shape, bbox):
    verts = bbox_to_verts(bbox)
    mask = verts_to_mask(shape, verts)
    return mask


def points_center(pts):
    # the polygons have the first point listed twice in order for them to be drawn as closed, but that point shouldn't be counted twice for computing the center (hence the [:-1] slice)
    return np.array(pts[:-1]).mean(axis=0)


def polygon_center(poly):
    return points_center(poly.xy)


def polygon_dims(poly):
    xs = [x for (x, y) in poly.basecoords]
    ys = [y for (x, y) in poly.basecoords]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    return (w, h)

def rotate_points_around(points, theta, ax, ay):
    sin, cos, array = np.sin, np.cos, np.array
    augpts = array([array((x, y, 1)) for (x, y) in points])
    ct = cos(theta)
    st = sin(theta)
    # correct matrix obtained from http://www.euclideanspace.com/maths/geometry/affine/aroundPoint/matrix2d/
    rot_mat = array(
        [(ct, -st, ax - ct * ax + st * ay),
         (st,  ct, ay - st * ax - ct * ay),
         ( 0,   0,                                 1)]
    )
    return [(x, y) for (x, y, z) in rot_mat.dot(augpts.T).T]


def calc_display_coords(oldcoords, theta):
    return rotate_points_around(oldcoords, theta, *points_center(oldcoords))


def set_display_coords(poly):
    poly.xy = calc_display_coords(poly.basecoords, poly.theta)


def is_within_distance(dist, p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dx2 = dx * dx
    dy2 = dy * dy
    d2 = dist * dist
    rv = d2 > (dx2 + dy2)
    #print('is_within_distance(%r, (%r, %r), (%r, %r)) = %r' % (dist, p1[0], p1[1], p2[0], p2[1], rv))
    return rv

def calc_handle_coords(poly):
    cx, cy = polygon_center(poly)
    w, h = polygon_dims(poly)
    x0, y0 = cx, (cy - (h / 2)) # start at top edge
    x1, y1 = (x0, y0-(h/4))
    pts = [(x0, y0), (x1, y1)]
    pts = rotate_points_around(pts, poly.theta, cx, cy)
    return pts


def make_handle_line(poly):
    _xs, _ys = zip(*calc_handle_coords(poly))
    line_width = 4
    line_color = (0, 1, 0)
    color = np.array(line_color)
    marker_face_color = line_color
    line_kwargs = {'lw': line_width, 'color': color, 'mfc': marker_face_color}
    lines = plt.Line2D(_xs, _ys, marker='o', alpha=1, animated=True, **line_kwargs)
    return lines


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
                 img_ind=None,
                 callback=None,
                 verts_list=None,
                 bbox_list=None,  # will get converted to verts_list
                 theta_list=None,
                 max_ds=10,
                 line_width=4,
                 line_color=(1, 1, 1),
                 face_color=(0, 0, 0),
                 fnum=None,

                 do_mask=False):
        if fnum is None:
            fnum = df2.next_fnum()
        if callback is not None:
            self.callback = callback
        else:
            self.callback = None
        if bbox_list is not None:
            self.original_bbox_list = bbox_list
        else:
            self.original_bbox_list = []
        if theta_list is not None:
            self.original_theta_list = theta_list
        else:
            self.original_theta_list = []
        self.img = img
        self.do_mask = do_mask
        self.fig = df2.figure(fnum=fnum, doclf=True, docla=True)
        self.fig.clear()
        self.fig.clf()
        #self.fig.cla()
        #utool.qflag()
        self.fnum = fnum
        print(self.fnum)
        #ax = plt.subplot(111)
        ax = df2.gca()
        self.ax = ax
        self.img_ind = img_ind

        df2.imshow(img, fnum=fnum)

        ax.set_clip_on(False)
        ax.set_title(('\n'.join([
            'Click and drag to select/move/resize an ROI',
            'Press \"r\" to remove selected ROI',
            'Press \"t\" to add an ROI.',
            'Press \"a\" to Accept new ROIs'])))

        self.showverts = True
        self.max_ds = max_ds
        self.fc_default = face_color
        #mouse coordinates
        self.mouseX = None
        self.mouseY = None
        self.indX = None
        self.indY = None
        #if a polygon is currently active
        self._polyHeld = False
        #the polygon that is currently active
        self._currently_selected_poly = None
        #used in small case to determine if polygon should be highlighted or not
        self.press1 = False
        #boolean to tell if the polygon SHOULD be active
        self.canUncolor = False
        #number of polygons in the image
        self._autoinc_polynum = 0

        #Something Jon added
        self.background = None

        def make_lines(poly):
            """ verts - list of (x, y) tuples """
            _xs, _ys = zip(*poly.xy)
            color = np.array(line_color)
            marker_face_color = line_color
            line_kwargs = {'lw': line_width, 'color': color, 'mfc': marker_face_color}
            lines = plt.Line2D(_xs, _ys, marker='o', alpha=1, animated=True, **line_kwargs)
            print('make_lines: linetype = %r' % type(lines))
            return lines

        def new_polygon(verts, theta):
            """ verts - list of (x, y) tuples """
            # create new polygon from verts
            poly = Polygon(verts, animated=True, fc=face_color, ec='none', alpha=0, picker=True)
            # register this polygon
            poly.num = self.next_polynum()
            poly.theta = theta
            poly.basecoords = poly.xy
            poly.lines = make_lines(poly)
            poly.handle = make_handle_line(poly)
            return poly

        # print(verts_list)
        # test_list = verts_to_bbox(verts_list)
        # print(test_list)
        # Ensure that our input is in verts_list format
        assert verts_list is None or bbox_list is None, 'only one can be specified'
        if bbox_list is not None:
            verts_list = [bbox_to_verts(bbox) for bbox in bbox_list]
        if theta_list is None:
            theta_list = [0 for verts in verts_list]
        # Create the list of polygons
        poly_list = [new_polygon(verts, theta) for (verts, theta) in izip(verts_list, theta_list)]
        assert len(theta_list) == len(poly_list), 'theta_list: %r, poly_list: %r' % (theta_list, poly_list)
        self.polys = dict({(poly.num, poly) for poly in poly_list})
        self._update_line()

        # Add polygons and lines to the axis
        for poly in self.polys.itervalues():
            ax.add_patch(poly)
            self.ax.add_line(poly.lines)
            self.ax.add_line(poly.handle)

        # Connect callbacks
        for poly in self.polys.itervalues():
            poly.add_callback(self.poly_changed)
        self._ind = None  # the active vert
        self.currently_rotating_poly = None

        canvas = ax.figure.canvas
        #http://matplotlib.org/1.3.1/api/backend_bases_api.html
        canvas.mpl_connect('draw_event', self.draw_callback)
        canvas.mpl_connect('button_press_event', self.button_press_callback)
        canvas.mpl_connect('button_release_event', self.button_release_callback)
        canvas.mpl_connect('key_press_event', self.key_press_callback)
        canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)
        canvas.mpl_connect('pick_event', self.onpick)
        canvas.mpl_connect('resize_event', self.on_resize)
        # canvas.mpl_connect('figure_enter_event', self.mouse_enter)
        # canvas.mpl_connect('figure_leave_event', self.mouse_leave)
        self.canvas = canvas

        # Define buttons
        self.accept_ax  = plt.axes([0.63, 0.01, 0.2, 0.06])
        self.accept_but = Button(self.accept_ax, 'Accept New ROIs')
        self.accept_but.on_clicked(self.accept_new_rois)

        self.add_ax  = plt.axes([0.2, .01, 0.16, 0.06])
        self.add_but = Button(self.add_ax, 'Add Rectangle')
        self.add_but.on_clicked(self.draw_new_poly)

        self.del_ax  = plt.axes([0.4, 0.01, 0.19, 0.06])
        self.del_but = Button(self.del_ax, 'Delete Rectangle')
        self.del_but.on_clicked(self.delete_current_poly)

    def on_resize(self, event):
        #print(utool.dict_str(event.__dict__))
        #self.canvas.draw()
        #self.fig.canvas.draw()
        #self.canvas.update()
        #self.fig.canvas.update()
        plt.draw()
        pass

    def show(self):
        self.fig.canvas.show()

    def next_polynum(self):
        num = self._autoinc_polynum
        self._autoinc_polynum += 1
        return num

    def update_colors(self, poly_ind):
        if poly_ind is None:
            print("WARNING: poly_ind is None in update_colors")
            return
        line = self.polys[poly_ind].lines
        if line.get_color() != 'white':
            line.set_color('white')
        if(poly_ind is not None and poly_ind >= 0):
            self.polys[poly_ind].lines.set_color(df2.ORANGE)
        plt.draw()

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
        for poly in self.polys.itervalues():
            self.ax.draw_artist(poly)
            self.ax.draw_artist(poly.lines)
            self.ax.draw_artist(poly.handle)
        self.canvas.blit(self.ax.bbox)

    def poly_changed(self, poly):
        """ this method is called whenever the polygon object is called """
        # only copy the artist props to the line (except visibility)
        num = poly.num
        vis = poly.lines.get_visible()
        vis = poly.handle.get_visible()
        #Artist.update_from(poly.lines, poly)
        poly.lines.set_visible(vis)
        poly.handle.set_visible(vis)
        #poly.lines.set_visible(vis)  # don't use the poly visibility state

    def draw_callback(self, event):
        #print('[mask] draw_callback(event=%r)' % event)
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        for poly in self.polys.itervalues():
            self.ax.draw_artist(poly)
            self.ax.draw_artist(poly.lines)
            self.ax.draw_artist(poly.handle)
        self.canvas.blit(self.ax.bbox)

    def get_most_recently_added_poly(self):
        if len(self.polys) != 0:
            poly_ind = max(self.polys.iterkeys())  # most recently added polygon has the highest index
            return poly_ind, self.polys[poly_ind]
        else:
            return (None, None)

    def button_press_callback(self, event):
        """ whenever a mouse button is pressed """
        if self._ind is not None:
            self._ind = None
            return
        ignore = not self.showverts or event.inaxes is None or event.button != 1
        if ignore:
            return

        if event.button == 1: # leftclick
            for poly in self.polys.itervalues():
                if is_within_distance(10, (event.xdata, event.ydata), calc_handle_coords(poly)[1]):
                    self.currently_rotating_poly = poly
                    break

        if self._currently_selected_poly is None:
            print('WARNING: Polygon unknown. Using last placed poly.')
            if len(self.polys) == 0:
                print('No polygons on screen')
                return
            else:
                poly_ind, self._currently_selected_poly = self.get_most_recently_added_poly()
                self.update_colors(poly_ind)
        polyind, self._ind = self.get_ind_under_cursor(event)

        if self._ind is not None and polyind is not None:
            self._currently_selected_poly = self.polys[polyind]
            if self._currently_selected_poly is None:
                return
            self.indX, self.indY = self._currently_selected_poly.xy[self._ind]
            self._polyHeld = True
            self.update_colors(polyind)

        self.mouseX, self.mouseY = event.xdata, event.ydata

        if self._polyHeld is True or self._ind is not None:
            self._currently_selected_poly.set_alpha(.2)
            self.update_colors(self._currently_selected_poly.num)
            #self._currently_selected_poly.set_facecolor('red')
            #self._currently_selected_poly.lines.set_color('red')
        self.press1 = True
        self.canUncolor = False
        self._update_line()
        if self.background is not None:
            self.canvas.restore_region(self.background)
        else:
            print("error: self.background is none. Trying refresh.")
            self.canvas.restore_region(self.background)
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        for poly in self.polys.itervalues():
            self.ax.draw_artist(poly)
            self.ax.draw_artist(poly.lines)
            self.ax.draw_artist(poly.handle)
        self.canvas.blit(self.ax.bbox)

    def button_release_callback(self, event):
        """ whenever a mouse button is released """
        if self._polyHeld is True and (self._ind is None or self.press1 is False):
            self._polyHeld = False

        self.currently_rotating_poly = None

        ignore = not self.showverts or event.button != 1 or self._currently_selected_poly is None
        if ignore:
            return
        if (self._ind is None) or self._polyHeld is False or \
           (self._ind is not None and self.press1 is True) and \
           self._currently_selected_poly is not None and self.canUncolor is True:
            self._currently_selected_poly.set_alpha(0)
            #self._currently_selected_poly.set_facecolor('white')

        self.update_UI()
        self.press1 = False

        if self._ind is None:
            return
        if self._currently_selected_poly is None:
            print('WARNING: Polygon unknown. Using default. (2)')
            if len(self.polys) == 0:
                print('No polygons on screen')
                return
            else:
                poly_ind, self._currently_selected_poly = self.get_most_recently_added_poly()
        currX, currY = self._currently_selected_poly.xy[self._ind]

        if self.indX and self.indY:
            if math.fabs(self.indX - currX) < 3 and math.fabs(self.indY - currY) < 3:
                return

        if (self._ind is None) or self._polyHeld is False or \
           (self._ind is not None and self.press1 is True) and \
           self._currently_selected_poly is not None:
            self._currently_selected_poly = None
            self.update_colors(None)
        self._ind = None
        self._polyHeld = False

    #might be a duplicate of a helper in __init__/part of __init__?
    def draw_new_poly(self, event=None):
        coords = default_vertices(self.img)
        poly = Polygon(coords, animated=True,
                            fc='white', ec='none', alpha=0.2, picker=True)
        poly.num = self.next_polynum()
        poly.theta = 0
        poly.basecoords = poly.xy
        #self.poly_list.append(poly)
        self.polys[poly.num] = poly
        #self.theta_list.append(0)
        self.ax.add_patch(poly)
        x, y = zip(*poly.xy)
        color = np.array((1, 1, 1))
        marker_face_color = (1, 1, 1)
        line_width = 4

        line_kwargs = {'lw': line_width, 'color': color, 'mfc': marker_face_color}
        poly.lines = plt.Line2D(x, y, marker='o', alpha=1, animated=True, **line_kwargs)
        poly.handle = make_handle_line(poly)
        self._update_line()

        self.ax.add_line(poly.lines)
        self.ax.add_line(poly.handle)

        poly.add_callback(self.poly_changed)
        self._ind = None  # the active vert
        poly_ind, self._currently_selected_poly = self.get_most_recently_added_poly()
        assert poly_ind == poly.num, 'ind %r, num %r' % (poly_ind, poly.num)
        self.update_colors(poly_ind)
        plt.draw()

    def delete_current_poly(self, event=None):
        if self._currently_selected_poly is None:
            print('No polygon selected to delete')
            return
        poly = self._currently_selected_poly
        lineNumber = poly.num
        ###print('poly list: ', len(self.poly_list), 'list size ', len(self.line), 'index of poly ', lineNumber)

        #line deletion
        print("length: ", len(self.polys), "number: ", lineNumber)
        #self.theta_list[lineNumber] = None
        #self.line[lineNumber] = None
        #poly deletion
        self.polys.pop(lineNumber)
        #self.poly_list.remove(poly)
        #remove the poly from the figure itself
        poly.remove()
        #reset anything that has to do with current poly
        poly_ind, self._currently_selected_poly = self.get_most_recently_added_poly()
        self._polyHeld = False
        if poly_ind is not None:
            self.update_colors(poly_ind)
        plt.draw()

    def load_points(self):
        return [poly.xy for poly in self.polys.itervalues()]

    def key_press_callback(self, event):
        """ whenever a key is pressed """
        print('key_press_callback')
        if not event.inaxes:
            return
        if event.key == 'a':
            self.accept_new_rois(event)

        if event.key == 't':
            self.draw_new_poly()
        # old code for adding and deleting Polygon vertices (would need to
        # rewrite for multiply polygons

        # code for deleting a polygon
        if event.key == 'r':
            self.delete_current_poly()

        if event.key == 'u':
            self.load_points()

        if event.key == 'p':
            print(plt.get_fignums())
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
        if not (event.xdata is None or event.ydata is None):
            # uses boolean punning for terseness
            lastX = self.mouseX or None
            lastY = self.mouseY or None
            self.mouseX, self.mouseY = event.xdata, event.ydata
            #print('mouse coords %r, %r; previous %r, %r' % (self.mouseX, self.mouseY, lastX, lastY))
            deltaX = lastX is not None and self.mouseX - lastX
            deltaY = lastY is not None and self.mouseY - lastY

        if ignore:
            return
        if self.press1 is True:
            self.canUncolor = True
        if self._ind is None and event.button == 1:
            # move all vertices
            if self._polyHeld is True and not (deltaX is None or deltaY is None):
                self.move_rectangle(self._currently_selected_poly, deltaX, deltaY)
            self.update_UI()
            self._ind = None

        if self.currently_rotating_poly:
                poly = self.currently_rotating_poly
                cx, cy = polygon_center(poly)
                theta = math.atan2(cy - self.mouseY, cx - self.mouseX) - np.tau/4
                dtheta = theta - poly.theta
                self.rotate_rectangle(poly, dtheta)
                self.update_UI()

        if self._ind is None:
            return
        if self._polyHeld is True:
            self.resize_rectangle(self._currently_selected_poly, self.mouseX, self.mouseY)
        else:
            print('error no poly known')
        self.update_UI()

    def onpick(self, event):
        """ Makes selected polygon translucent """
        print('onpick')
        self._currently_selected_poly = event.artist
        #x, y = event.mouseevent.xdata, event.mouseevent.xdata
        self._polyHeld = True

    def mouse_enter(self, event):
        print('mouse_enter')
        self._currently_selected_poly = event.artist
        self._currently_selected_poly.set_alpha(.2)

    def mouse_leave(self, event):
        print('mouse_leave')
        self._currently_selected_poly.set_alpha(0)
        self._currently_selected_poly = None

    def check_dims(self, coords):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        if coords[0] < xlim[0]:
            return False
            #coords[0] = xlim[0]
        if coords[0] > xlim[1]:
            return False
            #coords[0] = xlim[1]
        if coords[1] < ylim[1]:
            return False
            #coords[1] = ylim[1]
        if coords[1] > ylim[0]:
            return False
            #coords[1] = ylim[0]
        return True

    def clip_vert_to_bounds(self, coords):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        def clamp(lims, val):
            return max(lims[0], min(lims[1], val))
        return np.array((clamp(xlim, coords[0]), clamp(ylim, coords[1])))

    def check_valid_coords(self, coords_list):
        valid = True
        for coord in coords_list:
            if not self.check_dims(coord):
                valid = False
        return valid

    def rotate_rectangle(self, poly, dtheta):
        #print('rotate_rectangle')
        if self.check_valid_coords(calc_display_coords(poly.basecoords, poly.theta + dtheta)):
            poly.theta += dtheta
            set_display_coords(poly)

    def move_rectangle(self, poly, dx, dy):
        #print('move_rectangle')
        new_coords = [(x+dx, y+dy) for (x, y) in poly.basecoords]
        if self.check_valid_coords(calc_display_coords(new_coords, poly.theta)):
            poly.basecoords = new_coords
            set_display_coords(poly)

    def resize_rectangle(self, poly, x, y):
        #print('resize_rectangle')
        if poly is None:
            return
        indBefore = self._ind - 1
        if(indBefore < 0):
            indBefore = len(poly.xy) - 2
        indAfter = (self._ind + 1) % 4
        selectedX, selectedY = (poly.basecoords[self._ind])
        beforeX, beforeY = (poly.basecoords[indBefore])
        afterX, afterY = (poly.basecoords[indAfter])

        changeBefore = -1
        keepX, changeY = -1, -1
        changeAfter = -1
        changeX, keepY = -1, -1

        if beforeX != selectedX:
            changeBefore = indBefore
            keepX, changeY = poly.basecoords[indBefore]
            changeAfter = indAfter
            changeX, keepY = poly.basecoords[indAfter]
        else:
            changeBefore = indAfter
            keepX, changeY = poly.basecoords[indAfter]
            changeAfter = indBefore
            changeX, keepY = poly.basecoords[indBefore]

        # Change selected
        if self._ind == 0 or self._ind == self.last_vert_ind:
            poly.basecoords[0] = x, y
            poly.basecoords[self.last_vert_ind] = x, y
        else:
            poly.basecoords[self._ind] = x, y

        # Change vert
        if changeBefore == 0 or changeBefore == self.last_vert_ind:
            poly.basecoords[0] = keepX, y
            poly.basecoords[self.last_vert_ind] = keepX, y
        else:
            poly.basecoords[changeBefore] = keepX, y

        # Change horiz
        if changeAfter == 0 or changeAfter == self.last_vert_ind:
            poly.basecoords[0] = x, keepY
            poly.basecoords[self.last_vert_ind] = x, keepY
        else:
            poly.basecoords[changeAfter] = x, keepY
        set_display_coords(poly)

    def _update_line(self):
        # save verts because polygon gets deleted when figure is closed
        for poly in self.polys.itervalues():
            self.last_vert_ind = len(poly.xy) - 1
            poly.lines.set_data(zip(*poly.xy))
            poly.handle.set_data(zip(*calc_handle_coords(poly)))
            pass

    def get_ind_under_cursor(self, event):
        'get the index of the vertex under cursor if within max_ds tolerance'

        def get_ind_and_dist(poly):
            if poly is None:
                return (None, -1)
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
        ind_dist_list = []
        #for poly in self.poly_list:
        #    if poly is not None:
        #        ind_dist_list.append(get_ind_and_dist(poly))
        ind_dist_list = [get_ind_and_dist(poly) for poly in self.polys.itervalues()]
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
        """write a callback to redraw viz for bbox_list"""
        def get_bbox_list():
            bbox_list = []
            #theta_list = []
            for poly in self.polys.itervalues():
                assert poly is not None
#                if poly is None:
#                    bbox_list.append(None)
#                else:
                x = min(poly.xy[0][0], poly.xy[1][0], poly.xy[2][0], poly.xy[3][0])
                y = min(poly.xy[0][1], poly.xy[1][1], poly.xy[2][1], poly.xy[3][1])
                w = max(poly.xy[0][0], poly.xy[1][0], poly.xy[2][0], poly.xy[3][0]) - x
                h = max(poly.xy[0][1], poly.xy[1][1], poly.xy[2][1], poly.xy[3][1]) - y
                bbox_list.append((int(x), int(y), int(w), int(h), poly.theta))
                #theta_list.append(poly.theta)
            return bbox_list# , theta_list

        def send_back_rois():
            #point_list = self.load_points()
            #theta_list = self.theta_list
            #new_bboxes = verts_to_bbox(point_list)
            print("send_back_rois")
            bbox_list = get_bbox_list()
            deleted_list = []
            changed_list = []
            new_list = []
            for i, bbox_theta in enumerate(zip(self.original_bbox_list, self.original_theta_list)):
                if bbox_theta not in bbox_list:
                    deleted_list.append(i)
#                elif bbox_list[i] != self.original_list[i]:
#                    changed_list.append((i, bbox_list[i]))
#            for i in range(len(self.original_list), len(self.poly_list)):
#                if bbox_list[i] is not None:
#                    new_list.append(bbox_list[i])
            new_list = filter(lambda bbox_theta: bbox_theta not in zip(self.original_bbox_list, self.original_theta_list), bbox_list)
            #print("Deleted")
            #for bbox in deleted_list:
            #    print(bbox)
            #print("Changed")
            #for bbox in changed_list:
            #    print(bbox)
            #print("New")
            #for bbox in new_list:
            #    print(bbox)
            #print("send_back_rois() completed")
            #self.callback(self.img_ind, new_bboxes, theta_list)
            self.callback(deleted_list, changed_list, new_list)

        if self.callback is not None:
            send_back_rois()
        #else:
            #just print the updated points
            #self.load_points()
            #print(self.poly_list)
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
            return

        print('Accept Over')
        df2.close_figure(self.fig)
        #plt.close(self.fnum)
        #plt.draw()

    def get_mask(self, shape):
        """Return image mask given by mask creator"""
        mask_list = [verts_to_mask(shape, poly.xy) for poly in self.polys.itervalues()]
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


# def update_lists(self, img_ind, new_verts_list, new_thetas_list, new_indices_list):
#         """for each image: know if any bboxes changed, if any have been deleted, know if any new bboxes
#         pass all of this information to a function callback, so Jon can use the information"""
#         print("before: ",self.rids_list[img_ind])
#         self.rids_list[img_ind] = new_rids
#         print("after: ", self.rids_list[img_ind])

#         """add function call for redrawing the ROIs"""


def ROI_creator(img, verts_list):  # add callback as variable
    print('*** START DEMO ***')

    if verts_list is None:
        verts_list = [default_vertices(img)]
    # else:
    #     for verts in verts_list:
    #         if (len(verts) is not 5):
    #             print("verts list is not of correct length. ", len(verts))
    #             return

    if img is None:
        try:
            img_url = 'http://i.imgur.com/Vq9CLok.jpg'
            img_fpath = utool.grab_file_url(img_url)
            #img = mpimg.imread(img_fpath)
            from vtool import image as gtool
            img = gtool.imread(img_fpath)
        except Exception as ex:
            print('cant read zebra: %r' % ex)
            img = np.random.uniform(0, 255, size=(100, 100))
    #test_bbox = verts_to_bbox(verts_list)
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
    #verts = [[0,0,400,400]]
    verts = [((0, 400), (400, 400), (400, 0), (0, 0), (0, 400)),
             ((400, 700), (700, 700), (700, 400), (400, 400), (400, 700))]
    ROI_creator(None, verts_list=verts)
