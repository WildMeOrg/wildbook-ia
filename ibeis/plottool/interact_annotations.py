"""
Interactive tool to draw mask on an image or image-like array.

TODO:
    * need concept of subannotation
    * need to take options on a right click of an annotation
    * add support for arbitrary polygons back in .
    * rename species_list to label_list or category_list

Notes:
    3. Change bounding box and update continuously to the original image the
    new ANNOTATIONs

    2. Make new window and frames inside, double click to pull up normal window
    with editing start with just taking in 6 images and ANNOTATIONs

    1. ANNOTATION ID number, then list of 4 tuples

References:
    Adapted from matplotlib/examples/event_handling/poly_editor.py
    Jan 9 2014: taken from: https://gist.github.com/tonysyu/3090704

CommandLine:
    python -m plottool.interact_annotations --test-test_interact_annots --show
"""
from __future__ import absolute_import, division, print_function
import six
import matplotlib as mpl
#import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math as math
#import functools
from functools import partial
from plottool import abstract_interaction  # TODO inherit from this
#from matplotlib import nxutils  # Deprecated
#from matplotlib.mlab import dist_point_to_segment
# Scientific
import numpy as np
import vtool as vt
import utool as ut
import re
# FIXME: REMOVE IBEIS DEPENDENCY
from plottool import draw_func2 as df2
from six.moves import zip
#ut.noinject(__name__, '[interact_annotations]')
print, rrr, profile = ut.inject2(__name__, '[interact_annotations]')


DEFAULT_SPECIES_TAG = '____'
# FIXE THESE TO BE GENERIC
ACCEPT_SAVE_HOTKEY        = None  # 'ctrl+a'
ADD_RECTANGLE_HOTKEY      = 'ctrl+a'  # 'ctrl+d'
ADD_RECTANGLE_FULL_HOTKEY = 'ctrl+f'
DEL_RECTANGLE_HOTKEY      = 'ctrl+d'  # 'ctrl+r'
TOGGLE_LABEL_HOTKEY       = 'ctrl+t'

HACK_OFF_SPECIES_TYPING = True
if HACK_OFF_SPECIES_TYPING:
    ADD_RECTANGLE_HOTKEY      = 'a'  # 'ctrl+d'
    ADD_RECTANGLE_FULL_HOTKEY = 'f'
    DEL_RECTANGLE_HOTKEY      = 'd'  # 'ctrl+r'
    TOGGLE_LABEL_HOTKEY       = 't'

NEXT_IMAGE_HOTKEYS  = ['right', 'pagedown']
PREV_IMAGE_HOTKEYS  = ['left', 'pageup']


def pretty_hotkey_map(hotkeys):
    if hotkeys is None:
        return ''
    hotkeys = [hotkeys] if not isinstance(hotkeys, list) else hotkeys
    mapping = {
        #'right': 'right arrow',
        #'left':  'left arrow',
    }
    mapped_hotkeys = [mapping.get(hk, hk) for hk in hotkeys]
    hotkey_str = '(' + ut.conj_phrase(mapped_hotkeys, 'or') + ')'
    return hotkey_str


TAU = np.pi * 2  # References: tauday.com


def _nxutils_points_inside_poly(points, verts):
    """ nxutils is depricated """
    path = mpl.path.Path(verts)
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
    #return vt.verts_from_bbox(bbox)
    # TODO: Can use vtool for this
    (x, y, w, h) = bbox
    verts = np.array([(x + 0, y + h),
                      (x + 0, y + 0),
                      (x + w, y + 0),
                      (x + w, y + h), ], dtype=np.float32)
    return verts


def basecoords_to_bbox(basecoords):
    x = min(basecoords[0][0], basecoords[1][0], basecoords[2][0],
            basecoords[3][0])
    y = min(basecoords[0][1], basecoords[1][1], basecoords[2][1],
            basecoords[3][1])
    w = max(basecoords[0][0], basecoords[1][0], basecoords[2][0],
            basecoords[3][0]) - x
    h = max(basecoords[0][1], basecoords[1][1], basecoords[2][1],
            basecoords[3][1]) - y
    bbox = (x, y, w, h)
    return bbox


def points_center(pts):
    # the polygons have the first point listed twice in order for them to be
    # drawn as closed, but that point shouldn't be counted twice for computing
    # the center (hence the [:-1] slice)
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
    """
    References:
        http://www.euclideanspace.com/maths/geometry/affine/aroundPoint/matrix2d/
    """
    # TODO: Can use vtool for this
    sin, cos, array = np.sin, np.cos, np.array
    augpts = array([array((x, y, 1)) for (x, y) in points])
    ct = cos(theta)
    st = sin(theta)
    # correct matrix obtained from
    rot_mat = array(
        [(ct, -st, ax - ct * ax + st * ay),
         (st,  ct, ay - st * ax - ct * ay),
         ( 0,   0,                      1)]
    )
    return [(x, y) for (x, y, z) in rot_mat.dot(augpts.T).T]


def calc_display_coords(oldcoords, theta):
    return rotate_points_around(oldcoords, theta, *points_center(oldcoords))


def set_display_coords(poly):
    poly.xy = calc_display_coords(poly.basecoords, poly.theta)
    poly.species_tag.set_position(calc_tag_position(poly))
    #print(poly.species_tag.get_position())


def calc_tag_position(poly):
    points = [[
        max(list(zip(*poly.basecoords))[0]),
        min(list(zip(*poly.basecoords))[1])
    ]]
    tagpos = rotate_points_around(points, poly.theta, *polygon_center(poly))[0]
    return tagpos


def is_within_distance_from_line(dist, pt, line):
    """
    References:
        http://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
        http://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
    """
    x0, y0 = pt
    p1, p2 = line
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    # # This doesn't work due to being for lines, not line segments, leading to
    # potentially confusing behavior
    # numer = abs((dy * x0) - (dx * y0) - (x1 * y2) + (x2 * y1))
    # denom = math.sqrt((dx ** 2) + (dy ** 2))
    # distance = numer / denom
    # adapted from
    squared_mag_of_delta = (dx * dx) + (dy * dy)
    interpolation_factor = (
        (x0 - x1) * dx + (y0 - y1) * dy) / float(squared_mag_of_delta)
    # clamp to [0, 1]
    interpolation_factor = max(0, min(1, interpolation_factor))
    nx = x1 + interpolation_factor * dx
    ny = y1 + interpolation_factor * dy
    ndx = nx - x0
    ndy = ny - y0
    squared_dist = (ndx * ndx) + (ndy * ndy)
    return squared_dist < (dist * dist)


def calc_handle_coords(poly):
    cx, cy = polygon_center(poly)
    w, h = polygon_dims(poly)
    x0, y0 = cx, (cy - (h / 2))  # start at top edge
    MIN_HANDLE_LENGTH = 25
    HANDLE_LENGTH = max(MIN_HANDLE_LENGTH, (h / 4))
    x1, y1 = (x0, y0 - HANDLE_LENGTH)
    pts = [(x0, y0), (x1, y1)]
    pts = rotate_points_around(pts, poly.theta, cx, cy)
    return pts


def make_handle_line(poly):
    _xs, _ys = list(zip(*calc_handle_coords(poly)))
    line_width = 4
    line_color = (0, 1, 0)
    color = np.array(line_color)
    marker_face_color = line_color
    line_kwargs = {'lw': line_width, 'color': color, 'mfc': marker_face_color}
    lines = plt.Line2D(_xs, _ys, marker='o', alpha=1, animated=True,
                       **line_kwargs)
    return lines


def within_epsilon(x, y):
    return x - y < .000001


def meets_minimum_width_and_height(coords):
    """
    Depends on hardcoded indices, which is inelegant, but
    we're already depending on those for the FUDGE_FACTORS
    array above
    1----2
    |    |
    0----3
    """
    MIN_W = 5
    MIN_H = 5
    # the seperate 1 and 2 variables are not strictly necessary, but
    # provide a sanity check to ensure that we're dealing with the
    # right
    # shape
    width1 = coords[3][0] - coords[0][0]
    width2 = coords[2][0] - coords[1][0]
    assert within_epsilon(width1, width2), (
        'w1: %r, w2: %r' % (width1, width2))
    height1 = coords[0][1] - coords[1][1]
    height2 = coords[3][1] - coords[2][1]
    assert within_epsilon(height1, height2), (
        'h1: %r, h2: %r' % (height1, height2))
    #print('w, h = (%r, %r)' % (width1, height1))
    return (MIN_W < width1) and (MIN_H < height1)


def default_vertices(img, polys=None, mouseX=None, mouseY=None):
    """Default to rectangle that has a quarter-width/height border."""
    (h, w) = img.shape[0:2]
    # Center the new verts around wherever the mouse is
    if mouseX is not None and mouseY is not None:
        center_x = mouseX
        center_h = mouseY
    else:
        center_x = w // 2
        center_h = h // 2

    if polys is not None and len(polys) > 0:
        # Use the largest polygon size as the default verts
        wh_list = np.array([basecoords_to_bbox(poly.xy)[2:4]
                            for poly in six.itervalues(polys)])
        w_, h_ = wh_list.max(axis=0) // 2
    else:
        # If no poly exists use 1/4 of the image size
        w_, h_ = (w // 4, h // 4)
    # Get the x/y extents by offseting the centers
    x1, x2 = np.array([center_x, center_x]) + (w_ * np.array([-1, 1]))
    y1, y2 = np.array([center_h, center_h]) + (h_ * np.array([-1, 1]))
    # Clip to bounds
    x1 = max(x1, 1)
    y1 = max(y1, 1)
    x2 = min(x2, w - 1)
    y2 = min(y2, h - 1)
    return ((x1, y1), (x1, y2), (x2, y2), (x2, y1))


def test_interact_annots():
    r"""
    CommandLine:
        python -m plottool.interact_annotations --test-test_interact_annots --show

    Example:
        >>> # ENABLE_DOCTEST
        >>> from plottool.interact_annotations import *  # NOQA
        >>> import plottool as pt
        >>> # build test data
        >>> # execute function
        >>> self = test_interact_annots()
        >>> # verify results
        >>> print(self)
        >>> pt.show_if_requested()
    """
    verts_list = [((0, 400), (400, 400), (400, 0), (0, 0), (0, 400)),
                  ((400, 700), (700, 700), (700, 400), (400, 400), (400, 700))]
    print('[interact_annot] *** START DEMO ***')

    #if img is None:
    try:
        img_url = 'http://i.imgur.com/Vq9CLok.jpg'
        img_fpath = ut.grab_file_url(img_url)
        #img = mpimg.imread(img_fpath)
        from vtool import image as gtool
        img = gtool.imread(img_fpath)
    except Exception as ex:
        print('[interact_annot] cant read zebra: %r' % ex)
        img = np.random.uniform(0, 255, size=(100, 100))
    self = ANNOTATIONInteraction(img, verts_list=verts_list, fnum=0)  # NOQA
    return self
    # Do interaction
    #
    # Make mask from selection
    #mask = self.get_mask(img.shape)
    # User must close previous figure
    # Modify the image with the mask
    #masked_img = apply_mask(img, mask)
    # show the modified image
    #plt.imshow(masked_img)
    #plt.title('Region outside of mask is darkened')
    #print('show2')


# TODO: incorporate this
AbstractInteraction = abstract_interaction.AbstractInteraction
#BASE_CLASS = abstract_interaction.AbstractInteraction
BASE_CLASS = object


@six.add_metaclass(ut.ReloadingMetaclass)
class ANNOTATIONInteraction(BASE_CLASS):
    """
    An interactive polygon editor.

    SeeAlso:
        ibeis.viz.interact.interact_annotations2
        (ensure that any updates here are propogated there)

    Args:
        verts_list (list) : list of lists of (float, float)
            List of (x, y) coordinates used as vertices of the polygon.

        max_ds (float) : float
            Max pixel distance to count as a vertex hit.

    KeyBindings:
        't' : toggle vertex markers on and off.  When vertex markers are on,
              you can move them, delete them
        'd' : delete the vertex under point
        'i' : insert a vertex at point.  You must be within max_ds of the
              line connecting two existing vertices
    """

    # --- Initialization and Figure Widgets

    def __init__(self, img, img_ind=None, commit_callback=None,
                 verts_list=None,
                 bbox_list=None,
                 theta_list=None,
                 species_list=None,
                 metadata_list=None,
                 max_ds=10,
                 line_width=4, line_color=(1, 1, 1), face_color=(0, 0, 0),
                 fnum=None, default_species=DEFAULT_SPECIES_TAG,
                 next_callback=None, prev_callback=None, do_mask=False,
                 valid_species=[], **kwargs):
        if BASE_CLASS is not object:
            super(ANNOTATIONInteraction, self).__init__(**kwargs)
        else:
            if fnum is None:
                fnum = df2.next_fnum()
            abstract_interaction.register_interaction(self)
            ut.inject_func_as_method(self, AbstractInteraction.append_button.im_func)
            ut.inject_func_as_method(self, AbstractInteraction.show_popup_menu.im_func)
            self.scope = []

        self.valid_species = valid_species
        self.commit_callback = commit_callback  # commit_callback
        self.but_width = .14
        #self.but_height = .08
        self.next_prev_but_height = .08
        self.but_height = self.next_prev_but_height - .01
        self.callback_funcs = dict([
            ('close_event', self.on_close),
            ('draw_event', self.draw_callback),
            ('button_press_event', self.on_click),
            ('button_release_event', self.on_click_release),
            ('figure_leave_event', self.on_figure_leave),
            ('key_press_event', self.on_key_press),
            ('motion_notify_event', self.on_motion),
            ('pick_event', self.onpick),
            #('resize_event', self.on_resize),
        ])
        self.mpl_callback_ids = {}
        self.img = img
        self.show_species_tags = True
        def reinitialize_variables():
            self.do_mask = do_mask
            self.img_ind = img_ind
            self.species_tag = default_species
            self.showverts = True
            self.max_ds = max_ds
            self.fc_default = face_color
            self.mouseX = None  # mouse X coordinate
            self.mouseY = None  # mouse Y coordinate
            self.indX = None
            self.indY = None
            self.leftbutton_is_down = False
            self.canUncolor = False    # flag if the polygon SHOULD be active
            self._autoinc_polynum = 0  # num polys in image
            self._polyHeld = False                # if any poly is active
            self._currently_selected_poly = None  # active polygon
            self.background = None  # Something Jon added
        reinitialize_variables()
        # hack involving exploting lexical scoping to save defaults for a
        # restore operation
        self.reinitialize_variables = reinitialize_variables
        self.handle_matplotlib_initialization(fnum=fnum)
        assert verts_list is None or bbox_list is None, 'only one can be specified'
        # bbox_list will get converted to verts_list
        if verts_list is not None:
            bbox_list = vt.bboxes_from_vert_list(verts_list)
        if bbox_list is not None:
            verts_list = [bbox_to_verts(bbox) for bbox in bbox_list]
        if theta_list is None:
            theta_list = [0 for verts in verts_list]
        if species_list is None:
            species_list = [self.species_tag for verts in verts_list]
        if metadata_list is None:
            metadata_list = [None for verts in verts_list]

        # Create the list of polygons
        self.handle_polygon_creation(bbox_list, theta_list, species_list, metadata_list)
        self._ind = None  # the active vert
        self.currently_rotating_poly = None

        self.mpl_callback_ids = {}
        assert self.fig.canvas is self.fig.ax.figure.canvas, 'wow. something is weird'
        self.connect_mpl_callbacks(self.fig.canvas)

        self.add_action_buttons()
        self.update_callbacks(next_callback, prev_callback)

    def handle_matplotlib_initialization(self, fnum=None,
                                         instantiate_window=True):
        if instantiate_window:
            self.fig = df2.figure(fnum=fnum, doclf=True, docla=True)
            df2.close_figure(self.fig)
            self.fig = df2.figure(fnum=fnum, doclf=True, docla=True)
        self.fig.clear()
        self.fig.clf()
        #self.fig.cla()
        #ut.qflag()
        self.fnum = fnum
        #print(self.fnum)
        #ax = plt.subplot(111)
        ax = df2.gca()
        self.fig.ax = ax
        self.ax = ax
        df2.remove_patches(self.fig.ax)
        df2.imshow(self.img, fnum=fnum)

        ax.set_clip_on(False)
        ax.set_title(('\n'.join([
            'Click and drag to select/move/resize/orient an ANNOTATION',
            #'Press enter to clear the species tag of the selected ANNOTATION',
            'Press tab to cycle through annotation species',
            #'Type to edit the ANNOTATION species (press tab to autocomplete)'
        ])))

    def add_action_buttons(self):
        self.append_button(
            'Add Annotation\n' + pretty_hotkey_map(ADD_RECTANGLE_HOTKEY),
            rect=[0.18, 0.015, self.but_width, self.but_height],
            callback=self.add_new_annot
        )
        self.append_button(
            'Add Full Annotation\n' + pretty_hotkey_map(ADD_RECTANGLE_FULL_HOTKEY),
            rect=[0.34, 0.015, self.but_width, self.but_height],
            callback=partial(self.add_new_annot, full=True)
        )
        self.append_button(
            'Delete Annotation\n' + pretty_hotkey_map(DEL_RECTANGLE_HOTKEY),
            rect=[0.50, 0.015, self.but_width, self.but_height],
            callback=self.delete_current_annot
        )
        self.append_button(
            'Save and Exit\n' + pretty_hotkey_map(ACCEPT_SAVE_HOTKEY),
            rect=[0.66, 0.015, self.but_width, self.but_height],
            callback=self.save_and_exit
        )

    def disconnect_mpl_callbacks(self, canvas):
        """ disconnects all connected matplotlib callbacks """
        for name, callbackid in six.iteritems(self.mpl_callback_ids):
            canvas.mpl_disconnect(callbackid)
        self.mpl_callback_ids = {}

    def connect_mpl_callbacks(self, canvas):
        """ disconnects matplotlib callbacks specified in the
        self.mpl_callback_ids dict """
        #http://matplotlib.org/1.3.1/api/backend_bases_api.html
        # Create callback ids
        self.disconnect_mpl_callbacks(canvas)
        self.mpl_callback_ids = {
            name: canvas.mpl_connect(name, func)
            for name, func in six.iteritems(self.callback_funcs)
        }
        self.fig.canvas = canvas

    # --- Updates

    def update_callbacks(self, next_callback, prev_callback):
        self.prev_callback = prev_callback
        self.next_callback = next_callback
        # Hack because the callbacks actually need to be wrapped
        _next_callback = None if self.next_callback is None else self.next_image
        _prev_callback = None if self.prev_callback is None else self.prev_image
        self.append_button(
            'Previous Image\n' + pretty_hotkey_map(PREV_IMAGE_HOTKEYS),
            rect=[0.02, 0.01, self.but_width, self.next_prev_but_height],
            callback=_prev_callback,
        )
        self.append_button(
            'Next Image\n' + pretty_hotkey_map(NEXT_IMAGE_HOTKEYS),
            rect=[0.82, 0.01, self.but_width, self.next_prev_but_height],
            callback=_next_callback,
        )

    def update_image_and_callbacks(self, img, bbox_list, theta_list,
                                   species_list, metadata_list, next_callback,
                                   prev_callback):
        self.disconnect_mpl_callbacks(self.fig.canvas)
        for poly in six.itervalues(self.polys):
            poly.remove()
        self.polys = {}
        self.reinitialize_variables()
        self.img = img
        self.handle_matplotlib_initialization(
            fnum=self.fnum, instantiate_window=False)
        self.handle_polygon_creation(bbox_list, theta_list, species_list,
                                     metadata_list)
        self.add_action_buttons()
        self.fig.canvas.draw()
        self.connect_mpl_callbacks(self.fig.canvas)
        self.update_callbacks(next_callback, prev_callback)
        print('[interact_annot] drawing')
        self.fig.canvas.draw()
        self.update_UI()

    def update_UI(self):
        self._update_line()
        self.fig.canvas.restore_region(self.background)
        self.draw_artists()
        self.fig.canvas.blit(self.fig.ax.bbox)

    def update_colors(self, poly_ind):
        if poly_ind is None or poly_ind < 0:
            print('[interact_annot] WARNING: poly_ind is %r in update_colors' %
                  poly_ind)
            return
        # Remove unselected colors
        for poly in six.itervalues(self.polys):
            line = poly.lines
            line_color = line.get_color()
            desel_color = df2.WHITE if poly.is_orig else df2.LIGHTGRAY
            if np.any(line_color != np.array(desel_color)):
                line.set_color(np.array(desel_color))
        # Add selected color
        sel_poly = self.polys[poly_ind]
        sel_color = df2.ORANGE if sel_poly.is_orig else df2.LIGHT_BLUE
        sel_poly.lines.set_color(sel_color)
        plt.draw()

    # --- Data Matainence / Other

    def handle_polygon_creation(self, bbox_list, theta_list, species_list,
                                metadata_list):
        """ Maintain original input """
        assert bbox_list is not None
        if theta_list is None:
            theta_list = [0.0 for _ in range(len(bbox_list))]
        if species_list is None:
            species_list = ['' for _ in range(len(bbox_list))]
        assert len(bbox_list) == len(theta_list), 'inconconsitent data1'
        assert len(bbox_list) == len(species_list), 'inconconsitent data2'
        assert len(bbox_list) == len(metadata_list), 'inconconsitent data2'
        self.original_indices       = list(range(len(bbox_list)))
        self.original_bbox_list     = bbox_list
        self.original_theta_list    = theta_list
        self.original_species_list  = species_list
        self.original_metadata_list = metadata_list
        # Convert bbox to verticies
        verts_list = [bbox_to_verts(bbox) for bbox in bbox_list]
        for verts in verts_list:
            for vert in verts:
                self.enforce_dims(vert)
        # Create polygons
        poly_list = [self.new_polygon(verts, theta, species, is_orig=True,
                                      metadata=metadata)
                     for (verts, theta, species, metadata) in
                     zip(verts_list, theta_list, species_list, metadata_list)]
        self.polys = {poly.num: poly for poly in poly_list}
        if len(self.polys) != 0:
            wh_list = np.array([basecoords_to_bbox(poly.xy)[2:4] for poly in
                                six.itervalues(self.polys)])
            poly_index = list(self.polys.keys())[wh_list.prod(axis=1).argmax()]
            self._currently_selected_poly = self.polys[poly_index]
            self.update_colors(poly_index)
            self._update_line()
        else:
            self._currently_selected_poly = None
        # Add polygons and lines to the axis
        for poly in six.itervalues(self.polys):
            self.fig.ax.add_patch(poly)
            self.fig.ax.add_line(poly.lines)
            self.fig.ax.add_line(poly.handle)
        # Give polygons mpl change callbacks
        for poly in six.itervalues(self.polys):
            poly.add_callback(self.poly_changed)

    def check_dims(self, coords, margin=0.5):
        """ checks if bounding box dims are ok """
        num_out = 0
        xlim = self.fig.ax.get_xlim()
        ylim = self.fig.ax.get_ylim()
        if coords[0] < xlim[0] + margin:
            num_out += 1
            #coords[0] = xlim[0]
        if coords[0] > xlim[1] - margin:
            num_out += 1
            #coords[0] = xlim[1]
        if coords[1] < ylim[1] + margin:
            num_out += 1
            #coords[1] = ylim[1]
        if coords[1] > ylim[0] - margin:
            num_out += 1
            #coords[1] = ylim[0]
        # Allow the bounding box to go off the image
        # so orientations can be done correctly
        #return num_out == 0
        return num_out <= 3

    def load_points(self):
        return [poly.xy for poly in six.itervalues(self.polys)]

    def enforce_dims(self, coords, margin=0.5):
        """
        ONLY USE THIS ON UNROTATED RECTANGLES, as to do otherwise may yield
        arbitrary polygons
        """
        xlim = self.fig.ax.get_xlim()
        ylim = self.fig.ax.get_ylim()
        if coords[0] < xlim[0] + margin:
            coords[0] = xlim[0] + margin
        if coords[0] > xlim[1] - margin:
            coords[0] = xlim[1] - margin
        if coords[1] < ylim[1] + margin:
            coords[1] = ylim[1] + margin
        if coords[1] > ylim[0] - margin:
            coords[1] = ylim[0] - margin
        return True

    def clip_vert_to_bounds(self, coords, margin=0):
        xlim = self.fig.ax.get_xlim()
        ylim = self.fig.ax.get_ylim()
        def clamp(lims, val):
            return max(lims[0], min(lims[1], val))
        return np.array((clamp(xlim, coords[0]), clamp(ylim, coords[1])))

    def check_valid_coords(self, coords_list):
        valid = True
        for coord in coords_list:
            if not self.check_dims(coord):
                valid = False
        return valid

    def _update_line(self):
        """
        save verts because polygon gets deleted when figure is closed
        """
        for poly in six.itervalues(self.polys):
            self.last_vert_ind = len(poly.xy) - 1
            poly.lines.set_data(list(zip(*poly.xy)))
            poly.handle.set_data(list(zip(*calc_handle_coords(poly))))
            pass

    def get_ind_under_cursor(self, event):
        """
        get the index of the vertex under cursor if within max_ds tolerance
        """
        #print('[interact_annotion] enter get_ind_under_cursor')
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
        ind_dist_list = [
            (polyind, get_ind_and_dist(poly))
            for (polyind, poly) in six.iteritems(self.polys)]
        min_dist = None
        min_ind  = None
        sel_polyind = None
        for polyind, (ind, dist) in ind_dist_list:
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

    def new_polygon(self, verts, theta, species,
                    face_color=(0, 0, 0),
                    line_color=(1, 1, 1),
                    line_width=4,
                    is_orig=False,
                    metadata=None):
        """ verts - list of (x, y) tuples """
        # create new polygon from verts
        poly = mpl.patches.Polygon(
            verts, animated=True, fc=face_color, ec='none', alpha=0,
            picker=True)
        # register this polygon
        poly.num = self.next_polynum()
        #poly.status = 'orig' if is_orig else 'new'
        poly.is_orig = is_orig
        poly.theta = theta
        poly.basecoords = poly.xy
        poly.xy = calc_display_coords(poly.basecoords, poly.theta)
        poly.lines = self.make_lines(poly, line_color, line_width)
        poly.handle = make_handle_line(poly)
        tagpos = calc_tag_position(poly)
        poly.species_tag = self.fig.ax.text(
            tagpos[0], tagpos[1], species,
            bbox={'facecolor': 'white', 'alpha': 1})
        poly.species_tag.remove()  # eliminate "leftover" copies
        poly.metadata = metadata
        # put in previous text and tabcomplete list for autocompletion
        poly.tctext = ''
        poly.tab_list = self.valid_species
        poly.tcindex = 0
        poly.last_idx = 2
        return poly

    def make_lines(self, poly, line_color, line_width):
        """ verts - list of (x, y) tuples """
        _xs, _ys = list(zip(*poly.xy))
        color = np.array(line_color)
        marker_face_color = line_color
        line_kwargs = {'lw': line_width, 'color': color,
                       'mfc': marker_face_color}
        lines = plt.Line2D(_xs, _ys, marker='o', alpha=1,
                           animated=True, **line_kwargs)
        return lines

    def get_mask(self, shape):
        """Return image mask given by mask creator"""
        mask_list = [verts_to_mask(shape, poly.xy)
                     for poly in six.itervalues(self.polys)]
        if len(mask_list) == 0:
            print('[interact_annot] No polygons to make mask out of')
            return 0
        mask = mask_list[0]
        for mask_ in mask_list:
            mask = np.maximum(mask, mask_)
        return mask

    def get_most_recently_added_poly(self):
        if len(self.polys) == 0:
            return (None, None)
        else:
            # most recently added polygon has the highest index
            poly_ind = max(list(self.polys.keys()))
            return poly_ind, self.polys[poly_ind]

    # --- Actions

    def delete_current_annot(self, event=None):
        """
        Removes an annotation
        """
        if self._currently_selected_poly is None:
            print('[interact_annot] No polygon selected to delete')
            return
        poly = self._currently_selected_poly
        lineNumber = poly.num
        print('[interact_annot] delete annot. length=%d num=%d' % (
            len(self.polys), lineNumber))
        self.polys.pop(lineNumber)
        # remove the poly from the figure itself
        poly.remove()
        #reset anything that has to do with current poly
        _tup = self.get_most_recently_added_poly()
        poly_ind, self._currently_selected_poly = _tup
        self._polyHeld = False
        if poly_ind is not None:
            self.update_colors(poly_ind)
        plt.draw()

    def add_new_annot(self, event=None, full=False):
        """ Adds a new annotation to the image """
        if full:
            (h, w) = self.img.shape[0:2]
            x1 = 1
            y1 = 1
            x2 = w - 1
            y2 = h - 1
            coords = ((x1, y1), (x1, y2), (x2, y2), (x2, y1))
        else:
            if self._currently_selected_poly is not None:
                defaultshape_polys = {
                    self._currently_selected_poly.num:
                    self._currently_selected_poly
                }
            else:
                defaultshape_polys = self.polys
            coords = default_vertices(self.img, defaultshape_polys,
                                      self.mouseX, self.mouseY)

        poly = self.new_polygon(coords, 0, self.species_tag)
        #<hack reason="brittle resizing algorithm that doesn't work unless the
        #points are in the right order, see resize rectangle
        # and meets_minimum_width_and_height">
        bbox = basecoords_to_bbox(poly.basecoords)
        poly.basecoords = bbox_to_verts(bbox)
        set_display_coords(poly)
        #</hack>

        self.polys[poly.num] = poly
        self.fig.ax.add_patch(poly)
        self._update_line()

        self.fig.ax.add_line(poly.lines)
        self.fig.ax.add_line(poly.handle)

        poly.add_callback(self.poly_changed)
        self._ind = None  # the active vert
        _tup = self.get_most_recently_added_poly()
        poly_ind, self._currently_selected_poly = _tup
        assert poly_ind == poly.num, 'ind %r, num %r' % (poly_ind, poly.num)
        self.update_colors(poly_ind)
        plt.draw()

    def rotate_rectangle(self, poly, dtheta):
        coords_lis = calc_display_coords(poly.basecoords, poly.theta + dtheta)
        if self.check_valid_coords(coords_lis):
            poly.theta += dtheta
            set_display_coords(poly)

    def move_rectangle(self, poly, dx, dy):
        new_coords = [(x + dx, y + dy) for (x, y) in poly.basecoords]
        coords_list = calc_display_coords(new_coords, poly.theta)
        if self.check_valid_coords(coords_list):
            poly.basecoords = new_coords
            set_display_coords(poly)

    def resize_rectangle(self, poly, x, y, idx):
        """
        Resize a rectangle using idx as the given anchor point

        Args:
            poly (?):
            x (?):
            y (ndarray):  labels
            idx (?):

        Returns:
            ?:

        CommandLine:
            python -m plottool.interact_annotations --exec-resize_rectangle --show

        Example:
            >>> # DISABLE_DOCTEST
            >>> from plottool.interact_annotations import *  # NOQA
            >>> self = test_interact_annots()
            >>> (h, w) = self.img.shape[0:2]
            >>> x1, y1 = 10, 10
            >>> x2, y2 = w - 10,  h - 10
            >>> coords = ((x1, y1), (x1, y2), (x2, y2), (x2, y1))
            >>> #poly = self.new_polygon(coords, 0, self.species_tag)
            >>> poly = self._currently_selected_poly
            >>> x = 3 * w / 4
            >>> y = 3 * h / 4
            >>> idx = 3
            >>> self.resize_rectangle(poly, x, y, idx)
            >>> self.update_UI()
            >>> import plottool as pt
            >>> pt.show_if_requested()
        """
        #print('resize_rectangle')
        # TODO: allow resize by middle click to scale from the center
        if poly is None:
            return
        poly.last_idx = idx

        def distance(x, y):
            return math.sqrt(x ** 2 + y ** 2)

        def polarDelta(p1, p2):
            mag = distance(p2[0] - p1[0], p2[1] - p1[1])
            #mag = vt.L2(p1, p2)
            theta = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
            return [mag, theta]

        def apply_polarDelta(poldelt, cart):
            newx = cart[0] + (poldelt[0] * math.cos(poldelt[1]))
            newy = cart[1] + (poldelt[0] * math.sin(poldelt[1]))
            return (newx, newy)

        def isSegmentBetweenCoordsVertical(c1, c2):
            return c1[0] == c2[0]  # x coordinates are the same

        def rad2deg(t):
            return t * 360 / TAU

        # the minus one is because the last coordinate is duplicated (by
        # matplotlib) to get a closed polygon
        tmpcoords = poly.xy[:-1]
        def wrapIndex(i):
            return (i % len(tmpcoords))

        previdx, nextidx = wrapIndex(idx - 1), wrapIndex(idx + 1)
        #oppidx = wrapIndex(idx + 2)
        (dx, dy) = (x - poly.xy[idx][0], y - poly.xy[idx][1])

        tmpcoords = poly.xy[:-1]

        # this algorithm worked the best of the ones I tried, but needs
        # "experimentally determined constants" to work properly, since I
        # failed to properly derive them in the allotted time
        FUDGE_FACTORS = {0: -(TAU / 4),
                         1: 0,
                         2: (TAU / 4),
                         3: (TAU / 2)}

        polar_idx2prev = polarDelta(tmpcoords[idx], tmpcoords[previdx])
        polar_idx2next = polarDelta(tmpcoords[idx], tmpcoords[nextidx])
        tmpcoords[idx] = (tmpcoords[idx][0] + dx, tmpcoords[idx][1] + dy)
        mag_delta = distance(dx, dy)
        theta_delta = math.atan2(dy, dx)
        poly_theta = poly.theta + FUDGE_FACTORS.get(idx, 0)
        theta_rot = theta_delta - (poly_theta + TAU / 4)
        rotx = mag_delta * math.cos(theta_rot)
        roty = mag_delta * math.sin(theta_rot)
        polar_idx2prev[0] -= rotx
        polar_idx2next[0] += roty
        tmpcoords[previdx] = apply_polarDelta(polar_idx2prev, tmpcoords[idx])
        tmpcoords[nextidx] = apply_polarDelta(polar_idx2next, tmpcoords[idx])

        # rotate the points by -theta to get the "unrotated" points for use as
        # basecoords
        tmpcoords = rotate_points_around(tmpcoords, -poly.theta,
                                         *polygon_center(poly))
        # ensure the poly is closed, matplotlib might do this, but I'm not sure
        # if it preserves the ordering we depend on, even if it does add the
        # point
        tmpcoords = tmpcoords[:] + [tmpcoords[0]]

        dispcoords = calc_display_coords(tmpcoords, poly.theta)

        if (self.check_valid_coords(dispcoords) and
             meets_minimum_width_and_height(tmpcoords)):
            poly.basecoords = tmpcoords
        #else:
        #    print('[pt] Invalid resize poly')

        set_display_coords(poly)

    def toggle_species_label(self):
        print('[interact_annot] toggle_species_label()')
        self.show_species_tags = not self.show_species_tags
        self.update_UI()

    def next_image(self, event):
        if self.next_callback is not None:
            self.next_callback()

    def prev_image(self, event):
        if self.prev_callback is not None:
            self.prev_callback()

    def save_and_exit(self, event, do_close=True):
        """
        The Save and Exit Button

        write a callback to redraw viz for bbox_list
        """
        print('[interact_annot] Pressed Accept Button')

        def get_annottup_list():
            annottup_list = []
            indices_list = []
            #theta_list = []
            for poly in six.itervalues(self.polys):
                assert poly is not None
                index   = poly.num
                bbox    = tuple(map(int, basecoords_to_bbox(poly.basecoords)))
                theta   = poly.theta
                species = poly.species_tag.get_text()
                annottup = (bbox, theta, species)
                indices_list.append(index)
                annottup_list.append(annottup)
            return indices_list, annottup_list

        def send_back_annotations():
            print('[interact_annot] send_back_annotations')
            indices_list, annottup_list = get_annottup_list()
            # Delete if index is in original_indices but no in indices_list
            deleted_indices   = list(set(self.original_indices) -
                                     set(indices_list))
            changed_indices   = []
            unchanged_indices = []  # sanity check
            changed_annottups = []
            new_annottups     = []
            original_annottup_list = list(zip(self.original_bbox_list,
                                              self.original_theta_list,
                                              self.original_species_list))
            for index, annottup in zip(indices_list, annottup_list):
                # If the index is not in the originals then it is new
                if index not in self.original_indices:
                    new_annottups.append(annottup)
                else:
                    if annottup not in original_annottup_list:
                        changed_annottups.append(annottup)
                        changed_indices.append(index)
                    else:
                        unchanged_indices.append(index)
            self.commit_callback(unchanged_indices, deleted_indices,
                                 changed_indices, changed_annottups,
                                 new_annottups)

        if self.commit_callback is not None:
            send_back_annotations()
        #else:
            #just print the updated points
            #self.load_points()
            #print(self.poly_list)
        # Make mask from selection
        if self.do_mask is True:
            plt.clf()
            self.ax = ax = plt.subplot(111)
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

        print('[interact_annot] Accept Over')
        if do_close:
            df2.close_figure(self.fig)

    # --- Connected Slots and Callbacks

    def on_close(self, event=None):
        # TODO rectifify with abstract interaction
        # Hack: fake unregistration. does not inherit propertly
        AbstractInteraction.on_close.im_func(self, event)
        #abstract_interaction.unregister_interaction(self)

    def show(self):
        self.draw()
        self.bring_to_front()

    def draw(self):
        self.fig.canvas.draw()

    def bring_to_front(self):
        df2.bring_to_front(self.fig)

    def next_polynum(self):
        num = self._autoinc_polynum
        self._autoinc_polynum += 1
        return num

    def draw_callback(self, event):
        self.background = self.fig.canvas.copy_from_bbox(self.fig.ax.bbox)
        self.draw_artists()

    def draw_artists(self):
        for poly in six.itervalues(self.polys):
            self.fig.ax.draw_artist(poly)
            self.fig.ax.draw_artist(poly.lines)
            self.fig.ax.draw_artist(poly.handle)
            if self.show_species_tags:
                self.fig.ax.draw_artist(poly.species_tag)

    def on_click(self, event):
        """
        Called whenever a mouse button is pressed

        python -m ibeis.viz.interact.interact_annotations2 --test-ishow_image2 --show

        """
        if ut.VERBOSE:
            print('[on_click] key = %r' % (event.key))
        if self._ind is not None:
            self._ind = None
            return
        ignore = not self.showverts or event.inaxes is None
        if ignore:
            return

        if event.button == 1:  # leftclick
            if event.key == 'shift':
                self._currently_selected_poly
                self.currently_rotating_poly = self._currently_selected_poly
            else:
                for poly in six.itervalues(self.polys):
                    near_line = is_within_distance_from_line(
                        self.max_ds, (event.xdata, event.ydata),
                        calc_handle_coords(poly))
                    if near_line:
                        self.currently_rotating_poly = poly
                        break
        # CONTEXT MENU
        #if True:
        if event.button == 3:
            def make_options():
                def print_poly_info():
                    print('self._currently_selected_poly = %r' %
                          (self._currently_selected_poly,))
                    print('tag_text = %r' %
                          (self._currently_selected_poly.species_tag.get_text(),))
                    print('self._currently_selected_poly.metadata = %r' %
                          (self._currently_selected_poly.metadata,))

                metadata = self._currently_selected_poly.metadata
                options = []
                options += [
                    #('Foo: ',  functools.partial(print, 'bar')),
                    ('PolyInfo: ',  print_poly_info),
                ]
                if isinstance(metadata, ut.LazyDict):
                    options += metadata.nocache_eval('annot_context_options')

                return options
            options = make_options()
            self.show_popup_menu(options, event)

        if self._currently_selected_poly is None:
            print('[interact_annot] WARNING: Polygon unknown.'
                  ' Using last placed poly.')
            if len(self.polys) == 0:
                print('[interact_annot] No polygons on screen')
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
        if event.button == 1:  # left
            self.leftbutton_is_down = True
        self.canUncolor = False
        self._update_line()
        if self.background is not None:
            self.fig.canvas.restore_region(self.background)
        else:
            print('[interact_annot] error: self.background is none.'
                  ' Trying refresh.')
            self.fig.canvas.restore_region(self.background)
            self.background = self.fig.canvas.copy_from_bbox(self.fig.ax.bbox)
        for poly in six.itervalues(self.polys):
            self.fig.ax.draw_artist(poly)
            self.fig.ax.draw_artist(poly.lines)
            self.fig.ax.draw_artist(poly.handle)

        self.fig.canvas.blit(self.fig.ax.bbox)

    def on_figure_leave(self, event):
        if ut.VERBOSE:
            print('figure leave')
        self.on_click_release(event)

    def on_click_release(self, event):
        """
        Called whenever a mouse button is released
        """
        if ut.VERBOSE:
            print('click release')

        if self._polyHeld is True:
            self._polyHeld = False

        self.currently_rotating_poly = None

        ignore = not self.showverts or self._currently_selected_poly is None
        if ignore:
            return

        _flag = (
            self._ind is None or
            self._polyHeld is False or
            (self._ind is not None and
             self.leftbutton_is_down is True and
             self._currently_selected_poly is not None and
             self.canUncolor is True)
        )
        if _flag:
            self._currently_selected_poly.set_alpha(0)
            #self._currently_selected_poly.set_facecolor('white')

        self.update_UI()
        if event is None or event.button == 1:  # left
            self.leftbutton_is_down = False

        if self._ind is None:
            return
        if self._currently_selected_poly is None:
            print('[interact_annot] WARNING: Polygon unknown.'
                  ' Using default. (2)')
            if len(self.polys) == 0:
                print('[interact_annot] No polygons on screen')
                return
            else:
                _tup = self.get_most_recently_added_poly()
                poly_ind, self._currently_selected_poly = _tup
        currX, currY = self._currently_selected_poly.xy[self._ind]

        if self.indX and self.indY:
            if (math.fabs(self.indX - currX) < 3 and
                 math.fabs(self.indY - currY) < 3):
                return

        self._ind = None
        self._polyHeld = False

        self.fig.canvas.draw()

    def on_key_press(self, event):
        """
        Callback whenever a key is pressed
        """
        if ut.VERBOSE or True:
            print('[interact_annot] on_key_press')
            print('[interact_annot] Got key: %r' % event.key)
        if not event.inaxes:
            return

        def handle_control_command(keychar):
            print('[interact_annot] got hotkey=%r' % (keychar,))

        def handle_label_typing(keychar):
            if self._currently_selected_poly:
                #text = self._currently_selected_poly.species_tag.get_text()
                self._currently_selected_poly.tctext += keychar
                # TODO: Something better like greying out the tab suggestion
                # instead of just deleting it
                self._currently_selected_poly.species_tag.set_text(
                    self._currently_selected_poly.tctext)
                regen_tc()

        def regen_tc():
            # Setup tab completion
            # Yes this will redo the tab completion list every time a user
            # types. This should be improved if we move to having more species
            self._currently_selected_poly.tab_list = [
                spec
                for spec in self.valid_species
                if spec.startswith(self._currently_selected_poly.tctext)
            ]
            self._currently_selected_poly.tcindex = 0

        # perfect use case for anaphoric if, or assignment in if statements (if
        # python had either)
        if event.key == ACCEPT_SAVE_HOTKEY:
            self.save_and_exit(event)
        elif event.key == ADD_RECTANGLE_HOTKEY:
            self.add_new_annot()
        elif event.key == ADD_RECTANGLE_FULL_HOTKEY:
            self.add_new_annot(full=True)
        elif event.key == DEL_RECTANGLE_HOTKEY:
            self.delete_current_annot()
        elif event.key == TOGGLE_LABEL_HOTKEY:
            self.toggle_species_label()
        elif event.key == 'ctrl+u':
            self.load_points()
        elif event.key == 'ctrl+p':
            print('[interact_annot] fignums=%r' % (plt.get_fignums(),))

        # enter clears the species tag, workaround since matplotlib doesn't
        # seem to trigger 'key_press_event's for backspace (which would be the
        # preferred interface)
        #match = re.match('^enter$', event.key)
        #if match:
        #    self._currently_selected_poly.species_tag.set_text('')
        # FIXED, but leaving for posterity

        match = re.match('^backspace$', event.key)
        if match:
            # We want backspace to operate on the tctext
            #text = self._currently_selected_poly.species_tag.get_text()
            self._currently_selected_poly.tctext = self._currently_selected_poly.tctext[:-1]
            self._currently_selected_poly.species_tag.set_text(
                self._currently_selected_poly.tctext)
            regen_tc()

        match = re.match('^tab$', event.key)
        if match:
            if len(self._currently_selected_poly.tab_list) > 0:
                tci = self._currently_selected_poly.tcindex
                tci = (
                    tci + 1
                    if tci != len(self._currently_selected_poly.tab_list) - 1
                    else 0)
                self._currently_selected_poly.tcindex = tci
                # All tab is going to do is go through the possibilities
                self._currently_selected_poly.species_tag.set_text(
                    self._currently_selected_poly.tab_list[
                        self._currently_selected_poly.tcindex])
        #TODO: Similar functionality for shift+tab to go backwards

        if not HACK_OFF_SPECIES_TYPING:
            match = re.match('^.$', event.key)
            if match:
                handle_label_typing(match.group(0))

        # NEXT ANND PREV COMMAND
        print('[interact_annot] Got key: %r' % event.key)
        def matches_hotkey(key, hotkeys):
            hotkeys = [hotkeys] if not isinstance(hotkeys, list) else hotkeys
            #flags = [re.match(hk, '^' + key + '$') for hk in hotkeys]
            flags = [re.match(hk,  key) is not None for hk in hotkeys]
            print(hotkeys)
            print(flags)
            return any(flags)

        if matches_hotkey(event.key, PREV_IMAGE_HOTKEYS):
            self.prev_image(event)
        if matches_hotkey(event.key, NEXT_IMAGE_HOTKEYS):
            self.next_image(event)
        self.fig.canvas.draw()

    def on_motion(self, event):
        """
        CALLBACK FOR MOTION EVENTS
        Called on mouse movement
        """
        if ut.VERBOSE or True:
            print('[interact_annot] on_motion')
            print('[interact_annot] Got key: %r' % event.key)
        #ignore = (not self.showverts or event.inaxes is None)
        ignore = (not self.showverts)
        # uses boolean punning for terseness
        lastX = self.mouseX or None
        lastY = self.mouseY or None
        if not (event.xdata is None or event.ydata is None):
            self.mouseX, self.mouseY = event.xdata, event.ydata
            #print('mouse coords %r, %r; previous %r, %r' % (self.mouseX,
            #self.mouseY, lastX, lastY))
        else:
            # Allow for getting coordinates outside the axes
            ax = self.ax
            self.mouseX, self.mouseY = ax.transAxes.inverted().transform(
                [event.x, event.y])
            self.mouseX, self.mouseY = ax.transData.inverted().transform(
                [event.x, event.y])
        deltaX = lastX is not None and self.mouseX - lastX
        deltaY = lastY is not None and self.mouseY - lastY

        if ignore:
            return

        if self.leftbutton_is_down is True:
            self.canUncolor = True

        QUICK_RESIZE = (self._polyHeld is True and (
            event.button == 2 or
            event.button == 1 and event.key == 'shift'
        ))

        if self._polyHeld is True and self._ind is not None:
            # Resize by dragging corner
            self.resize_rectangle(self._currently_selected_poly, self.mouseX,
                                  self.mouseY, self._ind)
            self.update_UI()
            return
        elif QUICK_RESIZE:
            print('Quick Resize')
            # Quick resize
            anchor_idx = self._currently_selected_poly.last_idx
            idx = (anchor_idx + 2) % 4
            self.resize_rectangle(self._currently_selected_poly, self.mouseX,
                                  self.mouseY, idx)  # 0)
            self.update_UI()
            return

        if self.currently_rotating_poly:
            poly = self.currently_rotating_poly
            cx, cy = polygon_center(poly)
            theta = math.atan2(cy - self.mouseY, cx - self.mouseX) - TAU / 4
            dtheta = theta - poly.theta
            self.rotate_rectangle(poly, dtheta)
            self.update_UI()
            return

        if self._ind is None and event.button == 1:
            # move all vertices
            if (self._polyHeld is True and
                 not (deltaX is None or deltaY is None)):
                self.move_rectangle(self._currently_selected_poly, deltaX,
                                    deltaY)
            self.update_UI()
            self._ind = None
            return

    def mouse_enter(self, event):
        self._currently_selected_poly = event.artist
        self._currently_selected_poly.set_alpha(.2)

    def mouse_leave(self, event):
        self._currently_selected_poly.set_alpha(0)
        self._currently_selected_poly = None

    def onpick(self, event):
        """ Makes selected polygon translucent """
        #print('onpick')
        self._currently_selected_poly = event.artist
        #x, y = event.mouseevent.xdata, event.mouseevent.xdata
        self._polyHeld = True

    def poly_changed(self, poly):
        """ this method is called whenever the polygon object is called """
        # only copy the artist props to the line (except visibility)
        vis = poly.lines.get_visible()
        vis = poly.handle.get_visible()
        poly.lines.set_visible(vis)
        poly.handle.set_visible(vis)

    #def on_resize(self, event):
    #    self.fig.canvas.draw()
    #    plt.draw()


if __name__ == '__main__':
    """
    CommandLine:
        python -m plottool.interact_annotations --exec-test_interact_annots --show
    CommandLine:
        python -m plottool.interact_annotations
        python -m plottool.interact_annotations --allexamples
        python -m plottool.interact_annotations --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
