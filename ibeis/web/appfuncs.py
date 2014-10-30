from __future__ import absolute_import, division, print_function
from PIL import Image
import numpy as np
import cStringIO as StringIO
from ibeis.web.DBWEB_SCHEMA import VIEWPOINT_TABLE

ORIENTATIONS = {   # used in apply_orientation
    2: (Image.FLIP_LEFT_RIGHT,),
    3: (Image.ROTATE_180,),
    4: (Image.FLIP_TOP_BOTTOM,),
    5: (Image.FLIP_LEFT_RIGHT, Image.ROTATE_90),
    6: (Image.ROTATE_270,),
    7: (Image.FLIP_LEFT_RIGHT, Image.ROTATE_270),
    8: (Image.ROTATE_90,)
}


def open_oriented_image(im_path):
    im = Image.open(im_path)
    if hasattr(im, '_getexif'):
        exif = im._getexif()
        if exif is not None and 274 in exif:
            orientation = exif[274]
            im = apply_orientation(im, orientation)
    img = np.asarray(im).astype(np.float32) / 255.
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def apply_orientation(im, orientation):
    """
    This script handles the skimage exif problem.
    """
    if orientation in ORIENTATIONS:
        for method in ORIENTATIONS[orientation]:
            im = im.transpose(method)
    return im


def embed_image_html(image):
    """Creates an image embedded in HTML base64 format."""
    image_pil = Image.fromarray((255 * image).astype('uint8'))
    width, height = image_pil.size
    _width = 500
    _height = int((float(_width) / width) * height)
    image_pil = image_pil.resize((_width, _height))
    string_buf = StringIO.StringIO()
    image_pil.save(string_buf, format='jpeg')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    return 'data:image/jpeg;base64,' + data


def check_valid_function_name(string):
    return all([ char.isalpha() or char == "_" or char.isalnum() for char in string])


def database_init(app):
    aid_list = app.ibeis.get_valid_aids()
    cpath_list = app.ibeis.get_annot_cpaths(aid_list)
    viewpoint_list = app.ibeis.get_annot_viewpoints(aid_list)
    colnames = ('viewpoint_aid', 'viewpoint_cpath', 'viewpoint_value1')
    params_iter = zip(aid_list, cpath_list, viewpoint_list)
    app.db.add_cleanly(VIEWPOINT_TABLE, colnames, params_iter, (lambda x: x))
