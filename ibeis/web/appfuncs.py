from __future__ import absolute_import, division, print_function
from PIL import Image
import numpy as np
import cStringIO as StringIO
from functools import partial
import random
from ibeis.web.DBWEB_SCHEMA import VIEWPOINT_TABLE, REVIEW_TABLE

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
    '''
    This script handles the skimage exif problem.
    '''
    if orientation in ORIENTATIONS:
        for method in ORIENTATIONS[orientation]:
            im = im.transpose(method)
    return im


def embed_image_html(image, filter_width=True):
    '''Creates an image embedded in HTML base64 format.'''
    image_pil = Image.fromarray((255 * image).astype('uint8'))
    width, height = image_pil.size
    if filter_width:
        _height = 300
        _width = int((float(_height) / height) * width)
    else:
        _width = 500
        _height = int((float(_width) / width) * height)
    image_pil = image_pil.resize((_width, _height))
    string_buf = StringIO.StringIO()
    image_pil.save(string_buf, format='jpeg')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    return 'data:image/jpeg;base64,' + data


def check_valid_function_name(string):
    return all([ char.isalpha() or char == '_' or char.isalnum() for char in string])


################################################################################


def get_next_detection_turk_candidate(app):
    # Get available chips
    where_clause = "review_count=?"
    review_rowid_list = get_review_rowids_where(app, where_clause=where_clause, params=[0])
    count = len(review_rowid_list)
    if count == 0:
        return None, None
    else:
        status = 'Stage 1 - %s' % count
    print('[web] %s' % status)
    gid_list = get_review_gid(app, review_rowid_list)
    # Decide out of the candidates
    index = random.randint(0, len(gid_list) - 1)
    gid = gid_list[index]
    gpath = app.ibeis.get_image_paths(gid)
    return gid, gpath


def get_next_viewpoint_turk_candidate(app):
    # Get available chips
    where_clause = "viewpoint_value_1=?"
    viewpoint_rowid_list = get_viewpoint_rowids_where(app, where_clause=where_clause, params=[-1.0])
    count = len(viewpoint_rowid_list)
    if count == 0:
        where_clause = "viewpoint_value_2=?"
        viewpoint_rowid_list = get_viewpoint_rowids_where(app, where_clause=where_clause, params=[-1.0])
        count = len(viewpoint_rowid_list)
        if count == 0:
            return None, None
        else:
            status = 'Stage 2 - %s' % count
    else:
        status = 'Stage 1 - %s' % count
    print('[web] %s' % status)
    aid_list = get_viewpoint_aid(app, viewpoint_rowid_list)
    # Decide out of the candidates
    index = random.randint(0, len(aid_list) - 1)
    aid = aid_list[index]
    gpath = app.ibeis.get_annot_chip_paths(aid)
    return aid, gpath


################################################################################


def get_review_gid(app, review_rowid_list):
    gid_list = app.db.get(REVIEW_TABLE, ('image_rowid',), review_rowid_list)
    return gid_list


def get_review_rowids_where(app, where_clause, params):
    review_rowid_list = app.db.get_all_rowids_where(REVIEW_TABLE, where_clause=where_clause, params=params)
    return review_rowid_list


def get_review_rowids_from_gid(gid_list, app):
    review_rowid_list = app.db.get(REVIEW_TABLE, ('review_rowid',), gid_list, id_colname='image_rowid')
    return review_rowid_list


def get_review_counts_from_gids(app, gid_list):
    count_list = app.db.get(REVIEW_TABLE, ('review_count',), gid_list, id_colname='image_rowid')
    return  count_list


def get_viewpoint_aid(app, viewpoint_rowid_list):
    aid_list = app.db.get(VIEWPOINT_TABLE, ('annot_rowid',), viewpoint_rowid_list)
    return aid_list


def get_viewpoint_rowids_where(app, where_clause, params):
    viewpoint_rowid_list = app.db.get_all_rowids_where(VIEWPOINT_TABLE, where_clause=where_clause, params=params)
    return viewpoint_rowid_list


def get_viewpoint_rowids_from_aid(aid_list, app):
    viewpoint_rowid_list = app.db.get(VIEWPOINT_TABLE, ('viewpoint_rowid',), aid_list, id_colname='annot_rowid')
    return viewpoint_rowid_list


def get_viewpoint_values_from_aids(app, aid_list, value_type):
    value_list = app.db.get(VIEWPOINT_TABLE, (value_type,), aid_list, id_colname='annot_rowid')
    value_list = [None if value == -1.0 else value for value in value_list]
    return  value_list


################################################################################


def set_review_count_from_gids(app, gid_list, count_list):
    app.db.set(VIEWPOINT_TABLE, ('review_count',), count_list, gid_list, id_colname='image_rowid')


def set_viewpoint_values_from_aids(app, aid_list, value_list, value_type):
    app.db.set(VIEWPOINT_TABLE, (value_type,), value_list, aid_list, id_colname='annot_rowid')


################################################################################


def database_init(app):
    def rad_to_deg(radians):
        import numpy as np
        twopi = 2 * np.pi
        radians %= twopi
        return int((radians / twopi) * 360.0)

    # Detection Review
    gid_list = app.ibeis.get_valid_gids()
    colnames = ('image_rowid', )
    params_iter = zip(gid_list)
    get_rowid_from_superkey = partial(get_review_rowids_from_gid, app=app)
    app.db.add_cleanly(REVIEW_TABLE, colnames, params_iter, get_rowid_from_superkey)

    # Viewpoint Annotation
    # Get data out of ibeis
    aid_list = app.ibeis.get_valid_aids()
    # Grab viewpoints
    viewpoint_list = app.ibeis.get_annot_viewpoints(aid_list)
    viewpoint_list = [-1 if viewpoint is None else rad_to_deg(viewpoint) for viewpoint in viewpoint_list]
    # Add chips to temporary web viewpoint table
    redo = False
    if redo:
        colnames = ('annot_rowid', 'viewpoint_value_1', 'viewpoint_value_2',  'viewpoint_value_avg')
        params_iter = zip(aid_list, viewpoint_list, viewpoint_list, viewpoint_list)
        get_rowid_from_superkey = partial(get_viewpoint_rowids_from_aid, app=app)
        app.db.add_cleanly(VIEWPOINT_TABLE, colnames, params_iter, get_rowid_from_superkey)
    else:
        colnames = ('annot_rowid', 'viewpoint_value_1')
        params_iter = zip(aid_list, viewpoint_list)
        get_rowid_from_superkey = partial(get_viewpoint_rowids_from_aid, app=app)
        app.db.add_cleanly(VIEWPOINT_TABLE, colnames, params_iter, get_rowid_from_superkey)
