# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from six.moves import cStringIO as StringIO
# import cStringIO as StringIO
import flask
import random
from ibeis import constants as const
from flask import request, current_app, url_for
from os.path import join, dirname, abspath  # NOQA
from datetime import datetime
from datetime import date
import base64
import jinja2
import utool as ut
import pynmea2
import simplejson as json
import numpy as np


DEFAULT_WEB_API_PORT = ut.get_argval('--port', type_=int, default=5000)
TARGET_WIDTH = 1200.0
TARGET_HEIGHT = 800.0
PAGE_SIZE = 500
VALID_TURK_MODES = [
    ('turk_annotation', 'Annotation'),
    ('turk_viewpoint', 'Viewpoint'),
    ('turk_quality', 'Quality'),
    ('turk_demographics', 'Demographics'),
]


VIEWPOINT_MAPPING = {
    None: None,
    -1: None,
    0: 'left',
    1: 'frontleft',
    2: 'front',
    3: 'frontright',
    4: 'right',
    5: 'backright',
    6: 'back',
    7: 'backleft',
}
VIEWPOINT_MAPPING_INVERT = {
    value: key for key, value in VIEWPOINT_MAPPING.items()
    if key is not None
}


class NavbarClass(object):
    def __init__(nav):
        nav.item_list = [
            ('root', 'Home'),
            ('view_imagesets', 'ImageSets'),
            ('view_images', 'Images'),
            ('view_annotations', 'Annotations'),
            ('view_names', 'Names'),
            # ('action', 'Action'),
            ('turk', 'Turk'),
            # ('api_root',  'API'),
            # ('group_review',  'Group Review'),
        ]

    def __iter__(nav):
        _link = request.path
        for link, nice in nav.item_list:
            active = _link == url_for(link)
            print(_link, link, url_for(link))
            yield active, link, nice


def resize_via_web_parameters(image):
    w_pix = request.args.get('resize_pix_w',      request.form.get('resize_pix_w',      None ))
    h_pix = request.args.get('resize_pix_h',      request.form.get('resize_pix_h',      None ))
    w_per = request.args.get('resize_per_w',      request.form.get('resize_per_w',      None ))
    h_per = request.args.get('resize_per_h',      request.form.get('resize_per_h',      None ))
    _pix  = request.args.get('resize_prefer_pix', request.form.get('resize_prefer_pix', False))
    _per  = request.args.get('resize_prefer_per', request.form.get('resize_prefer_per', False))
    args = (w_pix, h_pix, w_per, h_per, _pix, _per, )
    print('CHECKING RESIZING WITH %r pix, %r pix, %r %%, %r %% [%r, %r]' % args)
    # Check for nothing
    if not (w_pix or h_pix or w_per or h_per):
        return image
    # Check for both pixels and images
    if (w_pix or h_pix) and (w_per or h_per):
        if _pix:
            w_per, h_per = None
        elif _per:
            w_pix, h_pix = None
        else:
            raise ValueError('Cannot resize using pixels and percentages, pick one')
    # Resize using percentages, transform to pixels
    if w_per:
        w_pix = float(w_per) * image.shape[1]
    if h_per:
        h_pix = float(h_per) * image.shape[0]
    # Perform resize
    return _resize(image, t_width=w_pix, t_height=h_pix)


def embed_image_html(imgBGR, target_width=TARGET_WIDTH, target_height=TARGET_HEIGHT):
    """ Creates an image embedded in HTML base64 format. """
    import cv2
    from PIL import Image
    if target_width is not None:
        imgBGR = _resize(imgBGR, t_width=target_width)
    elif target_width is not None:
        imgBGR = _resize(imgBGR, t_height=target_height)
    imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(imgRGB)
    string_buf = StringIO()
    pil_img.save(string_buf, format='jpeg')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    return 'data:image/jpeg;base64,' + data


def check_valid_function_name(string):
    return all([ char.isalpha() or char == '_' or char.isalnum() for char in string])


def encode_refer_url(url):
    return base64.urlsafe_b64encode(str(url))


def decode_refer_url(encode):
    return base64.urlsafe_b64decode(str(encode))


def template(template_directory=None, template_filename=None, **kwargs):
    global_args = {
        'NAVBAR': NavbarClass(),
        'YEAR':   date.today().year,
        'URL':    flask.request.url,
        'REFER_SRC_STR':  flask.request.url.replace(flask.request.url_root, ''),
        '__wrapper__' : True,
        '__wrapper_header__' : True,
        '__wrapper_footer__' : True,
    }
    global_args['REFER_SRC_ENCODED'] = encode_refer_url(global_args['REFER_SRC_STR'])
    if 'refer' in flask.request.args.keys():
        refer = flask.request.args['refer']
        print('[web] REFER: %r' % (refer, ))
        global_args['REFER_DST_ENCODED'] = refer
        global_args['REFER_DST_STR'] = decode_refer_url(refer)
    if template_directory is None:
        template_directory = ''
        #template_directory = abspath(join(dirname(__file__), 'templates'))
        #template_directory = join(dirname(dirname(__file__)))
    if template_filename is None:
        template_filename = 'index'
    template_ = join(template_directory, template_filename + '.html')
    # Update global args with the template's args
    _global_args = dict(global_args)
    _global_args.update(kwargs)
    print('[appfuncs] template()')
    from ibeis.control import controller_inject
    app = controller_inject.get_flask_app()
    # flask hates windows apparently
    template_ = template_.replace('\\', '/')
    print('[appfuncs.template] * app.template_folder = %r' % (app.template_folder,))
    print('[appfuncs.template] * template_directory = %r' % (template_directory,))
    print('[appfuncs.template] * template_filename = %r' % (template_filename,))
    print('[appfuncs.template] * template_ = %r' % (template_,))
    try:
        ret = flask.render_template(template_, **_global_args)
        #ret = flask.render_template(full_template_fpath, **_global_args)
    except jinja2.exceptions.TemplateNotFound as ex:
        print('Error template not found')
        full_template_fpath = join(app.template_folder, template_)
        print('[appfuncs.template] * full_template_fpath = %r' % (full_template_fpath,))
        ut.checkpath(full_template_fpath, verbose=True)
        ut.printex(ex, 'Template error in appfuncs', tb=True)
        raise
    except Exception as ex:
        ut.printex(ex, 'Error in appfuncs', tb=True)
        raise
    return ret


def send_csv_file(string, filename):
    response = flask.make_response(str(string))
    response.headers['Content-Description'] = 'File Transfer'
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Content-Type'] = 'text/csv'
    response.headers['Content-Disposition'] = 'attachment; filename=%s' % filename
    response.headers['Content-Length'] = len(string)
    return response


def get_turk_annot_args(is_reviewed_func, speed_hack=False):
    """
    Helper to return aids in an imageset or a group review
    """
    ibs = current_app.ibs
    def _ensureid(_id):
        return None if _id == 'None' or _id == '' else int(_id)

    imgsetid = request.args.get('imgsetid', '')
    src_ag = request.args.get('src_ag', '')
    dst_ag = request.args.get('dst_ag', '')

    imgsetid = _ensureid(imgsetid)
    src_ag = _ensureid(src_ag)
    dst_ag = _ensureid(dst_ag)

    group_review_flag = src_ag is not None and dst_ag is not None
    if not group_review_flag:
        print('NOT GROUP_REVIEW')
        if speed_hack:
            with ut.Timer():
                aid_list = ibs.get_valid_aids()
        else:
            gid_list = ibs.get_valid_gids(imgsetid=imgsetid)
            aid_list = ut.flatten(ibs.get_image_aids(gid_list))
        reviewed_list = is_reviewed_func(ibs, aid_list)
    else:
        src_gar_rowid_list = ibs.get_annotgroup_gar_rowids(src_ag)
        dst_gar_rowid_list = ibs.get_annotgroup_gar_rowids(dst_ag)
        src_aid_list = ibs.get_gar_aid(src_gar_rowid_list)
        dst_aid_list = ibs.get_gar_aid(dst_gar_rowid_list)
        aid_list = src_aid_list
        reviewed_list = [ src_aid in dst_aid_list for src_aid in src_aid_list ]

    try:
        progress = '%0.2f' % (100.0 * reviewed_list.count(True) / len(aid_list), )
    except ZeroDivisionError:
        progress = '0.00'
    aid = request.args.get('aid', '')
    if len(aid) > 0:
        aid = int(aid)
    else:
        aid_list_ = ut.filterfalse_items(aid_list, reviewed_list)
        if len(aid_list_) == 0:
            aid = None
        else:
            if group_review_flag:
                aid = aid_list_[0]
            else:
                aid = random.choice(aid_list_)

    previous = request.args.get('previous', None)

    print('aid = %r' % (aid,))
    #print(ut.dict_str(ibs.get_annot_info(aid)))
    print(ut.obj_str(ibs.get_annot_info(aid, default=True, nl=True)))
    return aid_list, reviewed_list, imgsetid, src_ag, dst_ag, progress, aid, previous


def movegroup_aid(ibs, aid, src_ag, dst_ag):
    gar_rowid_list = ibs.get_annot_gar_rowids(aid)
    annotgroup_rowid_list = ibs.get_gar_annotgroup_rowid(gar_rowid_list)
    src_index = annotgroup_rowid_list.index(src_ag)
    src_gar_rowid = gar_rowid_list[src_index]
    vals = (aid, src_ag, src_gar_rowid, dst_ag)
    print('Moving aid: %s from src_ag: %s (%s) to dst_ag: %s' % vals)
    # ibs.delete_gar([src_gar_rowid])
    ibs.add_gar([dst_ag], [aid])


def default_species(ibs):
    # hack function
    dbname = ibs.get_dbname()
    if dbname == 'CHTA_Master' or dbname == 'EWT_Cheetahs':
        default_species = 'cheetah'
    elif dbname == 'EWT_Lynx':
        default_species = 'lynx'
    elif dbname == 'ELPH_Master':
        default_species = 'elephant_savanna'
    elif dbname == 'GIR_Master':
        default_species = 'giraffe_reticulated'
    elif dbname == 'GZ_Master':
        default_species = 'zebra_grevys'
    elif dbname == 'LION_Master':
        default_species = 'lion'
    elif dbname == 'PZ_Master':
        default_species = 'zebra_plains'
    elif dbname == 'WD_Master':
        default_species = 'wild_dog'
    elif dbname == 'NNP_MasterGIRM':
        default_species = 'giraffe_masai'
    elif 'NNP_' in dbname:
        default_species = 'zebra_plains'
    elif 'GZC' in dbname:
        default_species = 'zebra_plains'
    else:
        default_species = None
    print('[web] DEFAULT SPECIES: %r' % (default_species))
    return default_species


def imageset_image_processed(ibs, gid_list):
    images_reviewed = [ reviewed == 1 for reviewed in ibs.get_image_reviewed(gid_list) ]
    return images_reviewed


def imageset_annot_processed(ibs, aid_list):
    annots_reviewed = [ reviewed == 1 for reviewed in ibs.get_annot_reviewed(aid_list) ]
    return annots_reviewed


def imageset_annot_viewpoint_processed(ibs, aid_list):
    annots_reviewed = [ reviewed is not None for reviewed in ibs.get_annot_viewpoints(aid_list) ]
    return annots_reviewed


def imageset_annot_quality_processed(ibs, aid_list):
    annots_reviewed = [ reviewed is not None and reviewed is not -1 for reviewed in ibs.get_annot_qualities(aid_list) ]
    return annots_reviewed


def imageset_annot_demographics_processed(ibs, aid_list, nid_list):
    sex_list = ibs.get_annot_sex(aid_list)
    age_list = ibs.get_annot_age_months_est(aid_list)
    annots_reviewed = [
        (nid < 0) or (nid > 0 and sex >= 0 and -1 not in list(age) and list(age).count(None) < 2)
        for nid, sex, age in zip(nid_list, sex_list, age_list)
    ]
    return annots_reviewed


def convert_nmea_to_json(nmea_str, filename, GMT_OFFSET=0):
    json_list = []
    filename = filename.strip('.LOG').strip('N')
    year = 2000 + int(filename[0:2])
    month = int(filename[2:4])
    day = int(filename[4:6])
    print(year, month, day)
    for line in nmea_str.split('\n'):
        line = line.strip()
        if '@' in line or 'GPRMC' in line or len(line) == 0:
            continue
        record = pynmea2.parse(line)
        dt = record.timestamp
        dt = datetime(year, month, day, dt.hour, dt.minute, dt.second)
        # Gather values
        posix = int(dt.strftime("%s"))
        posix += (60 * 60 * GMT_OFFSET)
        lat   = float(record.latitude)
        lon   = float(record.longitude)
        json_list.append({
            'time': posix,
            'lat':  lat,
            'lon':  lon,
        })
    return json.dumps({ "track": json_list })


def convert_tuple_to_viewpoint(viewpoint_tuple):
    viewpoint_list = list(sorted(map(int, list(viewpoint_tuple))))

    if viewpoint_tuple is None or viewpoint_list.count(-1) >= 3:
        return None
    else:
        viewpoint_text = '__'.join(map(str, viewpoint_list))
        viewpoint_text = '_%s_' % (viewpoint_text, )
        viewpoint_text = viewpoint_text.replace('_-1_', '')
        viewpoint_text = viewpoint_text.replace('_0_', 'up')
        viewpoint_text = viewpoint_text.replace('_1_', 'down')
        viewpoint_text = viewpoint_text.replace('_2_', 'front')
        viewpoint_text = viewpoint_text.replace('_3_', 'back')
        viewpoint_text = viewpoint_text.replace('_4_', 'left')
        viewpoint_text = viewpoint_text.replace('_5_', 'right')
        assert viewpoint_text in const.YAWALIAS
        return viewpoint_text


def convert_viewpoint_to_tuple(viewpoint_text):
    if viewpoint_text is None or viewpoint_text not in const.YAWALIAS:
        return (-1, -1, -1)
    else:
        viewpoint_text = viewpoint_text.replace('up',    '_0_')
        viewpoint_text = viewpoint_text.replace('down',  '_1_')
        viewpoint_text = viewpoint_text.replace('front', '_2_')
        viewpoint_text = viewpoint_text.replace('back',  '_3_')
        viewpoint_text = viewpoint_text.replace('left',  '_4_')
        viewpoint_text = viewpoint_text.replace('right', '_5_')
        viewpoint_list = viewpoint_text.split('_')
        viewpoint_list = [ _ for _ in viewpoint_list if len(_) > 0 ]
        assert 1 <= len(viewpoint_list) and len(viewpoint_list) <= 3
        viewpoint_list = map(int, viewpoint_list)
        viewpoint_list = list(sorted(viewpoint_list))
        while len(viewpoint_list) < 3:
            viewpoint_list.append(-1)
        return tuple(viewpoint_list)


def _resize(image, t_width=None, t_height=None):
    """
    TODO:
        # use vtool instead
    """
    if False:
        import vtool as vt
        maxdims = (int(round(t_width)), int(round(t_height)))
        interpolation = 'linear'
        return vt.resize_to_maxdims(image, maxdims, interpolation)
    else:
        import cv2
        print('RESIZING WITH t_width = %r and t_height = %r' % (t_width, t_height, ))
        height, width = image.shape[:2]
        if t_width is None and t_height is None:
            return image
        elif t_width is not None and t_height is not None:
            pass
        elif t_width is None:
            t_width = (width / height) * float(t_height)
        elif t_height is None:
            t_height = (height / width) * float(t_width)
        t_width, t_height = float(t_width), float(t_height)
        t_width, t_height = int(np.around(t_width)), int(np.around(t_height))
        assert t_width > 0 and t_height > 0, 'target size too small'
        assert t_width <= width * 100 and t_height <= height * 100, (
            'target size too large (capped at 10,000%)')
        # interpolation = cv2.INTER_LANCZOS4
        interpolation = cv2.INTER_LINEAR
        return cv2.resize(image, (t_width, t_height), interpolation=interpolation)
