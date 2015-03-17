# Dependencies: flask, tornado
from __future__ import absolute_import, division, print_function
# HTTP / HTML
import tornado.wsgi
import tornado.httpserver
import flask
from flask import request, redirect, url_for, make_response
import optparse
import logging
import socket
import simplejson as json
# IBEIS
import ibeis
from ibeis.control.SQLDatabaseControl import (SQLDatabaseController,  # NOQA
                                              SQLAtomicContext)
from ibeis.control import _sql_helpers
from ibeis.constants import KEY_DEFAULTS, SPECIES_KEY, Species
import utool
import utool as ut
# Web Internal
from ibeis.web import appfuncs as ap
from ibeis.web import DBWEB_SCHEMA
# Others
import random
from os.path import join
from ibeis.web.DBWEB_SCHEMA import VIEWPOINT_TABLE
import ibeis.constants as const


BROWSER = ut.get_argflag('--browser')
DEFAULT_PORT = 5000
app = flask.Flask(__name__)


################################################################################


@app.route('/')
@app.route('/<filename>')
def root(filename=None):
    return ap.template(None, filename)


@app.route('/view')
def view():
    eid_list = app.ibs.get_valid_eids()
    gid_list = app.ibs.get_valid_gids()
    aid_list = app.ibs.get_valid_aids()
    return ap.template('view',
                       eid_list=eid_list,
                       num_eids=len(eid_list),
                       gid_list=gid_list,
                       num_gids=len(gid_list),
                       aid_list=aid_list,
                       num_aids=len(aid_list))


@app.route('/view/encounters')
def view_encoutners():
    eid_list = app.ibs.get_valid_eids()
    encounter_list = zip(
        eid_list,
        app.ibs.get_encounter_enctext(eid_list),
        app.ibs.get_encounter_num_gids(eid_list),
        app.ibs.get_encounter_start_time_posix(eid_list),
    )
    # ut.unixtime_to_datetime(
    return ap.template('view', 'encounters',
                       encounter_list=encounter_list,
                       num_encounters=len(encounter_list))


@app.route('/view/images')
def view_images():
    eid = None
    if 'eid' in request.args.keys():
        eid = int(request.args['eid'])
    gid_list = app.ibs.get_valid_gids(eid=eid)
    image_list = zip(
        gid_list,
        app.ibs.get_image_gnames(gid_list),
        app.ibs.get_image_unixtime(gid_list),
        ut.tupstr(app.ibs.get_image_gps(gid_list)),
        app.ibs.get_image_party_tag(gid_list),
        app.ibs.get_image_contributor_tag(gid_list),
        app.ibs.get_image_notes(gid_list),
    )
    return ap.template('view', 'images',
                       image_list=image_list,
                       num_images=len(image_list))


@app.route('/view/annotations')
def view_annotations():
    aid_list = app.ibs.get_valid_aids()
    return ap.template('view', 'annotations',
                       aid_list=aid_list,
                       num_aids=len(aid_list))


@app.route('/turk')
@app.route('/turk/<filename>')
def turk(filename=None):
    if 'refer' in request.args.keys():
        refer = request.args['refer']
    else:
        refer = None

    if filename == 'detection':
        if 'gid' in request.args.keys():
            gid = int(request.args['gid'])
        else:
            with SQLAtomicContext(app.db):
                gid = ap.get_next_detection_turk_candidate(app)
        finished = gid is None
        review = 'review' in request.args.keys()
        if not finished:
            gpath = app.ibs.get_image_paths(gid)
            image = ap.open_oriented_image(gpath)
            image_src = ap.embed_image_html(image, filter_width=False)
            # Get annotations
            width, height = app.ibs.get_image_sizes(gid)
            scale_factor = 700.0 / float(width)
            aid_list = app.ibs.get_image_aids(gid)
            annot_bbox_list = app.ibs.get_annot_bboxes(aid_list)
            annot_thetas_list = app.ibs.get_annot_thetas(aid_list)
            species_list = app.ibs.get_annot_species_texts(aid_list)
            # Get annotation bounding boxes
            annotation_list = []
            for annot_bbox, annot_theta, species in zip(annot_bbox_list, annot_thetas_list, species_list):
                temp = {}
                temp['left']   = int(scale_factor * annot_bbox[0])
                temp['top']    = int(scale_factor * annot_bbox[1])
                temp['width']  = int(scale_factor * (annot_bbox[2]))
                temp['height'] = int(scale_factor * (annot_bbox[3]))
                temp['label']  = species
                temp['angle']  = float(annot_theta)
                annotation_list.append(temp)
            if len(species_list) > 0:
                species = max(set(species_list), key=species_list.count)  # Get most common species
            elif app.default_species is not None:
                species = app.default_species
            else:
                species = KEY_DEFAULTS[SPECIES_KEY]
        else:
            gpath = None
            image_src = None
            species = None
            annotation_list = []
        display_instructions = request.cookies.get('detection_instructions_seen', 0) == 0
        display_species_examples = False  # request.cookies.get('detection_example_species_seen', 0) == 0
        if refer is not None and 'refer_aid' in request.args.keys():
            refer_aid = request.args['refer_aid']
        else:
            refer_aid = None
        return ap.template('turk', filename,
                           gid=gid,
                           species=species,
                           image_path=gpath,
                           image_src=image_src,
                           finished=finished,
                           annotation_list=annotation_list,
                           refer=refer,
                           refer_aid=refer_aid,
                           display_instructions=display_instructions,
                           display_species_examples=display_species_examples,
                           review=review)
    elif filename == 'viewpoint':
        if 'aid' in request.args.keys():
            aid = int(request.args['aid'])
        else:
            with SQLAtomicContext(app.db):
                aid = ap.get_next_viewpoint_turk_candidate(app)
        value = request.args.get('value', None)
        review = 'review' in request.args.keys()
        finished = aid is None
        if not finished:
            gid       = app.ibs.get_annot_gids(aid)
            gpath     = app.ibs.get_annot_chip_fpaths(aid)
            image     = ap.open_oriented_image(gpath)
            image_src = ap.embed_image_html(image)
        else:
            print("\nADMIN: http://%s:%s/turk/viewpoint-commit\n" % (app.server_ip_address, app.port))
            gid       = None
            gpath     = None
            image_src = None
        display_instructions = request.cookies.get('viewpoint_instructions_seen', 0) == 0
        return ap.template('turk', filename,
                           aid=aid,
                           gid=gid,
                           value=value,
                           image_path=gpath,
                           image_src=image_src,
                           finished=finished,
                           refer=refer,
                           display_instructions=display_instructions,
                           review=review)
    elif filename == 'viewpoint-commit':
        # Things that need to be committed
        where_clause = "viewpoint_value_avg>=?"
        viewpoint_rowid_list = ap.get_viewpoint_rowids_where(app, where_clause=where_clause, params=[0.0])
        aid_list = ap.get_viewpoint_aid(app, viewpoint_rowid_list)
        viewpoint_list = ap.get_viewpoint_values_from_aids(app, aid_list, 'viewpoint_value_avg')
        def convert_old_viewpoint_to_yaw(view_angle):
            """ we initially had viewpoint coordinates inverted

            Example:
                >>> import math
                >>> TAU = 2 * math.pi
                >>> old_viewpoint_labels = [
                >>>     ('left'       , 0.000 * TAU,),
                >>>     ('frontleft'  , 0.125 * TAU,),
                >>>     ('front'      , 0.250 * TAU,),
                >>>     ('frontright' , 0.375 * TAU,),
                >>>     ('right'       , 0.500 * TAU,),
                >>>     ('backright'  , 0.625 * TAU,),
                >>>     ('back'       , 0.750 * TAU,),
                >>>     ('backleft'   , 0.875 * TAU,),
                >>> ]
                >>> fmtstr = 'old %15r %.2f -> new %15r %.2f'
                >>> for lbl, angle in old_viewpoint_labels:
                >>>     print(fmtstr % (lbl, angle, lbl, convert_old_viewpoint_to_yaw(angle)))
            """
            if view_angle is None:
                return None
            yaw = (-view_angle + (const.TAU / 2)) % const.TAU
            return yaw
        viewpoint_radians_list = [None if angle is None else ut.deg_to_rad(angle) for angle in viewpoint_list]
        yaw_list = list(map(convert_old_viewpoint_to_yaw, viewpoint_radians_list))
        app.ibs.set_annot_yaws(aid_list, yaw_list, input_is_degrees=False)
        count = len(aid_list)
        # Flagged aids
        where_clause = "viewpoint_value_2!=? AND viewpoint_value_avg=?"
        viewpoint_rowid_list = ap.get_viewpoint_rowids_where(app, where_clause=where_clause, params=[-1.0, -1.0])
        flagged_aid_list = ap.get_viewpoint_aid(app, viewpoint_rowid_list)
        # Skipped aids
        where_clause = "viewpoint_value_2!=? AND viewpoint_value_avg=?"
        viewpoint_rowid_list = ap.get_viewpoint_rowids_where(app, where_clause=where_clause, params=[-1.0, -2.0])
        skipped_aid_list = ap.get_viewpoint_aid(app, viewpoint_rowid_list)
        # Return output
        return "Commiting %d viewpoints to the database...<br/>Flagged: %r<br/>Skipped: %r" % (count, flagged_aid_list, skipped_aid_list)
    else:
        return ap.template('turk', filename)


@app.route('/submit/viewpoint', methods=['POST'])
def submit_viewpoint():
    aid = int(request.form['viewpoint-aid'])
    value = int(request.form['viewpoint-value'])
    turk_id = request.cookies.get('turk_id', -1)
    if random.randint(0, 40) == 0:
        print("!!!!!DETECTION QUALITY CONTROL!!!!!")
        url = 'http://%s:%s/turk/viewpoint?aid=%s&value=%s&turk_id=%s&review=true' % (app.server_ip_address, app.port, aid, value, turk_id)
        import webbrowser
        webbrowser.open(url)
    if request.form['viewpoint-submit'].lower() == 'skip':
        value = -2
    # Get current values
    value_1 = ap.get_viewpoint_values_from_aids(app, [aid], 'viewpoint_value_1')[0]
    value_2 = ap.get_viewpoint_values_from_aids(app, [aid], 'viewpoint_value_2')[0]
    if value_1 is None:
        ap.set_viewpoint_values_from_aids(app, [aid], [value], 'viewpoint_value_1')
        value_1 = value
    elif value_2 is None:
        ap.set_viewpoint_values_from_aids(app, [aid], [value], 'viewpoint_value_2')
        value_2 = value
        if value_1 >= 0 and value_2 >= 0:
            # perform check against two viewpoint annotations
            if abs(value_1 - value_2) <= 45:
                value = (value_1 + value_2) / 2
                ap.set_viewpoint_values_from_aids(app, [aid], [value], 'viewpoint_value_avg')
            else:
                # We don't need to do anything here, viewpoints are promoted out of the error state (default) if consistent
                print('[web] FLAGGED - VIEWPOINTS INCONSISTENT')
        else:
            print('[web] SKIPPED - VIEWPOINTS UNSURE')
            ap.set_viewpoint_values_from_aids(app, [aid], [-2], 'viewpoint_value_avg')
    else:
        print('[web] SKIPPED - TOO MANY VIEWPOINTS')
    print("[web] turk_id: %s, aid: %d, value: %d | %s %s" % (turk_id, aid, value, value_1, value_2))
    # Return HTML
    return redirect(url_for('turk', filename='viewpoint'))


@app.route('/submit/detection', methods=['POST'])
def submit_detection():
    gid = int(request.form['detection-gid'])
    turk_id = request.cookies.get('turk_id', -1)
    if random.randint(0, 10) == 0:
        print("!!!!!DETECTION QUALITY CONTROL!!!!!")
        url = 'http://%s:%s/turk/detection?gid=%s&turk_id=%s&review=true' % (app.server_ip_address, app.port, gid, turk_id)
        import webbrowser
        webbrowser.open(url)
    aid_list = app.ibs.get_image_aids(gid)
    count = 1
    if request.form['detection-submit'].lower() == 'skip':
        count = -1
    ap.set_review_count_from_gids(app, [gid], [count])
    if count == 1:
        width, height = app.ibs.get_image_sizes(gid)
        scale_factor = float(width) / 700.0
        # Get aids
        app.ibs.delete_annots(aid_list)
        annotation_list = json.loads(request.form['detection-annotations'])
        bbox_list = [
            (
                int(scale_factor * annot['left']),
                int(scale_factor * annot['top']),
                int(scale_factor * annot['width']),
                int(scale_factor * annot['height']),
            )
            for annot in annotation_list
        ]
        theta_list = [
            float(annot['angle'])
            for annot in annotation_list
        ]
        species_list = [
            annot['label']
            for annot in annotation_list
        ]
        print("[web] turk_id: %s, gid: %d, bbox_list: %r, species_list: %r" % (turk_id, gid, annotation_list, species_list))
        aid_list_new = app.ibs.add_annots([gid] * len(annotation_list), bbox_list, theta_list=theta_list, species_list=species_list)
        app.ibs.set_image_reviewed([gid], [1])
        ap.replace_aids(app, aid_list, aid_list_new)
    else:
        app.ibs.set_image_reviewed([gid], [0])
        viewpoint_rowids = ap.get_viewpoint_rowids_from_aid(aid_list, app)
        app.db.delete_rowids(VIEWPOINT_TABLE, viewpoint_rowids)
    if 'refer' in request.args.keys() and request.args['refer'] == 'viewpoint':
        return redirect(url_for('turk', filename='viewpoint'))
    else:
        return redirect(url_for('turk', filename='detection'))


@app.route('/ajax/cookie')
def set_cookie():
    response = make_response('true')
    try:
        response.set_cookie(request.args['name'], request.args['value'])
        print("Set Cookie: %r -> %r" % (request.args['name'], request.args['value'], ))
        return response
    except:
        print("COOKIE FAILED: %r" % (request.args, ))
        return make_response('false')


@app.route('/ajax/image/src/<gid>')
def image_src(gid=None):
    gpath = app.ibs.get_image_paths(gid)
    image = ap.open_oriented_image(gpath)
    image_src = ap.embed_image_html(image, filter_width=False)
    return image_src


@app.route('/api')
@app.route('/api/<function>.json', methods=['GET', 'POST'])
def api(function=None):
    template = {
        'status': {
            'success': False,
            'code': '',
        },
    }
    print('Function:', function)
    print('POST:', dict(request.form))
    print('GET:',  dict(request.args))
    if function is None:
        template['status']['success'] = True
        template['status']['code'] = 'USAGE: /api/[ibeis_function_name].json'
    else:
        function = function.lower()
        if ap.check_valid_function_name(function):
            function = 'app.ibs.%s' % function
            exists = True
            try:
                func = eval(function)
                ret = func()
            except AttributeError:
                exists = False
            if exists:
                template['status']['success'] = True
                template['function'] = function
                template['return'] = ret
            else:
                template['status']['success'] = False
                template['status']['code'] = 'ERROR: Specified IBEIS function not visible or implemented'
        else:
            template['status']['success'] = False
            template['status']['code'] = 'ERROR: Specified IBEIS function not valid Python function'
    return json.dumps(template)


################################################################################


def init_database(app, reset_db):
    database_dir = utool.get_app_resource_dir('ibeis', 'web')
    database_filename = 'app.sqlite3'
    database_filepath = join(database_dir, database_filename)
    utool.ensuredir(database_dir)
    if reset_db:
        utool.remove_file(database_filepath)
    app.dbweb_version_expected = '1.0.0'
    app.db = SQLDatabaseController(database_dir, database_filename)
    _sql_helpers.ensure_correct_version(
        app.ibs,
        app.db,
        app.dbweb_version_expected,
        DBWEB_SCHEMA
    )


def start_tornado(app, port=5000, browser=BROWSER, blocking=False, reset_db=True, database_init=None):
    def _start_tornado():
        http_server = tornado.httpserver.HTTPServer(
            tornado.wsgi.WSGIContainer(app))
        http_server.listen(port)
        tornado.ioloop.IOLoop.instance().start()
    # Open the web internal database
    init_database(app, reset_db=reset_db)
    # Initialize the web server
    logging.getLogger().setLevel(logging.INFO)
    try:
        app.server_ip_address = socket.gethostbyname(socket.gethostname())
        app.port = port
    except:
        app.server_ip_address = '127.0.0.1'
        app.port = port
    url = 'http://%s:%s' % (app.server_ip_address, app.port)
    print('[web] Tornado server starting at %s' % (url,))
    if browser:
        import webbrowser
        webbrowser.open(url)
    if database_init is not None:
        database_init(app)
    # Blocking
    _start_tornado()
    # if blocking:
    #     _start_tornado()
    # else:
    #     import threading
    #     threading.Thread(target=_start_tornado).start()


def start_from_terminal():
    '''
    Parse command line options and start the server.
    '''
    parser = optparse.OptionParser()
    parser.add_option(
        '-p', '--port',
        help='which port to serve content on',
        type='int', default=DEFAULT_PORT)
    parser.add_option(
        '--db',
        help='specify an IBEIS database',
        type='str', default='testdb0')
    parser.add_option(
        '--round',
        help='specify the round of turking',
        type='int', default='1')

    opts, args = parser.parse_args()
    app.round = opts.round
    print(app.round)
    app.ibs = ibeis.opendb(db=opts.db)
    start_tornado(app, opts.port, database_init=ap.database_init)


def start_from_ibeis(ibs, port=DEFAULT_PORT):
    '''
    Parse command line options and start the server.
    '''
    from ibeis import params
    dbname = ibs.get_dbname()
    if dbname == "CHTA_Master":
        app.default_species = Species.CHEETAH
    elif dbname == "ELPH_Master":
        app.default_species = Species.ELEPHANT_SAV
    elif dbname == "GIR_Master":
        app.default_species = Species.GIRAFFE
    elif dbname == "GZ_Master":
        app.default_species = Species.ZEB_GREVY
    elif dbname == "LION_Master":
        app.default_species = Species.LION
    elif dbname == "PZ_Master":
        app.default_species = Species.ZEB_PLAIN
    elif dbname == "WD_Master":
        app.default_species = Species.WILDDOG
    else:
        app.default_species = None
    print("DEFAULT SPECIES: %r" % (app.default_species))
    app.ibs = ibs
    app.round = params.args.round
    start_tornado(app, port, database_init=ap.database_init)


if __name__ == '__main__':
    start_from_terminal()
