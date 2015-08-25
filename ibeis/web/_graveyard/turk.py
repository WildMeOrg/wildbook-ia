

BROWSER = ut.get_argflag('--browser')
DEFAULT_PORT = 5000
app = flask.Flask(__name__)
TARGET_WIDTH = 700


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
            scale_factor = float(TARGET_WIDTH) / float(width)
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
        scale_factor = float(width) / float(TARGET_WIDTH)
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


from functools import partial
import random
from ibeis.web.DBWEB_SCHEMA import VIEWPOINT_TABLE, REVIEW_TABLE
import ibeis.constants as const


################################################################################


def get_next_detection_turk_candidate(app):
    # Get available chips
    where_clause = "review_count=?"
    review_rowid_list = get_review_rowids_where(app, where_clause=where_clause, params=[0])
    count = len(review_rowid_list)
    if count == 0:
        return None
    else:
        status = 'Stage 1 - %s' % count
    print('[web] %s' % status)
    gid_list = get_review_gid(app, review_rowid_list)
    # Decide out of the candidates
    index = random.randint(0, len(gid_list) - 1)
    gid = gid_list[index]
    print('Detection candidate: %r' % (gid, ))
    return gid


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
            return None
        else:
            status = 'Stage 2 - %s' % count
    else:
        status = 'Stage 1 - %s' % count
    print('[web] %s' % status)
    aid_list = get_viewpoint_aid(app, viewpoint_rowid_list)
    # Decide out of the candidates
    index = random.randint(0, len(aid_list) - 1)
    aid = aid_list[index]
    print('Viewpoint candidate: %r' % (aid, ))
    return aid


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
    app.db.set(REVIEW_TABLE, ('review_count',), count_list, gid_list, id_colname='image_rowid')


def set_viewpoint_values_from_aids(app, aid_list, value_list, value_type):
    viewpoint_rowids = get_viewpoint_rowids_from_aid(aid_list, app)
    if None not in viewpoint_rowids:
        app.db.set(VIEWPOINT_TABLE, (value_type,), value_list, aid_list, id_colname='annot_rowid')
    else:
        # Alright, this is a state issue.  An annotation, X, was querried for viewpoint and is "checkedout" by person A.
        # Between the time person A checks out X and commits the annotation, person B also checks out X (random chance).
        # Person B makes a judgment call and thinks that the bouding box for X need to be updated.  Person B updates the
        # bounding box for X, which deletes X and creates X' to replace it.  In the meantime, person A still has the
        # row_id for X (which has not been deleted).  So, when person A commits the viewpoint for X, we simply do nothing.
        # X' viewpoint annotation will get put back into the random queue for processing and we lose a little bit of
        # efficiency.
        #
        # Essentially, any viewpoint work that was done by A for X is simply ignored because X no longer exists in that form.
        print('[set_viewpoint_values_from_aids] WARNING - IGNORING VIEWPOINT BECAUSE AID_LIST: %s NOT LONGER EXISTS' % (aid_list, ))


################################################################################


def replace_aids(app, aid_list, aid_list_new):
    print('Replacing %r for %r' % (aid_list, aid_list_new, ))
    # Delete the old aid_list from the cache database
    viewpoint_rowids = get_viewpoint_rowids_from_aid(aid_list, app)
    app.db.delete_rowids(VIEWPOINT_TABLE, viewpoint_rowids)
    # Add the new aid_list to the cache database
    colnames = ('annot_rowid', )
    params_iter = zip(aid_list_new)
    get_rowid_from_superkey = partial(get_viewpoint_rowids_from_aid, app=app)
    app.db.add_cleanly(VIEWPOINT_TABLE, colnames, params_iter, get_rowid_from_superkey)


################################################################################


def database_init(app):
    def convert_yaw_to_old_viewpoint(yaw):
        """
        we initially had viewpoint coordinates inverted
        (this function is its own inverse)
        """
        if yaw is None:
            return None
        angle = (-yaw + (const.TAU / 2)) % const.TAU
        return angle

    def rad_to_deg(radians):
        import numpy as np
        twopi = 2 * np.pi
        radians %= twopi
        return int((radians / twopi) * 360.0)

    if app.round <= 1:
        # Detection Review
        gid_list = app.ibs.get_valid_gids()
        image_reviewed_list = app.ibs.get_image_reviewed(gid_list)
        if not all([image_reviewed == 0 for image_reviewed in image_reviewed_list]):
            print("WARNING: NOT ALL IMAGES ARE NOT-REVIEWED")
            raw_input("Enter to set to not-reviewed...")
            app.ibs.set_image_reviewed(gid_list, [0] * len(gid_list))
        # Viewpoint Annotation
        aid_list = app.ibs.get_valid_aids()
        # Grab ALL viewpoints
        yaw_list = app.ibs.get_annot_yaws(aid_list)
        viewpoint_list = list(map(convert_yaw_to_old_viewpoint, yaw_list))
        viewpoint_list = [-1 if viewpoint is None else rad_to_deg(viewpoint) for viewpoint in viewpoint_list]
        if not all([viewpoint is -1 for viewpoint in viewpoint_list]):
            print("WARNING: NOT ALL ANNOT THETAS ARE NULLED")
            raw_input("Enter to null annot thetas...")
            app.ibs.set_annot_yaws(aid_list, [None] * len(aid_list))
            # Grab ALL viewpoints
    elif app.round == 2:
        # Detection Review
        gid_list = app.ibs.get_valid_gids(reviewed=0)
        aid_list = app.ibs.get_valid_aids(yaw=None)
    else:
        gid_list = app.ibs.get_valid_gids(reviewed=1)
        aid_list = app.ibs.get_valid_aids(include_only_gid_list=gid_list, yaw=None)

    # Detection Review
    image_reviewed_list = app.ibs.get_image_reviewed(gid_list)
    colnames = ('image_rowid', )
    params_iter = zip(gid_list)
    get_rowid_from_superkey = partial(get_review_rowids_from_gid, app=app)
    app.db.add_cleanly(REVIEW_TABLE, colnames, params_iter, get_rowid_from_superkey)

    # Viewpoint Annotation
    yaw_list = app.ibs.get_annot_yaws(aid_list)
    viewpoint_list = list(map(convert_yaw_to_old_viewpoint, yaw_list))
    viewpoint_list = [-1 if viewpoint is None else rad_to_deg(viewpoint) for viewpoint in viewpoint_list]
    # Add chips to temporary web viewpoint table
    colnames = ('annot_rowid', 'viewpoint_value_1')
    params_iter = zip(aid_list, viewpoint_list)
    get_rowid_from_superkey = partial(get_viewpoint_rowids_from_aid, app=app)
    app.db.add_cleanly(VIEWPOINT_TABLE, colnames, params_iter, get_rowid_from_superkey)

    print("ROUND: %d" % (app.round, ))
    print("WEB CACHED %d IMAGES" % (len(gid_list), ))
    print("WEB CACHED %d ANNOTATIONS" % (len(aid_list), ))
