# -*- coding: utf-8 -*-
"""
This model provides the declarative interface to all of the api_*_models
in guitool. Each different type of model/view has to register its iders,
getters, and potentially setters (hopefully if guitool ever gets off the
ground the delters as well)

Different columns can be hidden / shown by modifying this file

TODO: need to cache the total number of annotations or something about
imagesets on disk to help startuptime.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import six
from six.moves import zip, map, range
from wbia import constants as const
import utool as ut
from functools import partial

(print, rrr, profile) = ut.inject2(__name__)

IMAGESET_TABLE = const.IMAGESET_TABLE
IMAGE_TABLE = const.IMAGE_TABLE
ANNOTATION_TABLE = const.ANNOTATION_TABLE
IMAGE_GRID = 'image_grid'
NAME_TABLE = 'names'
NAMES_TREE = 'names_tree'
QRES_TABLE = 'qres'
THUMB_TABLE = 'thumbs'

# -----------------
# Define the tables
# -----------------


def make_table_declarations(ibs):
    """
    these used to be global variables, hopefully we can make them a little more
    configurable
    """
    # available tables
    TABLENAME_LIST = [
        IMAGE_TABLE,
        ANNOTATION_TABLE,
        # NAME_TABLE,
        IMAGESET_TABLE,
        IMAGE_GRID,
        THUMB_TABLE,
        NAMES_TREE,
    ]

    # table nice names
    TABLE_NICE = {
        IMAGE_TABLE: 'Image Table',
        ANNOTATION_TABLE: 'Annotations Table',
        NAME_TABLE: 'Name Table',
        QRES_TABLE: 'Query Results Table',
        IMAGESET_TABLE: 'ImageSet Table',
        IMAGE_GRID: 'Thumbnail Grid',
        THUMB_TABLE: 'Thumbnail Table',
        NAMES_TREE: 'Tree of Names',
    }

    # COLUMN DEFINITIONS
    # the columns each wbia table has,
    TABLE_COLNAMES = {
        IMAGE_TABLE: [
            'gid',
            'thumb',
            # 'nAids',
            'img_gname',
            # 'ext',
            'reviewed',  # detection reviewed flag is not fullyused
            'datetime',
            'gps',
            'orientation',
            'party_tag',
            'contributor_tag',
            # 'gdconf',
            'imgnotes',
            'image_uuid',
        ],
        # debug with
        # --noannottbl
        # --nonametree
        # even just aid seems to be very slow
        ANNOTATION_TABLE: [
            # 'annotation_uuid',
            'aid',
            'thumb',
            'annot_gname',
            'name',
            'exemplar',
            'species',  # <put back in
            'viewpoint',
            'quality_text',
            'age_min',
            'age_max',
            'sex_text',
            # 'rdconf',
            # 'nGt',  # ## <put back in
            'imagesettext_names',
            'annotnotes',  # ## <put back in
            'tag_text',  # < Hack should have actual tag structure
            # 'annot_visual_uuid',
            # 'nFeats',
            # 'bbox',
            # 'theta',
            # 'verts',
            # 'num_verts',
        ],
        NAME_TABLE: ['nid', 'name', 'nAids', 'namenotes'],
        QRES_TABLE: ['rank', 'score', 'name', 'aid'],
        IMAGESET_TABLE: [
            'imagesettext',
            'nImgs',
            # 'num_imgs_reviewed',
            # 'num_annotmatch_reviewed',
            # 'imageset_end_datetime',
            # 'imageset_processed_flag',
            # 'imageset_shipped_flag',
            'imgsetid',
        ],
        NAMES_TREE: [
            'name',
            'nAids',
            'thumb',
            'nid',
            # 'exemplar',
            # 'nExAids',
            'aid',
            # 'annot_gname',
            # 'quality_text',
            # 'age_min',
            # 'age_max',
            # 'sex_text',
            # 'imagesettext_names',
            # 'datetime',
            # 'max_hourdiff',
            # 'max_speed',
            # 'has_split',
            # 'namenotes',
        ],
        IMAGE_GRID: ['thumb'],
        # TEST TABLE
        THUMB_TABLE: ['img_gname', 'thumb'],
    }

    # dynamicly defined headers
    if not const.SIMPLIFY_INTERFACE:
        from wbia.control import accessor_decors

        if accessor_decors.API_CACHE:
            # Too slow without api cache
            TABLE_COLNAMES[IMAGESET_TABLE].extend(
                ['percent_annotmatch_reviewed_str', 'percent_names_with_exemplar_str']
            )
        TABLE_COLNAMES[IMAGESET_TABLE].extend(
            [
                # 'percent_imgs_reviewed_str',
                'imageset_start_datetime',
                # 'imageset_end_datetime',
                'imageset_duration',
                'imageset_notes',
            ]
        )

    if ibs.cfg.other_cfg.show_shipped_imagesets:
        TABLE_COLNAMES[IMAGESET_TABLE].extend(
            ['imageset_processed_flag', 'imageset_shipped_flag']
        )

    # THUMB_TABLE     : ['thumb' 'thumb' 'thumb' 'thumb'],
    # NAMES_TREE      : {('name' 'nid' 'nAids') : ['aid' 'bbox' 'thumb']}

    TABLE_TREE_LEVELS = {
        NAMES_TREE: {
            'name': 0,
            'namenotes': 0,
            'nid': 0,
            'nAids': 0,
            'nExAids': 0,
            'sex_text': 0,
            'exemplar': 1,
            'thumb': 1,
            'viewpoint': 1,
            'quality_text': 1,
            'age_min': 1,
            'age_max': 1,
            'imagesettext_names': 1,
            'aid': 1,
            'annot_gname': 1,
            'datetime': 1,
            'max_hourdiff': 0,
            'max_speed': 0,
            'has_split': 0,
        },
    }

    # the columns which are editable
    TABLE_EDITSET = {
        IMAGE_TABLE: set(['reviewed', 'imgnotes', 'gps']),
        ANNOTATION_TABLE: set(
            [
                'name',
                'species',
                'annotnotes',
                'exemplar',
                'viewpoint',
                'quality_text',
                'age_min',
                'age_max',
                'sex_text',
                'tag_text',
            ]
        ),
        NAME_TABLE: set(['name', 'namenotes']),
        QRES_TABLE: set(['name']),
        IMAGESET_TABLE: set(
            ['imagesettext', 'imageset_shipped_flag', 'imageset_processed_flag']
        ),
        IMAGE_GRID: set([]),
        THUMB_TABLE: set([]),
        NAMES_TREE: set(
            [
                'exemplar',
                'name',
                'namenotes',
                'viewpoint',
                'quality_text',
                'age_min',
                'age_max',
                'sex_text',
            ]
        ),
    }

    if const.SIMPLIFY_INTERFACE:
        TABLE_EDITSET[NAMES_TREE].remove('name')

    TABLE_HIDDEN_LIST = {
        # IMAGE_TABLE      : [False, True, False, False, False, True, False, False, False, False, False],
        # ANNOTATION_TABLE : [False, False, False, False, False, False, False, True, True, True, True, True, True],
        # NAMES_TREE       : [False, False, False, False, False, False],
        # NAME_TABLE       : [False, False, False, False],
    }

    TABLE_STRIPE_LIST = {
        IMAGE_GRID: 9,
    }

    # Define the valid columns a table could have
    COL_DEF = dict(
        [
            ('annot_visual_uuid', (str, 'Annot Visual UUID')),
            ('image_uuid', (str, 'Image UUID')),
            ('gid', (int, 'Image ID')),
            ('aid', (int, 'Annotation ID')),
            ('nid', (int, 'Name ID')),
            ('imgsetid', (int, 'ImageSet ID')),
            ('nAids', (int, '#Annots')),
            ('nExAids', (int, '#Exemplars')),
            ('nGt', (int, '#GT')),
            ('nImgs', (int, '#Imgs')),
            ('nFeats', (int, '#Features')),
            ('quality_text', (str, 'Quality')),
            ('imagesettext_names', (str, 'ImageSet Names')),
            ('age_min', (int, 'Age (min)')),
            ('age_max', (int, 'Age (max)')),
            ('sex_text', (str, 'Sex')),
            ('rank', (str, 'Rank')),  # needs to be a string for !Query
            ('unixtime', (float, 'unixtime')),
            ('species', (str, 'Species')),
            ('viewpoint', (str, 'Viewpoint')),
            ('img_gname', (str, 'Image Name')),
            ('annot_gname', (str, 'Source Image')),
            ('gdconf', (str, 'Detection Confidence')),
            ('rdconf', (float, 'Detection Confidence')),
            ('name', (str, 'Name')),
            ('annotnotes', (str, 'Annot Notes')),
            ('namenotes', (str, 'Name Notes')),
            ('imgnotes', (str, 'Image Notes')),
            ('match_name', (str, 'Matching Name')),
            ('bbox', (str, 'BBOX (x, y, w, h))')),  # Non editables are safe as strs
            ('num_verts', (int, 'NumVerts')),
            ('verts', (str, 'Verts')),
            ('score', (str, 'Confidence')),
            ('theta', (str, 'Theta')),
            ('reviewed', (bool, 'Detection Reviewed')),
            ('exemplar', (bool, 'Is Exemplar')),
            ('imagesettext', (str, 'ImageSet')),
            ('datetime', (str, 'Date / Time')),
            ('ext', (str, 'EXT')),
            ('thumb', ('PIXMAP', 'Thumb')),
            ('gps', (str, 'GPS')),
            ('orientation', (str, 'Orientation')),
            ('imageset_processed_flag', (bool, 'Processed')),
            ('imageset_shipped_flag', (bool, 'Commited')),
            ('imageset_start_datetime', (str, 'Start Time')),
            ('imageset_end_datetime', (str, 'End Time')),
            ('imageset_duration', (str, 'Duration')),
            ('imageset_notes', (str, 'Notes')),
            ('party_tag', (str, 'Party')),
            ('contributor_tag', (str, 'Contributor')),
            ('percent_imgs_reviewed_str', (str, '%Imgs Reviewed')),
            ('percent_annotmatch_reviewed_str', (str, '%Queried')),
            ('num_imgs_reviewed', (str, '#Imgs Reviewed')),
            ('num_annotmatch_reviewed', (str, '#Matches Reviewed')),
            ('percent_names_with_exemplar_str', (str, '%Names with Exemplar')),
            ('max_speed', (float, 'Max Speed km/h')),
            ('has_split', (float, 'Needs Split')),
            ('max_hourdiff', (float, 'Max Hour Diff')),
            ('tag_text', (str, 'Tags')),
        ]
    )

    declare_tup = (
        TABLENAME_LIST,
        TABLE_NICE,
        TABLE_COLNAMES,
        TABLE_TREE_LEVELS,
        TABLE_EDITSET,
        TABLE_HIDDEN_LIST,
        TABLE_STRIPE_LIST,
        COL_DEF,
    )
    return declare_tup


# ----


def partial_imap_1to1(func, si_func):
    import functools

    @functools.wraps(si_func)
    def wrapper(input_):
        if not ut.isiterable(input_):
            return func(si_func(input_))
        else:
            return list(map(func, si_func(input_)))

    ut.set_funcname(
        wrapper, ut.get_callable_name(func) + '_mapper_' + ut.get_funcname(si_func)
    )
    return wrapper


def _tupstr(tuple_):
    """ maps each item in tuple to a string and doesnt include parens """
    return ', '.join(list(map(six.text_type, tuple_)))


def make_wbia_headers_dict(ibs):
    declare_tup = make_table_declarations(ibs)
    (
        TABLENAME_LIST,
        TABLE_NICE,
        TABLE_COLNAMES,
        TABLE_TREE_LEVELS,
        TABLE_EDITSET,
        TABLE_HIDDEN_LIST,
        TABLE_STRIPE_LIST,
        COL_DEF,
    ) = declare_tup

    #
    # Table Iders/Setters/Getters
    iders = {}
    setters = {}
    getters = {}
    widths = {}

    def infer_unspecified_getters(tablename, shortname):
        for colname in TABLE_COLNAMES[tablename]:
            if colname not in getters[tablename]:
                if ut.VERBOSE:
                    print(
                        '[guiheaders] infering getter for tablename=%r, colname=%r'
                        % (tablename, colname,)
                    )
                    # print('[guiheaders] infering %r' % (getters[tablename][colname],))
                try:
                    getters[tablename][colname] = getattr(
                        ibs, 'get_' + shortname + '_' + colname
                    )
                except AttributeError:
                    # we have inconsistently put in column names
                    # try to "just make things work"
                    getters[tablename][colname] = getattr(ibs, 'get_' + colname)

    # +--------------------------
    # ImageSet Iders/Setters/Getters
    SHOW_SHIPPED_IMAGESETS = ibs.cfg.other_cfg.show_shipped_imagesets
    # SHOW_SHIPPED_IMAGESETS = True
    if SHOW_SHIPPED_IMAGESETS:
        iders[IMAGESET_TABLE] = [ibs.get_valid_imgsetids]
    else:
        iders[IMAGESET_TABLE] = [partial(ibs.get_valid_imgsetids, shipped=False)]

    getters[IMAGESET_TABLE] = {
        'imgsetid': lambda imgsetids: imgsetids,
        'nImgs': ibs.get_imageset_num_gids,
        'imagesettext': ibs.get_imageset_text,
        'imageset_shipped_flag': ibs.get_imageset_shipped_flags,
        'imageset_processed_flag': ibs.get_imageset_processed_flags,
        #
        'imageset_start_datetime': partial_imap_1to1(
            ut.unixtime_to_datetimestr, ibs.get_imageset_start_time_posix
        ),
        'imageset_end_datetime': partial_imap_1to1(
            ut.unixtime_to_datetimestr, ibs.get_imageset_end_time_posix
        ),
        #
        'imageset_start_time_posix': ibs.get_imageset_start_time_posix,
        'imageset_end_time_posix': ibs.get_imageset_end_time_posix,
        'imageset_duration': ibs.get_imageset_duration,
        'imageset_notes': ibs.get_imageset_note,
    }
    infer_unspecified_getters(IMAGESET_TABLE, 'imageset')
    setters[IMAGESET_TABLE] = {
        'imagesettext': ibs.set_imageset_text,
        'imageset_shipped_flag': ibs.set_imageset_shipped_flags,
        'imageset_processed_flag': ibs.set_imageset_processed_flags,
    }
    widths[IMAGESET_TABLE] = {
        'nImgs': 55,
    }
    # +--------------------------
    # Image Iders/Setters/Getters
    iders[IMAGE_TABLE] = [ibs.get_valid_gids]
    getters[IMAGE_TABLE] = {
        'gid': ut.identity,
        'imgsetid': ibs.get_image_imgsetids,
        'imagesettext': partial_imap_1to1(_tupstr, ibs.get_image_imagesettext),
        'reviewed': ibs.get_image_reviewed,
        'img_gname': ibs.get_image_gnames,
        'nAids': ibs.get_image_num_annotations,
        'unixtime': ibs.get_image_unixtime,
        'datetime': ibs.get_image_datetime_str,
        'gdconf': ibs.get_image_detect_confidence,
        'imgnotes': ibs.get_image_notes,
        'image_uuid': ibs.get_image_uuids,
        'ext': ibs.get_image_exts,
        'thumb': ibs.get_image_thumbtup,
        'gps': partial_imap_1to1(_tupstr, ibs.get_image_gps),
        'orientation': ibs.get_image_orientation_str,
    }
    infer_unspecified_getters(IMAGE_TABLE, 'image')
    setters[IMAGE_TABLE] = {
        'reviewed': ibs.set_image_reviewed,
        'imgnotes': ibs.set_image_notes,
        'gps': ibs.set_image_gps_str,
    }
    # +--------------------------
    # IMAGE GRID
    iders[IMAGE_GRID] = [ibs.get_valid_gids]
    getters[IMAGE_GRID] = {
        'thumb': ibs.get_image_thumbtup,
        'img_gname': ibs.get_image_gnames,
        'aid': ibs.get_image_aids,
    }
    setters[IMAGE_GRID] = {}
    # +--------------------------
    # ANNOTATION Iders/Setters/Getters
    iders[ANNOTATION_TABLE] = [ibs.get_valid_aids]
    getters[ANNOTATION_TABLE] = {
        'aid': ut.identity,
        'name': ibs.get_annot_names,
        'species': ibs.get_annot_species_texts,
        'viewpoint': ibs.get_annot_viewpoints,
        'quality_text': ibs.get_annot_quality_texts,
        'imagesettext_names': ibs.get_annot_image_set_texts,
        'age_min': ibs.get_annot_age_months_est_min,
        'age_max': ibs.get_annot_age_months_est_max,
        'sex_text': ibs.get_annot_sex_texts,
        'annot_gname': ibs.get_annot_image_names,
        'nGt': ibs.get_annot_num_groundtruth,
        'theta': partial_imap_1to1(ut.theta_str, ibs.get_annot_thetas),
        'bbox': partial_imap_1to1(ut.bbox_str, ibs.get_annot_bboxes),
        'num_verts': ibs.get_annot_num_verts,
        'verts': partial_imap_1to1(ut.verts_str, ibs.get_annot_verts),
        'nFeats': ibs.get_annot_num_feats,
        'rdconf': ibs.get_annot_detect_confidence,
        'annotnotes': ibs.get_annot_notes,
        'thumb': ibs.get_annot_chip_thumbtup,
        'exemplar': ibs.get_annot_exemplar_flags,
        'annot_visual_uuid': ibs.get_annot_visual_uuids,
        'datetime': ibs.get_annot_image_datetime_str,
    }
    infer_unspecified_getters(ANNOTATION_TABLE, 'annot')
    setters[ANNOTATION_TABLE] = {
        'name': ibs.set_annot_names,
        'species': ibs.set_annot_species_and_notify,
        'viewpoint': ibs.set_annot_viewpoints,
        'age_min': ibs.set_annot_age_months_est_min,
        'age_max': ibs.set_annot_age_months_est_max,
        'sex_text': ibs.set_annot_sex_texts,
        'annotnotes': ibs.set_annot_notes,
        'exemplar': ibs.set_annot_exemplar_flags,
        'quality_text': ibs.set_annot_quality_texts,
        'tag_text': ibs.set_annot_tag_text,
    }
    # +--------------------------
    # Name Iders/Setters/Getters
    iders[NAME_TABLE] = [ibs.get_valid_nids]
    getters[NAME_TABLE] = {
        'nid': ut.identity,
        'name': ibs.get_name_texts,
        'nAids': ibs.get_name_num_annotations,
        'namenotes': ibs.get_name_notes,
        # 'has_split'  : ibs.get_name_has_split,
    }
    setters[NAME_TABLE] = {
        'name': ibs.set_name_texts,
        'namenotes': ibs.set_name_notes,
    }
    # +--------------------------
    # NAMES TREE
    iders[NAMES_TREE] = [ibs.get_valid_nids, ibs.get_name_aids]
    getters[NAMES_TREE] = {
        # level 0
        'nid': ut.identity,
        'name': ibs.get_name_texts,
        'nAids': ibs.get_name_num_annotations,
        'nExAids': ibs.get_name_num_exemplar_annotations,
        'namenotes': ibs.get_name_notes,
        'sex_text': ibs.get_name_sex_text,
        # level 1
        'aid': ut.identity,
        'exemplar': ibs.get_annot_exemplar_flags,
        'thumb': ibs.get_annot_chip_thumbtup,
        'annot_gname': ibs.get_annot_image_names,
        'age_min': ibs.get_annot_age_months_est_min,
        'age_max': ibs.get_annot_age_months_est_max,
        'imagesettext_names': ibs.get_annot_image_set_texts,
        'viewpoint': getters[ANNOTATION_TABLE]['viewpoint'],
        'quality_text': getters[ANNOTATION_TABLE]['quality_text'],
        'datetime': getters[ANNOTATION_TABLE]['datetime'],
    }
    setters[NAMES_TREE] = {
        'name': ibs.set_name_texts,
        'namenotes': ibs.set_name_notes,
        'sex_text': ibs.set_name_sex_text,
        'age_min': ibs.set_annot_age_months_est_min,
        'age_max': ibs.set_annot_age_months_est_max,
        'exemplar': setters[ANNOTATION_TABLE]['exemplar'],
        'viewpoint': setters[ANNOTATION_TABLE]['viewpoint'],
        'quality_text': setters[ANNOTATION_TABLE]['quality_text'],
    }
    widths[NAMES_TREE] = {
        'thumb': lambda: ibs.cfg.other_cfg.thumb_size,
        'nAids': 65,
        'nid': 50,
    }
    infer_unspecified_getters(NAMES_TREE, 'name')
    # +--------------------------
    # THUMB TABLE
    iders[THUMB_TABLE] = [ibs.get_valid_gids]
    getters[THUMB_TABLE] = {
        'thumb': ibs.get_image_thumbtup,
        'img_gname': ibs.get_image_gnames,
        'aid': ibs.get_image_aids,
    }
    setters[THUMB_TABLE] = {}
    # L________________________

    def make_header(tblname):
        """
        Args:
            table_name - the internal table name
        """
        tblnice = TABLE_NICE[tblname]
        colnames = TABLE_COLNAMES[tblname]
        editset = TABLE_EDITSET[tblname]
        tblgetters = getters[tblname]
        tblsetters = setters[tblname]
        # if levels aren't found, we're not dealing with a tree, so everything is at level 0
        collevel_dict = TABLE_TREE_LEVELS.get(tblname, ut.ddict(lambda: 0))
        collevels = [collevel_dict[colname] for colname in colnames]
        hiddencols = TABLE_HIDDEN_LIST.get(tblname, [False for _ in range(len(colnames))])
        numstripes = TABLE_STRIPE_LIST.get(tblname, 1)

        colwidths_dict = widths.get(tblname, {})
        colwidths = [colwidths_dict.get(colname, 100) for colname in colnames]

        def get_column_data(colname):
            try:
                coldef_tup = COL_DEF[colname]
                coltype, colnice = coldef_tup
            except KeyError as ex:
                strict = False
                ut.printex(
                    ex,
                    'Need to add type info for colname=%r to COL_DEF' % colname,
                    iswarning=not strict,
                )
                if strict:
                    raise
                else:
                    # default coldef to give a string type and nice=colname
                    coltype, colnice = (str, colname)
            coledit = colname in editset
            colgetter = tblgetters[colname]
            colsetter = None if not coledit else tblsetters.get(colname, None)
            return (coltype, colnice, coledit, colgetter, colsetter)

        try:
            _tuplist = list(zip(*list(map(get_column_data, colnames))))
            (coltypes, colnices, coledits, colgetters, colsetters) = _tuplist
        except KeyError as ex:
            ut.printex(ex, key_list=['tblname', 'colnames'])
            raise
        header = {
            'name': tblname,
            'nice': tblnice,
            'iders': iders[tblname],
            'col_name_list': colnames,
            'col_type_list': coltypes,
            'col_nice_list': colnices,
            'col_edit_list': coledits,
            'col_getter_list': colgetters,
            'col_setter_list': colsetters,
            'col_level_list': collevels,
            'col_hidden_list': hiddencols,
            'num_duplicates': numstripes,
            'get_thumb_size': lambda: ibs.cfg.other_cfg.thumb_size,
            'col_width_list': colwidths,  # TODO
        }
        return header

    header_dict = {tblname: make_header(tblname) for tblname in TABLENAME_LIST}
    return header_dict, declare_tup
