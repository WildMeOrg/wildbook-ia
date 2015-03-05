"""
Module Licence and docstring

TODO: ideally the ibeis.constants module would not be used here
and each function would use its own constant variables that are suffixed
with the last version number that they existed in

TODO: Add a table for original_image_path
"""
from __future__ import absolute_import, division, print_function
from ibeis import constants as const
try:
    from ibeis.control import DB_SCHEMA_CURRENT
    UPDATE_CURRENT  = DB_SCHEMA_CURRENT.update_current
    VERSION_CURRENT = DB_SCHEMA_CURRENT.VERSION_CURRENT
except:
    UPDATE_CURRENT  = None
    VERSION_CURRENT = None
    print("[dbcache] NO DB_SCHEMA_CURRENT AUTO-GENERATED!")
import utool
profile = utool.profile


NAME_TABLE_v121   = const.NAME_TABLE_v121
NAME_TABLE_v130   = const.NAME_TABLE_v130
ANNOT_VISUAL_UUID = 'annot_visual_uuid'
ANNOT_SEMANTIC_UUID = 'annot_semantic_uuid'
ANNOT_UUID        = 'annot_uuid'
ANNOT_YAW         = 'annot_yaw'
ANNOT_VIEWPOINT   = 'annot_viewpoint'


# =======================
# Schema Version 1.0.0
# =======================


@profile
def update_1_0_0(db, ibs=None):
    db.add_table(const.IMAGE_TABLE, (
        ('image_rowid',                  'INTEGER PRIMARY KEY'),
        ('image_uuid',                   'UUID NOT NULL'),
        ('image_uri',                    'TEXT NOT NULL'),
        ('image_ext',                    'TEXT NOT NULL'),
        ('image_original_name',          'TEXT NOT NULL'),  # We could parse this out of original_path
        ('image_width',                  'INTEGER DEFAULT -1'),
        ('image_height',                 'INTEGER DEFAULT -1'),
        ('image_time_posix',             'INTEGER DEFAULT -1'),  # this should probably be UCT
        ('image_gps_lat',                'REAL DEFAULT -1.0'),   # there doesn't seem to exist a GPSPoint in SQLite (TODO: make one in the __SQLITE3__ custom types
        ('image_gps_lon',                'REAL DEFAULT -1.0'),
        ('image_toggle_enabled',         'INTEGER DEFAULT 0'),
        ('image_toggle_reviewed',        'INTEGER DEFAULT 0'),
        ('image_note',                   'TEXT',),
    ),
        superkey_colnames_list=[('image_uuid',)],
        docstr='''
        First class table used to store image locations and meta-data''')

    db.add_table(const.ENCOUNTER_TABLE, (
        ('encounter_rowid',              'INTEGER PRIMARY KEY'),
        ('encounter_uuid',               'UUID NOT NULL'),
        ('encounter_text',               'TEXT NOT NULL'),
        ('encounter_note',               'TEXT NOT NULL'),
    ),
        superkey_colnames_list=[('encounter_text',)],
        docstr='''
        List of all encounters''')

    db.add_table(const.LBLTYPE_TABLE, (
        ('lbltype_rowid',                'INTEGER PRIMARY KEY'),
        ('lbltype_text',                 'TEXT NOT NULL'),
        ('lbltype_default',              'TEXT NOT NULL'),
    ),
        superkey_colnames_list=[('lbltype_text',)],
        docstr='''
        List of keys used to define the categories of annotation lables, text
        is for human-readability. The lbltype_default specifies the
        lblannot_value of annotations with a relationship of some
        lbltype_rowid''')

    db.add_table(const.CONFIG_TABLE, (
        ('config_rowid',                 'INTEGER PRIMARY KEY'),
        ('config_suffix',                'TEXT NOT NULL'),
    ),
        superkey_colnames_list=[('config_suffix',)],
        docstr='''
        Used to store the ids of algorithm configurations that generate
        annotation lblannots.  Each user will have a config id for manual
        contributions ''')

    ##########################
    # FIRST ORDER            #
    ##########################
    db.add_table(const.ANNOTATION_TABLE, (
        ('annot_rowid',                  'INTEGER PRIMARY KEY'),
        (ANNOT_UUID,                     'UUID NOT NULL'),
        ('image_rowid',                  'INTEGER NOT NULL'),
        ('annot_xtl',                    'INTEGER NOT NULL'),
        ('annot_ytl',                    'INTEGER NOT NULL'),
        ('annot_width',                  'INTEGER NOT NULL'),
        ('annot_height',                 'INTEGER NOT NULL'),
        ('annot_theta',                  'REAL DEFAULT 0.0'),
        ('annot_num_verts',              'INTEGER NOT NULL'),
        ('annot_verts',                  'TEXT'),
        ('annot_detect_confidence',      'REAL DEFAULT -1.0'),
        ('annot_exemplar_flag',          'INTEGER DEFAULT 0'),
        ('annot_note',                   'TEXT'),
    ),
        superkey_colnames_list=[(ANNOT_UUID,)],
        docstr='''
        Mainly used to store the geometry of the annotation within its parent
        image The one-to-many relationship between images and annotations is
        encoded here Attributes are stored in the Annotation Label Relationship
        Table''')

    db.add_table(const.LBLIMAGE_TABLE, (
        ('lblimage_rowid',               'INTEGER PRIMARY KEY'),
        ('lblimage_uuid',                'UUID NOT NULL'),
        ('lbltype_rowid',                'INTEGER NOT NULL'),  # this is "category" in the proposal
        ('lblimage_value',               'TEXT NOT NULL'),
        ('lblimage_note',                'TEXT'),
    ),
        superkey_colnames_list=[('lbltype_rowid', 'lblimage_value',)],
        docstr='''
        Used to store the labels (attributes) of images''')

    db.add_table(const.LBLANNOT_TABLE, (
        ('lblannot_rowid',               'INTEGER PRIMARY KEY'),
        ('lblannot_uuid',                'UUID NOT NULL'),
        ('lbltype_rowid',                'INTEGER NOT NULL'),  # this is "category" in the proposal
        ('lblannot_value',               'TEXT NOT NULL'),
        ('lblannot_note',                'TEXT'),
    ),
        superkey_colnames_list=[('lbltype_rowid', 'lblannot_value',)],
        docstr='''
        Used to store the labels / attributes of annotations.
        E.G name, species ''')

    ##########################
    # SECOND ORDER           #
    ##########################
    # TODO: constraint needs modification
    db.add_table(const.CHIP_TABLE, (
        ('chip_rowid',                   'INTEGER PRIMARY KEY'),
        ('annot_rowid',                  'INTEGER NOT NULL'),
        ('config_rowid',                 'INTEGER DEFAULT 0'),
        ('chip_uri',                     'TEXT'),
        ('chip_width',                   'INTEGER NOT NULL'),
        ('chip_height',                  'INTEGER NOT NULL'),
    ),
        superkey_colnames_list=[('annot_rowid', 'config_rowid',)],
        docstr='''
        Used to store *processed* annots as chips''')

    db.add_table(const.FEATURE_TABLE, (
        ('feature_rowid',                'INTEGER PRIMARY KEY'),
        ('chip_rowid',                   'INTEGER NOT NULL'),
        ('config_rowid',                 'INTEGER DEFAULT 0'),
        ('feature_num_feats',            'INTEGER NOT NULL'),
        ('feature_keypoints',            'NUMPY'),
        ('feature_sifts',                'NUMPY'),
    ),
        superkey_colnames_list=[('chip_rowid, config_rowid',)],
        docstr='''
        Used to store individual chip features (ellipses)''')

    db.add_table(const.EG_RELATION_TABLE, (
        ('egr_rowid',                    'INTEGER PRIMARY KEY'),
        ('image_rowid',                  'INTEGER NOT NULL'),
        ('encounter_rowid',              'INTEGER'),
    ),
        superkey_colnames_list=[('image_rowid, encounter_rowid',)],
        docstr='''
        Relationship between encounters and images (many to many mapping) the
        many-to-many relationship between images and encounters is encoded here
        encounter_image_relationship stands for encounter-image-pairs.''')

    ##########################
    # THIRD ORDER            #
    ##########################
    db.add_table(const.GL_RELATION_TABLE, (
        ('glr_rowid',                    'INTEGER PRIMARY KEY'),
        ('image_rowid',                  'INTEGER NOT NULL'),
        ('lblimage_rowid',               'INTEGER NOT NULL'),
        ('config_rowid',                 'INTEGER DEFAULT 0'),
        ('glr_confidence',               'REAL DEFAULT 0.0'),
    ),
        superkey_colnames_list=[('image_rowid', 'lblimage_rowid', 'config_rowid',)],
        docstr='''
        Used to store one-to-many the relationship between images
        and labels''')

    db.add_table(const.AL_RELATION_TABLE, (
        ('alr_rowid',                    'INTEGER PRIMARY KEY'),
        ('annot_rowid',                  'INTEGER NOT NULL'),
        ('lblannot_rowid',               'INTEGER NOT NULL'),
        ('config_rowid',                 'INTEGER DEFAULT 0'),
        ('alr_confidence',               'REAL DEFAULT 0.0'),
    ),
        superkey_colnames_list=[('annot_rowid', 'lblannot_rowid', 'config_rowid',)],
        docstr='''
        Used to store one-to-many the relationship between annotations (annots)
        and labels''')


def post_1_0_0(db, ibs=None):
    # We are dropping the versions table and rather using the metadata table
    print('applying post_1_0_0')
    db.drop_table(const.VERSIONS_TABLE)


def post_1_2_0(db, ibs=None):
    print('applying post_1_2_0')

    def schema_1_2_0_postprocess_fixuuids(ibs):
        """
        schema_1_2_0_postprocess_fixuuids

        Args:
            ibs (IBEISController):

        Example:
            >>> # DISABLE_DOCTEST
            >>> from ibeis.model.preproc.preproc_annot import *  # NOQA
            >>> import ibeis
            >>> #import sys
            #>>> sys.argv.append('--force-fresh')
            #>>> ibs = ibeis.opendb('PZ_MTEST')
            #>>> ibs = ibeis.opendb('testdb1')
            >>> ibs = ibeis.opendb('GZ_ALL')
            >>> # should be auto applied
            >>> ibs.print_annotation_table(verbosity=1)
            >>> result = schema_1_2_0_postprocess_fixuuids(ibs)
            >>> ibs.print_annotation_table(verbosity=1)
        """
        import utool as ut

        aid_list = ibs.get_valid_aids()
        #ibs.get_annot_name_rowids(aid_list)
        #ANNOT_PARENT_ROWID      = 'annot_parent_rowid'
        #ANNOT_ROWID             = 'annot_rowid'
        ANNOTATION_TABLE        = 'annotations'
        ANNOT_SEMANTIC_UUID     = 'annot_semantic_uuid'
        NAME_ROWID              = 'name_rowid'
        SPECIES_ROWID           = 'species_rowid'
        AL_RELATION_TABLE    = 'annotation_lblannot_relationship'

        def set_annot_semantic_uuids(ibs, aid_list, annot_semantic_uuid_list):
            id_iter = aid_list
            colnames = (ANNOT_SEMANTIC_UUID,)
            ibs.db.set(ANNOTATION_TABLE, colnames,
                       annot_semantic_uuid_list, id_iter)

        def set_annot_visual_uuids(ibs, aid_list, annot_visual_uuid_list):
            id_iter = aid_list
            colnames = (ANNOT_VISUAL_UUID,)
            ibs.db.set(ANNOTATION_TABLE, colnames,
                       annot_visual_uuid_list, id_iter)

        def set_annot_species_rowids(ibs, aid_list, species_rowid_list):
            id_iter = aid_list
            colnames = (SPECIES_ROWID,)
            ibs.db.set(
                ANNOTATION_TABLE, colnames, species_rowid_list, id_iter)

        def set_annot_name_rowids(ibs, aid_list, name_rowid_list):
            id_iter = aid_list
            colnames = (NAME_ROWID,)
            ibs.db.set(ANNOTATION_TABLE, colnames, name_rowid_list, id_iter)

        def get_alr_lblannot_rowids(alrid_list):
            lblannot_rowids_list = ibs.db.get(AL_RELATION_TABLE, ('lblannot_rowid',), alrid_list)
            return lblannot_rowids_list

        def get_annot_alrids_oftype(aid_list, lbltype_rowid):
            """
            Get all the relationship ids belonging to the input annotations where the
            relationship ids are filtered to be only of a specific lbltype/category/type
            """
            alrids_list = ibs.db.get(AL_RELATION_TABLE, ('alr_rowid',), aid_list, id_colname='annot_rowid', unpack_scalars=False)
            # Get lblannot_rowid of each relationship
            lblannot_rowids_list = ibs.unflat_map(get_alr_lblannot_rowids, alrids_list)
            # Get the type of each lblannot
            lbltype_rowids_list = ibs.unflat_map(ibs.get_lblannot_lbltypes_rowids, lblannot_rowids_list)
            # only want the nids of individuals, not species, for example
            valids_list = [[typeid == lbltype_rowid for typeid in rowids] for rowids in lbltype_rowids_list]
            alrids_list = [ut.filter_items(alrids, valids) for alrids, valids in zip(alrids_list, valids_list)]
            alrids_list = [
                alrid_list[0:1]
                if len(alrid_list) > 1 else
                alrid_list
                for alrid_list in alrids_list
            ]
            assert all([len(alrid_list) < 2 for alrid_list in alrids_list]),\
                ("More than one type per lbltype.  ALRIDS: " + str(alrids_list) +
                 ", ROW: " + str(lbltype_rowid) + ", KEYS:" + str(ibs.lbltype_ids))
            return alrids_list

        def get_annot_speciesid_from_lblannot_relation(aid_list, distinguish_unknowns=True):
            """ function for getting speciesid the old way """
            species_lbltype_rowid = ibs.db.get('keys', ('lbltype_rowid',), ('SPECIES_KEY',), id_colname='lbltype_text')[0]
            alrids_list = get_annot_alrids_oftype(aid_list, species_lbltype_rowid)
            lblannot_rowids_list = ibs.unflat_map(get_alr_lblannot_rowids, alrids_list)
            speciesid_list = [lblannot_rowids[0] if len(lblannot_rowids) > 0 else ibs.UNKNOWN_LBLANNOT_ROWID for
                              lblannot_rowids in lblannot_rowids_list]
            return speciesid_list

        def get_annot_name_rowids_from_lblannot_relation(aid_list):
            """ function for getting nids the old way """
            individual_lbltype_rowid = ibs.db.get('keys', ('lbltype_rowid',), ('INDIVIDUAL_KEY',), id_colname='lbltype_text')[0]
            alrids_list = get_annot_alrids_oftype(aid_list, individual_lbltype_rowid)
            lblannot_rowids_list = ibs.unflat_map(get_alr_lblannot_rowids, alrids_list)
            # Get a single nid from the list of lblannot_rowids of type INDIVIDUAL
            # TODO: get index of highest confidencename
            nid_list = [lblannot_rowids[0] if len(lblannot_rowids) > 0 else ibs.UNKNOWN_LBLANNOT_ROWID for
                         lblannot_rowids in lblannot_rowids_list]
            return nid_list

        nid_list1 = ibs.get_annot_name_rowids(aid_list, distinguish_unknowns=False)
        speciesid_list1 = ibs.get_annot_species_rowids(aid_list)

        # Get old values from lblannot table
        nid_list = get_annot_name_rowids_from_lblannot_relation(aid_list)
        speciesid_list = get_annot_speciesid_from_lblannot_relation(aid_list)

        assert len(nid_list1) == len(nid_list), 'cannot update to 1_2_0 name length error'
        assert len(speciesid_list1) == len(speciesid_list), 'cannot update to 1_2_0 species length error'

        if (ut.list_all_eq_to(nid_list, 0) and ut.list_all_eq_to(speciesid_list, 0)):
            print('... returning No information in lblannot table to transfer')
            return

        # Make sure information has not gotten out of sync
        try:
            assert all([(nid1 == nid or nid1 == 0) for nid1, nid in zip(nid_list1, nid_list)])
            assert all([(sid1 == sid or sid1 == 0) for sid1, sid in zip(speciesid_list1, speciesid_list)])
        except AssertionError as ex:
            ut.printex(ex, 'Cannot update database to 1_2_0 information out of sync')
            raise

        # Move values into the annotation table as a native column
        set_annot_name_rowids(ibs, aid_list, nid_list)
        set_annot_species_rowids(ibs, aid_list, speciesid_list)
        # Update visual uuids
        # Moved this to post_process 1.21
        #ibs.update_annot_visual_uuids(aid_list)
        #ibs.update_annot_semantic_uuids(aid_list)

    #ibs.print_annotation_table(verbosity=1)
    if ibs is not None:
        ibs._init_rowid_constants()
        schema_1_2_0_postprocess_fixuuids(ibs)
    else:
        print('warning: ibs is None, so cannot apply name/species column fixes to existing database')


def post_1_2_1(db, ibs=None):
    if ibs is not None:
        print('applying post_1_2_1')
        import utool as ut
        from ibeis.model.preproc import preproc_annot
        if ibs is not None:
            ibs._init_rowid_constants()
            #db = ibs.db
        UNKNOWN_ROWID = 0
        UNKNOWN             = '____'
        UNKNOWN_NAME_ROWID  = 0
        ANNOTATION_TABLE    = 'annotations'
        SPECIES_TABLE       = 'species'
        LBLANNOT_TABLE      = 'lblannot'
        SPECIES_ROWID       = 'species_rowid'
        NAME_ROWID          = 'name_rowid'
        NAME_TEXT           = 'name_text'
        ANNOT_SEMANTIC_UUID = 'annot_semantic_uuid'
        lblannot_colnames   = ('lblannot_uuid', 'lblannot_value', 'lblannot_note',)
        name_colnames       = ('name_uuid', 'name_text', 'name_note',)
        species_colspeciess = ('species_uuid', 'species_text', 'species_note',)
        # Get old name and species rowids from annotaiton tables
        aid_list = db.get_all_rowids(ANNOTATION_TABLE)
        name_rowids1    = db.get(ANNOTATION_TABLE, (NAME_ROWID,), aid_list)
        species_rowids1 = db.get(ANNOTATION_TABLE, (SPECIES_ROWID,), aid_list)
        # Look at the unique non-unknown ones
        unique_name_rowids1    = sorted(list(   set(name_rowids1) - set([UNKNOWN_ROWID])))
        unique_species_rowids1 = sorted(list(set(species_rowids1) - set([UNKNOWN_ROWID])))
        # Get params out of label annotation tables
        name_params_list    = db.get(LBLANNOT_TABLE, lblannot_colnames, unique_name_rowids1)
        species_params_list = db.get(LBLANNOT_TABLE, lblannot_colnames, unique_species_rowids1)
        # Move params into name and species tables
        unique_name_rowids2    = db._add(NAME_TABLE_v121,    name_colnames,       name_params_list)
        unique_species_rowids2 = db._add(SPECIES_TABLE, species_colspeciess, species_params_list)
        # Build mapping from old table to new table
        name_rowid_mapping = dict(zip(unique_name_rowids1, unique_name_rowids2))
        speices_rowid_mapping = dict(zip(unique_species_rowids1, unique_species_rowids2))
        name_rowid_mapping[UNKNOWN_ROWID] = UNKNOWN_ROWID
        speices_rowid_mapping[UNKNOWN_ROWID] = UNKNOWN_ROWID
        # Apply mapping
        name_rowids2   = ut.dict_take_list(name_rowid_mapping, name_rowids1)
        species_rowid2 = ut.dict_take_list(speices_rowid_mapping, species_rowids1)
        # Put new rowids back into annotation table
        db.set(ANNOTATION_TABLE, (NAME_ROWID,), name_rowids2, aid_list)
        db.set(ANNOTATION_TABLE, (SPECIES_ROWID,), species_rowid2, aid_list)
        # HACK TODO use actual SQL to fix and move to 1.2.0

        def get_annot_names_v121(aid_list):
            name_rowid_list = ibs.get_annot_name_rowids(aid_list)
            name_text_list = ibs.db.get(NAME_TABLE_v121, (NAME_TEXT,), name_rowid_list)
            name_text_list = [
                UNKNOWN
                if rowid == UNKNOWN_NAME_ROWID or name_text is None
                else name_text
                for name_text, rowid in zip(name_text_list, name_rowid_list)]
            return name_text_list

        def get_annot_semantic_uuid_info_v121(aid_list, _visual_infotup):
            visual_infotup = _visual_infotup
            image_uuid_list, verts_list, theta_list = visual_infotup
            # It is visual info augmented with name and species
            def get_annot_viewpoints(ibs, aid_list):
                viewpoint_list = ibs.db.get(const.ANNOTATION_TABLE, (ANNOT_VIEWPOINT,), aid_list)
                viewpoint_list = [viewpoint if viewpoint >= 0.0 else None for viewpoint in viewpoint_list]
                return viewpoint_list
            view_list       = get_annot_viewpoints(ibs, aid_list)
            name_list       = get_annot_names_v121(aid_list)
            species_list    = ibs.get_annot_species_texts(aid_list)
            semantic_infotup = (image_uuid_list, verts_list, theta_list, view_list,
                                name_list, species_list)
            return semantic_infotup

        def get_annot_visual_uuid_info_v121(aid_list):
            image_uuid_list = ibs.get_annot_image_uuids(aid_list)
            verts_list      = ibs.get_annot_verts(aid_list)
            theta_list      = ibs.get_annot_thetas(aid_list)
            visual_infotup = (image_uuid_list, verts_list, theta_list)
            return visual_infotup

        def update_annot_semantic_uuids_v121(aid_list, _visual_infotup=None):
            semantic_infotup = get_annot_semantic_uuid_info_v121(aid_list, _visual_infotup)
            annot_semantic_uuid_list = preproc_annot.make_annot_semantic_uuid(semantic_infotup)
            ibs.db.set(ANNOTATION_TABLE, (ANNOT_SEMANTIC_UUID,), annot_semantic_uuid_list, aid_list)

        def update_annot_visual_uuids_v121(aid_list):
            visual_infotup = get_annot_visual_uuid_info_v121(aid_list)
            annot_visual_uuid_list = preproc_annot.make_annot_visual_uuid(visual_infotup)
            ibs.db.set(ANNOTATION_TABLE, (ANNOT_VISUAL_UUID,), annot_visual_uuid_list, aid_list)
            # If visual uuids are changes semantic ones are also changed
            update_annot_semantic_uuids_v121(aid_list, visual_infotup)
        update_annot_visual_uuids_v121(aid_list)


def post_1_3_4(db, ibs=None):
    if ibs is not None:
        ibs._init_rowid_constants()
        ibs._init_config()
        ibs.update_annot_visual_uuids(ibs.get_valid_aids())


def pre_1_3_1(db, ibs=None):
    """
    need to ensure that visual uuid columns are unique before we add that
    constaint to sql. This will remove any annotations that are not unique
    """
    if ibs is not None:
        #from ibeis import ibsfuncs
        import utool as ut
        import six
        ibs._init_rowid_constants()
        ibs._init_config()
        aid_list = ibs.get_valid_aids()
        def pre_1_3_1_update_visual_uuids(ibs, aid_list):
            from ibeis.model.preproc import preproc_annot
            def pre_1_3_1_get_annot_visual_uuid_info(ibs, aid_list):
                image_uuid_list = ibs.get_annot_image_uuids(aid_list)
                verts_list      = ibs.get_annot_verts(aid_list)
                theta_list      = ibs.get_annot_thetas(aid_list)
                visual_infotup = (image_uuid_list, verts_list, theta_list)
                return visual_infotup
            def pre_1_3_1_get_annot_semantic_uuid_info(ibs, aid_list, _visual_infotup):
                image_uuid_list, verts_list, theta_list = visual_infotup
                def get_annot_viewpoints(ibs, aid_list):
                    viewpoint_list = ibs.db.get(const.ANNOTATION_TABLE, (ANNOT_VIEWPOINT,), aid_list)
                    viewpoint_list = [viewpoint if viewpoint >= 0.0 else None for viewpoint in viewpoint_list]
                    return viewpoint_list
                # It is visual info augmented with name and species
                viewpoint_list  = get_annot_viewpoints(ibs, aid_list)
                name_list       = ibs.get_annot_names(aid_list)
                species_list    = ibs.get_annot_species_texts(aid_list)
                semantic_infotup = (image_uuid_list, verts_list, theta_list, viewpoint_list,
                                    name_list, species_list)
                return semantic_infotup
            visual_infotup = pre_1_3_1_get_annot_visual_uuid_info(ibs, aid_list)
            annot_visual_uuid_list = preproc_annot.make_annot_visual_uuid(visual_infotup)
            ibs.db.set(const.ANNOTATION_TABLE, (ANNOT_VISUAL_UUID,), annot_visual_uuid_list, aid_list)
            # If visual uuids are changes semantic ones are also changed
            # update semeantic pre 1_3_1
            _visual_infotup = visual_infotup
            semantic_infotup = pre_1_3_1_get_annot_semantic_uuid_info(ibs, aid_list, _visual_infotup)
            annot_semantic_uuid_list = preproc_annot.make_annot_semantic_uuid(semantic_infotup)
            ibs.db.set(const.ANNOTATION_TABLE, (ANNOT_SEMANTIC_UUID,), annot_semantic_uuid_list, aid_list)
            pass
        pre_1_3_1_update_visual_uuids(ibs, aid_list)
        #ibsfuncs.fix_remove_visual_dupliate_annotations(ibs)
        aid_list = ibs.get_valid_aids()
        visual_uuid_list = ibs.get_annot_visual_uuids(aid_list)
        ibs_dup_annots = ut.debug_duplicate_items(visual_uuid_list)
        dupaids_list = []
        if len(ibs_dup_annots):
            for key, dupxs in six.iteritems(ibs_dup_annots):
                aids = ut.list_take(aid_list, dupxs)
                dupaids_list.append(aids[1:])
            toremove_aids = ut.flatten(dupaids_list)
            print('About to delete toremove_aids=%r' % (toremove_aids,))
            #if ut.are_you_sure():
            #ibs.delete_annots(toremove_aids)
            #from ibeis.model.preproc import preproc_annot
            #preproc_annot.on_delete(ibs, toremove_aids)
            ibs.db.delete_rowids(const.ANNOTATION_TABLE, toremove_aids)

            aid_list = ibs.get_valid_aids()
            visual_uuid_list = ibs.get_annot_visual_uuids(aid_list)
            ibs_dup_annots = ut.debug_duplicate_items(visual_uuid_list)
            assert len(ibs_dup_annots) == 0


# =======================
# Schema Version 1.0.1
# =======================


@profile
def update_1_0_1(db, ibs=None):
    # Add a contributor's table
    db.add_table(const.CONTRIBUTOR_TABLE, (
        ('contributor_rowid',            'INTEGER PRIMARY KEY'),
        ('contributor_tag',              'TEXT'),
        ('contributor_name_first',       'TEXT'),
        ('contributor_name_last',        'TEXT'),
        ('contributor_location_city',    'TEXT'),
        ('contributor_location_state',   'TEXT'),
        ('contributor_location_country', 'TEXT'),
        ('contributor_location_zip',     'INTEGER'),
        ('contributor_note',             'INTEGER'),
    ),
        superkey_colnames_list=[('contributor_rowid',)],
        docstr='''
        Used to store the contributors to the project
        ''')

    db.modify_table(const.IMAGE_TABLE, (
        # add column to v1.0.0 at index 1
        (1, 'contributor_rowid', 'INTEGER', None),
    ))

    db.modify_table(const.ANNOTATION_TABLE, (
        # add column to v1.0.0 at index 1
        (1, 'annot_parent_rowid', 'INTEGER', None),
    ))

    db.modify_table(const.FEATURE_TABLE, (
        # append column to v1.0.0 because None
        (None, 'feature_weight', 'REAL DEFAULT 1.0', None),
    ))


# =======================
# Schema Version 1.0.2
# =======================


@profile
def update_1_0_2(db, ibs=None):
    # Fix the contibutor table's constraint
    db.modify_table(const.CONTRIBUTOR_TABLE, (
        # add column to v1.0.1 at index 1
        (1, 'contributor_uuid', 'UUID NOT NULL', None),
    ),
        table_constraints=[],
        superkey_colnames_list=[('contributor_tag',)]
    )

# =======================
# Schema Version 1.1.0
# =======================


@profile
def update_1_1_0(db, ibs=None):
    # Moving chips and features to their own cache database
    db.drop_table(const.CHIP_TABLE)
    db.drop_table(const.FEATURE_TABLE)

    # Add viewpoint (radians) to annotations
    db.modify_table(const.ANNOTATION_TABLE, (
        # add column to v1.0.2 at index 11
        (11, ANNOT_VIEWPOINT, 'REAL DEFAULT 0.0', None),
    ))

    # Add contributor to configs
    db.modify_table(const.CONFIG_TABLE, (
        # add column to v1.0.2 at index 1
        (1, 'contributor_uuid', 'UUID', None),
    ),
        table_constraints=[],
        # FIXME: This change may have broken things
        superkey_colnames_list=[('contributor_uuid', 'config_suffix',)]
    )

    # Add config to encounters
    db.modify_table(const.ENCOUNTER_TABLE, (
        # add column to v1.0.2 at index 2
        (2, 'config_rowid', 'INTEGER', None),
    ),
        table_constraints=[],
        superkey_colnames_list=[('encounter_uuid', 'encounter_text',)]
    )

    # Error in the drop table script, re-drop again from post_1_0_0 to kill table's metadata
    db.drop_table(const.VERSIONS_TABLE)


# =======================
# Schema Version 1.1.1
# =======================


@profile
def update_1_1_1(db, ibs=None):
    # Change name of column
    db.modify_table(const.CONFIG_TABLE, (
        # rename column and change it's type
        ('contributor_uuid', 'contributor_rowid', '', None),
    ),
        table_constraints=[],
        superkey_colnames_list=[('contributor_rowid', 'config_suffix',)]
    )

    # Change type of column
    db.modify_table(const.CONFIG_TABLE, (
        # rename column and change it's type
        ('contributor_rowid', '', 'INTEGER', None),
    ))

    # Change type of columns
    db.modify_table(const.CONTRIBUTOR_TABLE, (
        # Update column's types
        ('contributor_location_zip', '', 'TEXT', None),
        ('contributor_note', '', 'TEXT', None),
    ))


@profile
def update_1_2_0(db, ibs=None):
    # Add columns to annotaiton table
    tablename = const.ANNOTATION_TABLE
    colmap_list = (
        # the visual uuid will be unique w.r.t. the appearence of the annotation
        (None, ANNOT_VISUAL_UUID, 'UUID', None),
        # the visual uuid will be unique w.r.t. the appearence, name, and species of the annotation
        (None, 'annot_semantic_uuid', 'UUID', None),
        (None, 'name_rowid',    'INTEGER DEFAULT 0', None),
        (None, 'species_rowid', 'INTEGER DEFAULT 0', None),
    )
    aid_before = db.get_all_rowids(const.ANNOTATION_TABLE)
    #import utool as ut
    db.modify_table(tablename, colmap_list)
    # Sanity check
    aid_after = db.get_all_rowids(const.ANNOTATION_TABLE)
    assert aid_before == aid_after


@profile
def update_1_2_1(db, ibs=None):
    # Names and species are taken away from lblannot table and upgraded
    # to their own thing
    db.add_table(NAME_TABLE_v121, (
        ('name_rowid',               'INTEGER PRIMARY KEY'),
        ('name_uuid',                'UUID NOT NULL'),
        ('name_text',                'TEXT NOT NULL'),
        ('name_note',                'TEXT'),
    ),
        superkey_colnames_list=[('name_text',)],
        docstr='''
        Stores the individual animal names
        ''')

    db.add_table(const.SPECIES_TABLE, (
        ('species_rowid',               'INTEGER PRIMARY KEY'),
        ('species_uuid',                'UUID NOT NULL'),
        ('species_text',                'TEXT NOT NULL'),
        ('species_note',                'TEXT'),
    ),
        superkey_colnames_list=[('species_text',)],
        docstr='''
        Stores the different animal species
        ''')


def update_1_3_0(db, ibs=None):
    db.modify_table(const.IMAGE_TABLE, (
        (None, 'image_timedelta_posix', 'INTEGER DEFAULT 0', None),
    ))

    db.modify_table(NAME_TABLE_v121, (
        (None, 'name_temp_flag',  'INTEGER DEFAULT 0', None),
        (None, 'name_alias_text', 'TEXT',              None),
    ), tablename_new=NAME_TABLE_v130)

    db.drop_table(NAME_TABLE_v121)

    db.modify_table(const.ENCOUNTER_TABLE, (
        (None, 'encounter_start_time_posix', 'INTEGER',           None),
        (None, 'encounter_end_time_posix',   'INTEGER',           None),
        (None, 'encounter_gps_lat',          'INTEGER',           None),
        (None, 'encounter_gps_lon',          'INTEGER',           None),
        (None, 'encounter_processed_flag',   'INTEGER DEFAULT 0', None),
        (None, 'encounter_shipped_flag',     'INTEGER DEFAULT 0', None),
    ))

    """
    * New Image Columns
        - image_posix_timedelta

    * New Name Columns
        - name_temp_flag
        - name_alias_text

        - name_uuid
        - name_visual_uuid
        - name_member_annot_rowids_evalstr
        - name_member_num_annot_rowids

    * New Encounter Columns
        - encounter_start_posix_time
        - encounter_end_time_posix
        - encounter_gps_lat
        - encounter_gps_lon
        - encounter_processed_flag
        - encounter_shipped_flag
    """
    # TODO: changed shipped to commited
    # encounter_detected_flag
    # encounter_identified_flag???

    pass
    # Need encounter processed and shipped flag
    #db.modify_table(const.CONFIG_TABLE, (
    #    # rename column and change it's type
    #)


def update_1_3_1(db, ibs=None):
    """
    update the visual_uuid to be a superkey by adding a constraint
    """
    # make annot_visual_uuid not null and add it as a superkey
    db.modify_table(
        const.ANNOTATION_TABLE,
        [
            # change type of annot_visual_uuid
            (ANNOT_VISUAL_UUID, '', 'UUID NOT NULL', None),
        ],
        superkey_colnames_list=[(ANNOT_UUID,), (ANNOT_VISUAL_UUID,)])
    #pass


def update_1_3_2(db, ibs=None):
    """
    for SMART DATA
    """
    db.modify_table(const.ENCOUNTER_TABLE, (
        (None, 'encounter_smart_xml_fpath',   'TEXT',           None),
        (None, 'encounter_smart_waypoint_id', 'INTEGER',        None),
    ))


def update_1_3_3(db, ibs=None):
    # we should only be storing names here not paths
    db.modify_table(const.ENCOUNTER_TABLE, (
        ('encounter_smart_xml_fpath', 'encounter_smart_xml_fname',  'TEXT',           None),
    ))


def update_1_3_4(db, ibs=None):
    # OLD ANNOT VIEWPOINT FUNCS. Hopefully these are not needed
    #def get_annot_viewpoints(ibs, aid_list):
    #    viewpoint_list = ibs.db.get(const.ANNOTATION_TABLE, (ANNOT_VIEWPOINT,), aid_list)
    #    viewpoint_list = [viewpoint if viewpoint >= 0.0 else None for viewpoint in viewpoint_list]
    #    return viewpoint_list
    #def set_annot_viewpoint(ibs, aid_list, viewpoint_list, input_is_degrees=False):
    #    id_iter = ((aid,) for aid in aid_list)
    #    #viewpoint_list = [-1 if viewpoint is None else viewpoint for viewpoint in viewpoint_list]
    #    if input_is_degrees:
    #        viewpoint_list = [-1 if viewpoint is None else ut.deg_to_rad(viewpoint)
    #                          for viewpoint in viewpoint_list]
    #    #assert all([0.0 <= viewpoint < 2 * np.pi or viewpoint == -1.0 for viewpoint in viewpoint_list])
    #    val_iter = ((viewpoint, ) for viewpoint in viewpoint_list)
    #    ibs.db.set(const.ANNOTATION_TABLE, (ANNOT_VIEWPOINT,), val_iter, id_iter)
    #    ibs.update_annot_visual_uuids(aid_list)
    print('executing update_1_3_4')
    TAU = const.TAU

    def convert_old_viewpoint_to_yaw(angle):
        """ we initially had viewpoint coordinates inverted

        Example:
            >>> import math
            >>> TAU = 2 * math.pi
            >>> old_viewpoint_labels = [
            >>>     ('left'       , 0.000 * TAU,),
            >>>     ('frontleft'  , 0.125 * TAU,),
            >>>     ('front'      , 0.250 * TAU,),
            >>>     ('frontright' , 0.375 * TAU,),
            >>>     ('right'      , 0.500 * TAU,),
            >>>     ('backright'  , 0.625 * TAU,),
            >>>     ('back'       , 0.750 * TAU,),
            >>>     ('backleft'   , 0.875 * TAU,),
            >>> ]
            >>> for lbl, angle in old_viewpoint_labels:
            >>>     yaw = convert_old_viewpoint_to_yaw(angle)
            >>>     angle2 = convert_old_viewpoint_to_yaw(yaw)
            >>>     print('old %15r %.2f -> new %15r %.2f' % (lbl, angle, lbl, yaw))
            >>>     print('old %15r %.2f -> new %15r %.2f' % (lbl, yaw, lbl, angle2))
        """
        if angle is None:
            return None
        yaw = (-angle + (TAU / 2)) % TAU
        return yaw

    from ibeis.control import SQLDatabaseControl
    assert isinstance(db,  SQLDatabaseControl.SQLDatabaseController)

    db.modify_table(const.IMAGE_TABLE, (
        # Add original image path to image table for more data persistance and
        # stewardship
        (None, 'image_original_path',          'TEXT', None),
        # Add image location as a simple workaround for not using the gps
        (None, 'image_location_code',          'TEXT', None),
    ))
    db.modify_table(const.ANNOTATION_TABLE, (
        # Add image quality as an integer to filter the database more easilly
        (None, 'annot_quality',          'INTEGER', None),
        # Add a path to a file that will represent if a pixel belongs to the
        # object of interest within the annotation.
        #(None, 'annot_mask_fpath',       'STRING', None),
        (ANNOT_VIEWPOINT, ANNOT_YAW,  'REAL', convert_old_viewpoint_to_yaw),
    ))


def update_1_3_5(db, ibs=None):
    """ expand datasets to use new quality measures """
    if ibs is not None:
        aid_list = ibs.get_valid_aids()
        qual_list = ibs.get_annot_qualities(aid_list)
        assert len(qual_list) == 0 or max(qual_list) < 3, 'there were no qualities higher than 3 at this point'
        old_to_new = {
            2: 3,
            1: 2,
        }
        new_qual_list = [old_to_new.get(qual, qual) for qual in qual_list]
        ibs.set_annot_qualities(aid_list, new_qual_list)
    # Adds a few different degrees of quality
    pass

# ========================
# Valid Versions & Mapping
# ========================

# TODO: do we save a backup with the older version number in the file name?


base = const.BASE_DATABASE_VERSION
VALID_VERSIONS = utool.odict([
    #version:   (Pre-Update Function,  Update Function,    Post-Update Function)
    (base   ,    (None,                 None,               None                )),
    ('1.0.0',    (None,                 update_1_0_0,       post_1_0_0          )),
    ('1.0.1',    (None,                 update_1_0_1,       None                )),
    ('1.0.2',    (None,                 update_1_0_2,       None                )),
    ('1.1.0',    (None,                 update_1_1_0,       None                )),
    ('1.1.1',    (None,                 update_1_1_1,       None                )),
    ('1.2.0',    (None,                 update_1_2_0,       post_1_2_0          )),
    ('1.2.1',    (None,                 update_1_2_1,       post_1_2_1          )),
    ('1.3.0',    (None,                 update_1_3_0,       None                )),
    ('1.3.1',    (pre_1_3_1,            update_1_3_1,       None                )),
    ('1.3.2',    (None,                 update_1_3_2,       None                )),
    ('1.3.3',    (None,                 update_1_3_3,       None                )),
    ('1.3.4',    (None,                 update_1_3_4,       post_1_3_4          )),
    ('1.3.5',    (None,                 update_1_3_5,       None          )),
])


def test_db_schema():
    """
    test_db_schema

    CommandLine:
        python -m ibeis.control.DB_SCHEMA --test-test_db_schema
        python -m ibeis.control.DB_SCHEMA --test-test_db_schema -n=-1
        python -m ibeis.control.DB_SCHEMA --test-test_db_schema -n=0
        python -m ibeis.control.DB_SCHEMA --test-test_db_schema -n=1
        python -m ibeis.control.DB_SCHEMA --force-incremental-db-update
        python -m ibeis.control.DB_SCHEMA --test-test_db_schema --dump-autogen-schema
        python -m ibeis.control.DB_SCHEMA --test-test_db_schema --force-incremental-db-update --dump-autogen-schema
        python -m ibeis.control.DB_SCHEMA --test-test_db_schema --force-incremental-db-update


    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.DB_SCHEMA import *  # NOQA
        >>> test_db_schema()
    """
    from ibeis.control import DB_SCHEMA
    from ibeis.control import _sql_helpers
    from ibeis import params
    autogenerate = params.args.dump_autogen_schema
    n = utool.get_argval('-n', int, default=-1)
    db = _sql_helpers.get_nth_test_schema_version(DB_SCHEMA, n=n, autogenerate=autogenerate)
    autogen_cmd = 'python -m ibeis.control.DB_SCHEMA --test-test_db_schema --force-incremental-db-update --dump-autogen-schema'
    autogen_str = db.get_schema_current_autogeneration_str(autogen_cmd)
    print(autogen_str)
    print(' Run with --dump-autogen-schema to autogenerate latest schema version')


if __name__ == '__main__':
    """
    python -m ibeis.model.preproc.preproc_chip
    python -m ibeis.control.DB_SCHEMA --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()
    import utool as ut
    ut.doctest_funcs()
