# -*- coding: utf-8 -*-
"""
Module Licence and docstring

TODO: ideally the wbia.constants module would not be used here
and each function would use its own constant variables that are suffixed
with the last version number that they existed in

TODO:
    Add a table for original_image_path
    Add column for image exif orientation


CommandLine:
    python -m wbia.control.DB_SCHEMA --test-autogen_db_schema
"""
from __future__ import absolute_import, division, print_function
from wbia import constants as const
from wbia.control import _sql_helpers
import utool as ut

(print, rrr, profile) = ut.inject2(__name__)


try:
    from wbia.control import DB_SCHEMA_CURRENT

    UPDATE_CURRENT = DB_SCHEMA_CURRENT.update_current
    VERSION_CURRENT = DB_SCHEMA_CURRENT.VERSION_CURRENT
except Exception:
    UPDATE_CURRENT = None
    VERSION_CURRENT = None
    print('[dbcache] NO DB_SCHEMA_CURRENT AUTO-GENERATED!')

profile = ut.profile


ANNOTMATCH_TABLE = 'annotmatch'
NAME_TABLE_v121 = const.NAME_TABLE_v121
NAME_TABLE_v130 = const.NAME_TABLE_v130
ANNOT_VISUAL_UUID = 'annot_visual_uuid'
ANNOT_SEMANTIC_UUID = 'annot_semantic_uuid'
ANNOT_UUID = 'annot_uuid'
ANNOT_STAGED_UUID = 'annot_staged_uuid'
ANNOT_YAW = 'annot_yaw'
ANNOT_VIEWPOINT = 'annot_viewpoint'
PART_ROWID = 'part_rowid'
PART_UUID = 'part_uuid'
PART_STAGED_UUID = 'part_staged_uuid'
NAME_ROWID = 'name_rowid'
SPECIES_ROWID = 'species_rowid'
IMAGE_ROWID = 'image_rowid'
ANNOT_ROWID = 'annot_rowid'
ANNOT_PARENT_ROWID = 'annot_parent_rowid'
ANNOTGROUP_ROWID = 'annotgroup_rowid'
NAME_TEXT = 'name_text'
SPECIES_TEXT = 'species_text'
ANNOT_ROWID1 = 'annot_rowid1'
ANNOT_ROWID2 = 'annot_rowid2'
PARTY_ROWID = 'party_rowid'
PARTY_TAG = 'party_tag'
CONFIG_ROWID = 'config_rowid'
IMAGE_UUID = 'image_uuid'
CONFIG_SUFFIX = 'config_suffix'

# =======================
# Schema Version 1.0.0
# =======================


@profile
def update_1_0_0(db, ibs=None):
    db.add_table(
        const.IMAGE_TABLE,
        (
            (IMAGE_ROWID, 'INTEGER PRIMARY KEY'),
            (IMAGE_UUID, 'UUID NOT NULL'),
            ('image_uri', 'TEXT NOT NULL'),
            ('image_ext', 'TEXT NOT NULL'),
            (
                'image_original_name',
                'TEXT NOT NULL',
            ),  # We could parse this out of original_path
            ('image_width', 'INTEGER DEFAULT -1'),
            ('image_height', 'INTEGER DEFAULT -1'),
            ('image_time_posix', 'INTEGER DEFAULT -1'),  # this should probably be UCT
            (
                'image_gps_lat',
                'REAL DEFAULT -1.0',
            ),  # there doesn't seem to exist a GPSPoint in SQLite (TODO: make one in the __SQLITE3__ custom types
            ('image_gps_lon', 'REAL DEFAULT -1.0'),
            ('image_toggle_enabled', 'INTEGER DEFAULT 0'),
            ('image_toggle_reviewed', 'INTEGER DEFAULT 0'),
            ('image_note', 'TEXT',),
        ),
        superkeys=[(IMAGE_UUID,)],
        docstr="""
        First class table used to store image locations and meta-data""",
    )

    db.add_table(
        'encounters',
        (
            ('encounter_rowid', 'INTEGER PRIMARY KEY'),
            ('encounter_uuid', 'UUID NOT NULL'),
            ('encounter_text', 'TEXT NOT NULL'),
            ('encounter_note', 'TEXT NOT NULL'),
        ),
        superkeys=[('encounter_text',)],
        docstr="""
        List of all encounters""",
    )

    db.add_table(
        const.LBLTYPE_TABLE,
        (
            ('lbltype_rowid', 'INTEGER PRIMARY KEY'),
            ('lbltype_text', 'TEXT NOT NULL'),
            ('lbltype_default', 'TEXT NOT NULL'),
        ),
        superkeys=[('lbltype_text',)],
        docstr="""
        List of keys used to define the categories of annotation lables, text
        is for human-readability. The lbltype_default specifies the
        lblannot_value of annotations with a relationship of some
        lbltype_rowid""",
    )

    db.add_table(
        const.CONFIG_TABLE,
        ((CONFIG_ROWID, 'INTEGER PRIMARY KEY'), (CONFIG_SUFFIX, 'TEXT NOT NULL'),),
        superkeys=[(CONFIG_SUFFIX,)],
        docstr="""
        Used to store the ids of algorithm configurations that generate
        annotation lblannots.  Each user will have a config id for manual
        contributions """,
    )

    ##########################
    # FIRST ORDER            #
    ##########################
    db.add_table(
        const.ANNOTATION_TABLE,
        (
            (ANNOT_ROWID, 'INTEGER PRIMARY KEY'),
            (ANNOT_UUID, 'UUID NOT NULL'),
            ('image_rowid', 'INTEGER NOT NULL'),
            ('annot_xtl', 'INTEGER NOT NULL'),
            ('annot_ytl', 'INTEGER NOT NULL'),
            ('annot_width', 'INTEGER NOT NULL'),
            ('annot_height', 'INTEGER NOT NULL'),
            ('annot_theta', 'REAL DEFAULT 0.0'),
            ('annot_num_verts', 'INTEGER NOT NULL'),
            ('annot_verts', 'TEXT'),
            ('annot_detect_confidence', 'REAL DEFAULT -1.0'),
            ('annot_exemplar_flag', 'INTEGER DEFAULT 0'),
            ('annot_note', 'TEXT'),
        ),
        superkeys=[(ANNOT_UUID,)],
        docstr="""
        Mainly used to store the geometry of the annotation within its parent
        image The one-to-many relationship between images and annotations is
        encoded here Attributes are stored in the Annotation Label Relationship
        Table""",
    )

    db.add_table(
        const.LBLIMAGE_TABLE,
        (
            ('lblimage_rowid', 'INTEGER PRIMARY KEY'),
            ('lblimage_uuid', 'UUID NOT NULL'),
            ('lbltype_rowid', 'INTEGER NOT NULL'),  # this is "category" in the proposal
            ('lblimage_value', 'TEXT NOT NULL'),
            ('lblimage_note', 'TEXT'),
        ),
        superkeys=[('lbltype_rowid', 'lblimage_value',)],
        docstr="""
        Used to store the labels (attributes) of images""",
    )

    db.add_table(
        const.LBLANNOT_TABLE,
        (
            ('lblannot_rowid', 'INTEGER PRIMARY KEY'),
            ('lblannot_uuid', 'UUID NOT NULL'),
            ('lbltype_rowid', 'INTEGER NOT NULL'),  # this is "category" in the proposal
            ('lblannot_value', 'TEXT NOT NULL'),
            ('lblannot_note', 'TEXT'),
        ),
        superkeys=[('lbltype_rowid', 'lblannot_value',)],
        docstr="""
        Used to store the labels / attributes of annotations.
        E.G name, species """,
    )

    ##########################
    # SECOND ORDER           #
    ##########################
    # TODO: constraint needs modification
    db.add_table(
        const.CHIP_TABLE,
        (
            ('chip_rowid', 'INTEGER PRIMARY KEY'),
            ('annot_rowid', 'INTEGER NOT NULL'),
            ('config_rowid', 'INTEGER DEFAULT 0'),
            ('chip_uri', 'TEXT'),
            ('chip_width', 'INTEGER NOT NULL'),
            ('chip_height', 'INTEGER NOT NULL'),
        ),
        superkeys=[('annot_rowid', 'config_rowid',)],
        docstr="""
        Used to store *processed* annots as chips""",
    )

    db.add_table(
        const.FEATURE_TABLE,
        (
            ('feature_rowid', 'INTEGER PRIMARY KEY'),
            ('chip_rowid', 'INTEGER NOT NULL'),
            ('config_rowid', 'INTEGER DEFAULT 0'),
            ('feature_num_feats', 'INTEGER NOT NULL'),
            ('feature_keypoints', 'NUMPY'),
            ('feature_sifts', 'NUMPY'),
        ),
        superkeys=[('chip_rowid, config_rowid',)],
        docstr="""
        Used to store individual chip features (ellipses)""",
    )

    db.add_table(
        'encounter_image_relationship',
        (
            ('egr_rowid', 'INTEGER PRIMARY KEY'),
            ('image_rowid', 'INTEGER NOT NULL'),
            ('encounter_rowid', 'INTEGER'),
        ),
        superkeys=[(IMAGE_ROWID, 'encounter_rowid',)],
        docstr="""
        Relationship between encounters and images (many to many mapping) the
        many-to-many relationship between images and encounters is encoded here
        encounter_image_relationship stands for encounter-image-pairs.""",
    )

    ##########################
    # THIRD ORDER            #
    ##########################
    db.add_table(
        const.GL_RELATION_TABLE,
        (
            ('glr_rowid', 'INTEGER PRIMARY KEY'),
            ('image_rowid', 'INTEGER NOT NULL'),
            ('lblimage_rowid', 'INTEGER NOT NULL'),
            ('config_rowid', 'INTEGER DEFAULT 0'),
            ('glr_confidence', 'REAL DEFAULT 0.0'),
        ),
        superkeys=[('image_rowid', 'lblimage_rowid', 'config_rowid',)],
        docstr="""
        Used to store one-to-many the relationship between images
        and labels""",
    )

    db.add_table(
        const.AL_RELATION_TABLE,
        (
            ('alr_rowid', 'INTEGER PRIMARY KEY'),
            ('annot_rowid', 'INTEGER NOT NULL'),
            ('lblannot_rowid', 'INTEGER NOT NULL'),
            ('config_rowid', 'INTEGER DEFAULT 0'),
            ('alr_confidence', 'REAL DEFAULT 0.0'),
        ),
        superkeys=[('annot_rowid', 'lblannot_rowid', 'config_rowid',)],
        docstr="""
        Used to store one-to-many the relationship between annotations (annots)
        and labels""",
    )


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
            >>> import wbia
            >>> #import sys
            #>>> sys.argv.append('--force-fresh')
            #>>> ibs = wbia.opendb('PZ_MTEST')
            #>>> ibs = wbia.opendb('testdb1')
            >>> ibs = wbia.opendb('GZ_ALL')
            >>> # should be auto applied
            >>> ibs.print_annotation_table(verbosity=1)
            >>> result = schema_1_2_0_postprocess_fixuuids(ibs)
            >>> ibs.print_annotation_table(verbosity=1)
        """
        import utool as ut

        aid_list = ibs.get_valid_aids(is_staged=None)
        # ibs.get_annot_name_rowids(aid_list)
        #
        # ANNOT_ROWID             = 'annot_rowid'
        ANNOTATION_TABLE = 'annotations'
        NAME_ROWID = 'name_rowid'
        SPECIES_ROWID = 'species_rowid'
        AL_RELATION_TABLE = 'annotation_lblannot_relationship'

        def set_annot_species_rowids(ibs, aid_list, species_rowid_list):
            id_iter = aid_list
            colnames = (SPECIES_ROWID,)
            ibs.db.set(ANNOTATION_TABLE, colnames, species_rowid_list, id_iter)

        def set_annot_name_rowids(ibs, aid_list, name_rowid_list):
            id_iter = aid_list
            colnames = (NAME_ROWID,)
            ibs.db.set(ANNOTATION_TABLE, colnames, name_rowid_list, id_iter)

        def get_alr_lblannot_rowids(alrid_list):
            lblannot_rowids_list = ibs.db.get(
                AL_RELATION_TABLE, ('lblannot_rowid',), alrid_list
            )
            return lblannot_rowids_list

        def get_annot_alrids_oftype(aid_list, lbltype_rowid):
            """
            Get all the relationship ids belonging to the input annotations where the
            relationship ids are filtered to be only of a specific lbltype/category/type
            """
            alrids_list = ibs.db.get(
                AL_RELATION_TABLE,
                ('alr_rowid',),
                aid_list,
                id_colname='annot_rowid',
                unpack_scalars=False,
            )
            # Get lblannot_rowid of each relationship
            lblannot_rowids_list = ibs.unflat_map(get_alr_lblannot_rowids, alrids_list)
            # Get the type of each lblannot
            lbltype_rowids_list = ibs.unflat_map(
                ibs.get_lblannot_lbltypes_rowids, lblannot_rowids_list
            )
            # only want the nids of individuals, not species, for example
            valids_list = [
                [typeid == lbltype_rowid for typeid in rowids]
                for rowids in lbltype_rowids_list
            ]
            alrids_list = [
                ut.compress(alrids, valids)
                for alrids, valids in zip(alrids_list, valids_list)
            ]
            alrids_list = [
                alrid_list[0:1] if len(alrid_list) > 1 else alrid_list
                for alrid_list in alrids_list
            ]
            assert all([len(alrid_list) < 2 for alrid_list in alrids_list]), (
                'More than one type per lbltype.  ALRIDS: '
                + str(alrids_list)
                + ', ROW: '
                + str(lbltype_rowid)
                + ', KEYS:'
                + str(ibs.lbltype_ids)
            )
            return alrids_list

        def get_annot_speciesid_from_lblannot_relation(
            aid_list, distinguish_unknowns=True
        ):
            """ function for getting speciesid the old way """
            species_lbltype_rowid = ibs.db.get(
                'keys', ('lbltype_rowid',), ('SPECIES_KEY',), id_colname='lbltype_text'
            )[0]
            alrids_list = get_annot_alrids_oftype(aid_list, species_lbltype_rowid)
            lblannot_rowids_list = ibs.unflat_map(get_alr_lblannot_rowids, alrids_list)
            speciesid_list = [
                lblannot_rowids[0]
                if len(lblannot_rowids) > 0
                else const.UNKNOWN_LBLANNOT_ROWID
                for lblannot_rowids in lblannot_rowids_list
            ]
            return speciesid_list

        def get_annot_name_rowids_from_lblannot_relation(aid_list):
            """ function for getting nids the old way """
            individual_lbltype_rowid = ibs.db.get(
                'keys',
                ('lbltype_rowid',),
                ('INDIVIDUAL_KEY',),
                id_colname='lbltype_text',
            )[0]
            alrids_list = get_annot_alrids_oftype(aid_list, individual_lbltype_rowid)
            lblannot_rowids_list = ibs.unflat_map(get_alr_lblannot_rowids, alrids_list)
            # Get a single nid from the list of lblannot_rowids of type INDIVIDUAL
            # TODO: get index of highest confidencename
            nid_list = [
                lblannot_rowids[0]
                if len(lblannot_rowids) > 0
                else const.UNKNOWN_LBLANNOT_ROWID
                for lblannot_rowids in lblannot_rowids_list
            ]
            return nid_list

        nid_list1 = ibs.get_annot_name_rowids(aid_list, distinguish_unknowns=False)
        speciesid_list1 = ibs.get_annot_species_rowids(aid_list)

        # Get old values from lblannot table
        nid_list = get_annot_name_rowids_from_lblannot_relation(aid_list)
        speciesid_list = get_annot_speciesid_from_lblannot_relation(aid_list)

        assert len(nid_list1) == len(nid_list), 'cannot update to 1_2_0 name length error'
        assert len(speciesid_list1) == len(
            speciesid_list
        ), 'cannot update to 1_2_0 species length error'

        if ut.list_all_eq_to(nid_list, 0) and ut.list_all_eq_to(speciesid_list, 0):
            print('... returning No information in lblannot table to transfer')
            return

        # Make sure information has not gotten out of sync
        try:
            assert all(
                [(nid1 == nid or nid1 == 0) for nid1, nid in zip(nid_list1, nid_list)]
            )
            assert all(
                [
                    (sid1 == sid or sid1 == 0)
                    for sid1, sid in zip(speciesid_list1, speciesid_list)
                ]
            )
        except AssertionError as ex:
            ut.printex(ex, 'Cannot update database to 1_2_0 information out of sync')
            raise

        # Move values into the annotation table as a native column
        set_annot_name_rowids(ibs, aid_list, nid_list)
        set_annot_species_rowids(ibs, aid_list, speciesid_list)
        # Update visual uuids
        # Moved this to post_process 1.21
        # ibs.update_annot_visual_uuids(aid_list)
        # ibs.update_annot_semantic_uuids(aid_list)

    # ibs.print_annotation_table(verbosity=1)
    if ibs is not None:
        ibs._init_rowid_constants()
        schema_1_2_0_postprocess_fixuuids(ibs)
    else:
        print(
            'warning: ibs is None, so cannot apply name/species column fixes to existing database'
        )


def post_1_2_1(db, ibs=None):
    if ibs is not None:
        print('applying post_1_2_1')
        import utool as ut

        if ibs is not None:
            ibs._init_rowid_constants()
            # db = ibs.db
        UNKNOWN_ROWID = 0
        UNKNOWN = '____'
        UNKNOWN_NAME_ROWID = 0
        ANNOTATION_TABLE = 'annotations'
        SPECIES_TABLE = 'species'
        LBLANNOT_TABLE = 'lblannot'
        SPECIES_ROWID = 'species_rowid'
        NAME_ROWID = 'name_rowid'
        NAME_TEXT = 'name_text'
        ANNOT_SEMANTIC_UUID = 'annot_semantic_uuid'
        lblannot_colnames = (
            'lblannot_uuid',
            'lblannot_value',
            'lblannot_note',
        )
        name_colnames = (
            'name_uuid',
            'name_text',
            'name_note',
        )
        species_colspeciess = (
            'species_uuid',
            'species_text',
            'species_note',
        )
        # Get old name and species rowids from annotaiton tables
        aid_list = db.get_all_rowids(ANNOTATION_TABLE)
        name_rowids1 = db.get(ANNOTATION_TABLE, (NAME_ROWID,), aid_list)
        species_rowids1 = db.get(ANNOTATION_TABLE, (SPECIES_ROWID,), aid_list)
        # Look at the unique non-unknown ones
        unique_name_rowids1 = sorted(list(set(name_rowids1) - set([UNKNOWN_ROWID])))
        unique_species_rowids1 = sorted(list(set(species_rowids1) - set([UNKNOWN_ROWID])))
        # Get params out of label annotation tables
        name_params_list = db.get(LBLANNOT_TABLE, lblannot_colnames, unique_name_rowids1)
        species_params_list = db.get(
            LBLANNOT_TABLE, lblannot_colnames, unique_species_rowids1
        )
        # Move params into name and species tables
        unique_name_rowids2 = db._add(NAME_TABLE_v121, name_colnames, name_params_list)
        unique_species_rowids2 = db._add(
            SPECIES_TABLE, species_colspeciess, species_params_list
        )
        # Build mapping from old table to new table
        name_rowid_mapping = dict(zip(unique_name_rowids1, unique_name_rowids2))
        speices_rowid_mapping = dict(zip(unique_species_rowids1, unique_species_rowids2))
        name_rowid_mapping[UNKNOWN_ROWID] = UNKNOWN_ROWID
        speices_rowid_mapping[UNKNOWN_ROWID] = UNKNOWN_ROWID
        # Apply mapping
        name_rowids2 = ut.dict_take_list(name_rowid_mapping, name_rowids1)
        species_rowid2 = ut.dict_take_list(speices_rowid_mapping, species_rowids1)
        # Put new rowids back into annotation table
        db.set(ANNOTATION_TABLE, (NAME_ROWID,), name_rowids2, aid_list)
        db.set(ANNOTATION_TABLE, (SPECIES_ROWID,), species_rowid2, aid_list)
        # HACK TODO use actual SQL to fix and move to 1.2.0

        def get_annot_names_v121(aid_list):
            name_rowid_list = ibs.get_annot_name_rowids(aid_list)
            name_text_list = ibs.db.get(NAME_TABLE_v121, (NAME_TEXT,), name_rowid_list)
            name_text_list = [
                UNKNOWN if rowid == UNKNOWN_NAME_ROWID or name_text is None else name_text
                for name_text, rowid in zip(name_text_list, name_rowid_list)
            ]
            return name_text_list

        def get_annot_semantic_uuid_info_v121(aid_list, _visual_infotup):
            visual_infotup = _visual_infotup
            image_uuid_list, verts_list, theta_list = visual_infotup
            # It is visual info augmented with name and species

            def get_annot_viewpoints(ibs, aid_list):
                viewpoint_list = ibs.db.get(
                    const.ANNOTATION_TABLE, (ANNOT_VIEWPOINT,), aid_list
                )
                viewpoint_list = [
                    viewpoint if viewpoint is None or viewpoint >= 0.0 else None
                    for viewpoint in viewpoint_list
                ]
                return viewpoint_list

            view_list = get_annot_viewpoints(ibs, aid_list)
            name_list = get_annot_names_v121(aid_list)
            species_list = ibs.get_annot_species_texts(aid_list)
            semantic_infotup = (
                image_uuid_list,
                verts_list,
                theta_list,
                view_list,
                name_list,
                species_list,
            )
            return semantic_infotup

        def get_annot_visual_uuid_info_v121(aid_list):
            image_uuid_list = ibs.get_annot_image_uuids(aid_list)
            verts_list = ibs.get_annot_verts(aid_list)
            theta_list = ibs.get_annot_thetas(aid_list)
            visual_infotup = (image_uuid_list, verts_list, theta_list)
            return visual_infotup

        def update_annot_semantic_uuids_v121(aid_list, _visual_infotup=None):
            semantic_infotup = get_annot_semantic_uuid_info_v121(
                aid_list, _visual_infotup
            )
            assert len(semantic_infotup) == 6, 'len=%r' % (len(semantic_infotup),)
            annot_semantic_uuid_list = [
                ut.augment_uuid(*tup) for tup in zip(*semantic_infotup)
            ]
            ibs.db.set(
                ANNOTATION_TABLE,
                (ANNOT_SEMANTIC_UUID,),
                annot_semantic_uuid_list,
                aid_list,
            )

        def update_annot_visual_uuids_v121(aid_list):
            visual_infotup = get_annot_visual_uuid_info_v121(aid_list)
            assert len(visual_infotup) == 3, 'len=%r' % (len(visual_infotup),)
            annot_visual_uuid_list = [
                ut.augment_uuid(*tup) for tup in zip(*visual_infotup)
            ]
            ibs.db.set(
                ANNOTATION_TABLE, (ANNOT_VISUAL_UUID,), annot_visual_uuid_list, aid_list
            )
            # If visual uuids are changes semantic ones are also changed
            update_annot_semantic_uuids_v121(aid_list, visual_infotup)

        update_annot_visual_uuids_v121(aid_list)


def post_1_3_4(db, ibs=None):
    if ibs is not None:
        ibs._init_rowid_constants()
        ibs._init_config()
        # Move up because this has changed
        # ibs.update_annot_visual_uuids(ibs.get_valid_aids())


def pre_1_3_1(db, ibs=None):
    """
    need to ensure that visual uuid columns are unique before we add that
    constaint to sql. This will remove any annotations that are not unique
    """
    if ibs is not None:
        # from wbia.other import ibsfuncs
        import utool as ut
        import six

        ibs._init_rowid_constants()
        ibs._init_config()
        aid_list = ibs.get_valid_aids(is_staged=None)

        def pre_1_3_1_update_visual_uuids(ibs, aid_list):
            def pre_1_3_1_get_annot_visual_uuid_info(ibs, aid_list):
                image_uuid_list = ibs.get_annot_image_uuids(aid_list)
                verts_list = ibs.get_annot_verts(aid_list)
                theta_list = ibs.get_annot_thetas(aid_list)
                visual_infotup = (image_uuid_list, verts_list, theta_list)
                return visual_infotup

            def pre_1_3_1_get_annot_semantic_uuid_info(ibs, aid_list, _visual_infotup):
                image_uuid_list, verts_list, theta_list = visual_infotup

                def get_annot_viewpoints(ibs, aid_list):
                    viewpoint_list = ibs.db.get(
                        const.ANNOTATION_TABLE, (ANNOT_VIEWPOINT,), aid_list
                    )
                    viewpoint_list = [
                        viewpoint if viewpoint is None or viewpoint >= 0.0 else None
                        for viewpoint in viewpoint_list
                    ]
                    return viewpoint_list

                # It is visual info augmented with name and species
                viewpoint_list = get_annot_viewpoints(ibs, aid_list)
                name_list = ibs.get_annot_names(aid_list)
                species_list = ibs.get_annot_species_texts(aid_list)
                semantic_infotup = (
                    image_uuid_list,
                    verts_list,
                    theta_list,
                    viewpoint_list,
                    name_list,
                    species_list,
                )
                return semantic_infotup

            visual_infotup = pre_1_3_1_get_annot_visual_uuid_info(ibs, aid_list)
            assert len(visual_infotup) == 3, 'len=%r' % (len(visual_infotup),)
            annot_visual_uuid_list = [
                ut.augment_uuid(*tup) for tup in zip(*visual_infotup)
            ]
            ibs.db.set(
                const.ANNOTATION_TABLE,
                (ANNOT_VISUAL_UUID,),
                annot_visual_uuid_list,
                aid_list,
            )
            # If visual uuids are changes semantic ones are also changed
            # update semeantic pre 1_3_1
            _visual_infotup = visual_infotup
            semantic_infotup = pre_1_3_1_get_annot_semantic_uuid_info(
                ibs, aid_list, _visual_infotup
            )
            assert len(semantic_infotup) == 6, 'len=%r' % (len(semantic_infotup),)
            annot_semantic_uuid_list = [
                ut.augment_uuid(*tup) for tup in zip(*semantic_infotup)
            ]
            ibs.db.set(
                const.ANNOTATION_TABLE,
                (ANNOT_SEMANTIC_UUID,),
                annot_semantic_uuid_list,
                aid_list,
            )
            pass

        pre_1_3_1_update_visual_uuids(ibs, aid_list)
        # ibsfuncs.fix_remove_visual_dupliate_annotations(ibs)
        aid_list = ibs.get_valid_aids(is_staged=None)
        visual_uuid_list = ibs.get_annot_visual_uuids(aid_list)
        ibs_dup_annots = ut.debug_duplicate_items(visual_uuid_list)
        dupaids_list = []
        if len(ibs_dup_annots):
            for key, dupxs in six.iteritems(ibs_dup_annots):
                aids = ut.take(aid_list, dupxs)
                dupaids_list.append(aids[1:])
            toremove_aids = ut.flatten(dupaids_list)
            print('About to delete toremove_aids=%r' % (toremove_aids,))
            ibs.db.delete_rowids(const.ANNOTATION_TABLE, toremove_aids)

            aid_list = ibs.get_valid_aids(is_staged=None)
            visual_uuid_list = ibs.get_annot_visual_uuids(aid_list)
            ibs_dup_annots = ut.debug_duplicate_items(visual_uuid_list)
            assert len(ibs_dup_annots) == 0


# =======================
# Schema Version 1.0.1
# =======================


@profile
def update_1_0_1(db, ibs=None):
    # Add a contributor's table
    db.add_table(
        const.CONTRIBUTOR_TABLE,
        (
            ('contributor_rowid', 'INTEGER PRIMARY KEY'),
            ('contributor_tag', 'TEXT'),
            ('contributor_name_first', 'TEXT'),
            ('contributor_name_last', 'TEXT'),
            ('contributor_location_city', 'TEXT'),
            ('contributor_location_state', 'TEXT'),
            ('contributor_location_country', 'TEXT'),
            ('contributor_location_zip', 'INTEGER'),
            ('contributor_note', 'INTEGER'),
        ),
        superkeys=[('contributor_rowid',)],
        docstr="""
        Used to store the contributors to the project
        """,
    )

    db.modify_table(
        const.IMAGE_TABLE,
        (
            # add column to v1.0.0 at index 1
            (1, 'contributor_rowid', 'INTEGER', None),
        ),
    )

    db.modify_table(
        const.ANNOTATION_TABLE,
        (
            # add column to v1.0.0 at index 1
            (1, 'annot_parent_rowid', 'INTEGER', None),
        ),
    )

    db.modify_table(
        const.FEATURE_TABLE,
        (
            # append column to v1.0.0 because None
            (None, 'feature_weight', 'REAL DEFAULT 1.0', None),
        ),
    )


# =======================
# Schema Version 1.0.2
# =======================


@profile
def update_1_0_2(db, ibs=None):
    # Fix the contibutor table's constraint
    db.modify_table(
        const.CONTRIBUTOR_TABLE,
        (
            # add column to v1.0.1 at index 1
            (1, 'contributor_uuid', 'UUID NOT NULL', None),
        ),
        superkeys=[('contributor_tag',)],
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
    db.modify_table(
        const.ANNOTATION_TABLE,
        (
            # add column to v1.0.2 at index 11
            # (11, ANNOT_VIEWPOINT, 'REAL DEFAULT 0.0', None),
            (11, ANNOT_VIEWPOINT, 'REAL', None),
        ),
    )

    # Add contributor to configs
    db.modify_table(
        const.CONFIG_TABLE,
        (
            # add column to v1.0.2 at index 1
            (1, 'contributor_uuid', 'UUID', None),
        ),
        # FIXME: This change may have broken things
        superkeys=[('contributor_uuid', CONFIG_SUFFIX,)],
    )

    # Add config to encounters
    db.modify_table(
        'encounters',
        (
            # add column to v1.0.2 at index 2
            (2, 'config_rowid', 'INTEGER', None),
        ),
        superkeys=[('encounter_uuid', 'encounter_text',)],
    )

    # Error in the drop table script, re-drop again from post_1_0_0 to kill table's metadata
    db.drop_table(const.VERSIONS_TABLE)


# =======================
# Schema Version 1.1.1
# =======================


@profile
def update_1_1_1(db, ibs=None):
    # Change name of column
    db.modify_table(
        const.CONFIG_TABLE,
        (
            # rename column and change it's type
            ('contributor_uuid', 'contributor_rowid', 'INTEGER', None),
        ),
        superkeys=[('contributor_rowid', CONFIG_SUFFIX,)],
    )

    # Change type of column
    # db.modify_table(const.CONFIG_TABLE, (
    #    # rename column and change it's type
    #    ('contributor_rowid', '', 'INTEGER', None),
    # ))

    # Change type of columns
    db.modify_table(
        const.CONTRIBUTOR_TABLE,
        (
            # Update column's types
            ('contributor_location_zip', '', 'TEXT', None),
            ('contributor_note', '', 'TEXT', None),
        ),
    )


@profile
def update_1_2_0(db, ibs=None):
    # Add columns to annotaiton table
    tablename = const.ANNOTATION_TABLE
    colmap_list = (
        # the visual uuid will be unique w.r.t. the appearence of the annotation
        (None, ANNOT_VISUAL_UUID, 'UUID', None),
        # the visual uuid will be unique w.r.t. the appearence, name, and species of the annotation
        (None, 'annot_semantic_uuid', 'UUID', None),
        (None, 'name_rowid', 'INTEGER DEFAULT 0', None),
        (None, 'species_rowid', 'INTEGER DEFAULT 0', None),
    )
    aid_before = db.get_all_rowids(const.ANNOTATION_TABLE)
    # import utool as ut
    db.modify_table(tablename, colmap_list)
    # Sanity check
    aid_after = db.get_all_rowids(const.ANNOTATION_TABLE)
    assert aid_before == aid_after


@profile
def update_1_2_1(db, ibs=None):
    # Names and species are taken away from lblannot table and upgraded
    # to their own thing
    db.add_table(
        NAME_TABLE_v121,
        (
            ('name_rowid', 'INTEGER PRIMARY KEY'),
            ('name_uuid', 'UUID NOT NULL'),
            (NAME_TEXT, 'TEXT NOT NULL'),
            ('name_note', 'TEXT'),
        ),
        superkeys=[(NAME_TEXT,)],
        docstr="""
        Stores the individual animal names
        """,
    )

    db.add_table(
        const.SPECIES_TABLE,
        (
            ('species_rowid', 'INTEGER PRIMARY KEY'),
            ('species_uuid', 'UUID NOT NULL'),
            (SPECIES_TEXT, 'TEXT NOT NULL'),
            ('species_note', 'TEXT'),
        ),
        superkeys=[(SPECIES_TEXT,)],
        docstr="""
        Stores the different animal species
        """,
    )


def update_1_3_0(db, ibs=None):
    db.modify_table(
        const.IMAGE_TABLE, ((None, 'image_timedelta_posix', 'INTEGER DEFAULT 0', None),)
    )

    db.modify_table(
        NAME_TABLE_v121,
        (
            (None, 'name_temp_flag', 'INTEGER DEFAULT 0', None),
            (None, 'name_alias_text', 'TEXT', None),
        ),
        tablename_new=NAME_TABLE_v130,
    )

    db.drop_table(NAME_TABLE_v121)

    db.modify_table(
        'encounters',
        (
            (None, 'encounter_start_time_posix', 'INTEGER', None),
            (None, 'encounter_end_time_posix', 'INTEGER', None),
            (None, 'encounter_gps_lat', 'INTEGER', None),
            (None, 'encounter_gps_lon', 'INTEGER', None),
            (None, 'encounter_processed_flag', 'INTEGER DEFAULT 0', None),
            (None, 'encounter_shipped_flag', 'INTEGER DEFAULT 0', None),
        ),
    )

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
    # db.modify_table(const.CONFIG_TABLE, (
    #    # rename column and change it's type
    # )


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
        # ERROR: this should have been ANNOT_SEMANTIC_UUID
        superkeys=[(ANNOT_UUID,), (ANNOT_VISUAL_UUID,)],
    )
    # pass


def update_1_3_2(db, ibs=None):
    """
    for SMART DATA
    """
    db.modify_table(
        'encounters',
        (
            (None, 'encounter_smart_xml_fpath', 'TEXT', None),
            (None, 'encounter_smart_waypoint_id', 'INTEGER', None),
        ),
    )


def update_1_3_3(db, ibs=None):
    # we should only be storing names here not paths
    db.modify_table(
        'encounters',
        (('encounter_smart_xml_fpath', 'encounter_smart_xml_fname', 'TEXT', None),),
    )


def update_1_3_4(db, ibs=None):
    # OLD ANNOT VIEWPOINT FUNCS. Hopefully these are not needed
    # def get_annot_viewpoints(ibs, aid_list):
    #    viewpoint_list = ibs.db.get(const.ANNOTATION_TABLE, (ANNOT_VIEWPOINT,), aid_list)
    #    viewpoint_list = [viewpoint if viewpoint >= 0.0 else None for viewpoint in viewpoint_list]
    #    return viewpoint_list
    # def set_annot_viewpoint(ibs, aid_list, viewpoint_list, input_is_degrees=False):
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
            >>> # DISABLE_DOCTEST
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

    from wbia.dtool.sql_control import SQLDatabaseController

    assert isinstance(db, SQLDatabaseController)

    db.modify_table(
        const.IMAGE_TABLE,
        (
            # Add original image path to image table for more data persistance and
            # stewardship
            (None, 'image_original_path', 'TEXT', None),
            # Add image location as a simple workaround for not using the gps
            (None, 'image_location_code', 'TEXT', None),
        ),
    )
    db.modify_table(
        const.ANNOTATION_TABLE,
        (
            # Add image quality as an integer to filter the database more easilly
            (None, 'annot_quality', 'INTEGER', None),
            # Add a path to a file that will represent if a pixel belongs to the
            # object of interest within the annotation.
            # (None, 'annot_mask_fpath',       'STRING', None),
            (ANNOT_VIEWPOINT, ANNOT_YAW, 'REAL', convert_old_viewpoint_to_yaw),
        ),
    )


def update_1_3_5(db, ibs=None):
    """ expand datasets to use new quality measures """
    if ibs is not None:
        # Adds a few different degrees of quality
        aid_list = ibs.get_valid_aids(is_staged=None)
        qual_list = ibs.get_annot_qualities(aid_list)
        flags = [q is not None for q in qual_list]
        qual_list = ut.compress(qual_list, flags)
        aid_list = ut.compress(aid_list, flags)
        assert (
            len(qual_list) == 0 or max(qual_list) < 3
        ), 'there were no qualities higher than 3 at this point'
        old_to_new = {
            2: 3,
            1: 2,
        }
        new_qual_list = [old_to_new.get(qual, qual) for qual in qual_list]
        ibs.set_annot_qualities(aid_list, new_qual_list)


def update_1_3_6(db, ibs=None):
    # Add table for explicit annotation-vs-annotation match information

    db.add_table(
        const.PARTY_TABLE,
        ((PARTY_ROWID, 'INTEGER PRIMARY KEY'), (PARTY_TAG, 'TEXT NOT NULL'),),
        superkeys=[(PARTY_TAG,)],
        docstr="""
        Serves as a group for contributors
        """,
    )

    # instead of adding a specific many to many mapping relate images to both parties
    # and contributors
    db.modify_table(
        const.IMAGE_TABLE,
        [(None, 'party_rowid', 'INTEGER', None)],
        shortname='image',
        extern_tables=[const.PARTY_TABLE, const.CONTRIBUTOR_TABLE],
        # TODO: add in many to 1 attribute mapping
    )

    # db.add_table(const.PARTY_CONTRIB_RELATION_TABLE, (
    #    ('party_contrib_relation_rowid',               'INTEGER PRIMARY KEY'),
    #    ('party_rowid',                                'INTEGER NOT NULL'),
    #    ('contributor_rowid',                          'INTEGER NOT NULL'),
    # ),
    #    superkeys=[('party_rowid', 'contributor_rowid')],
    #    relates=(const.PARTY_TABLE, const.CONTRIBUTOR_TABLE),
    #    docstr='''
    #    Relates parties and contributors
    #    ''',
    # )
    db.add_table(
        const.ANNOTMATCH_TABLE,
        (
            ('annotmatch_rowid', 'INTEGER PRIMARY KEY'),
            (ANNOT_ROWID1, 'INTEGER NOT NULL'),
            (ANNOT_ROWID2, 'INTEGER NOT NULL'),
            ('annotmatch_truth', 'INTEGER DEFAULT 2'),
            ('annotmatch_confidence', 'REAL DEFAULT 0'),
        ),
        superkeys=[('annot_rowid1', 'annot_rowid2',)],
        relates=(const.ANNOTATION_TABLE, const.ANNOTATION_TABLE),
        # shortname='annotmatch',
        docstr="""
        Sparsely stores explicit matching / not matching information. This
        serves as marking weather or not an annotation pair has been reviewed.
        """,
    )

    # add metadata props
    db.modify_table(
        'encounter_image_relationship',
        shortname='egr',
        relates=(const.IMAGE_TABLE, 'encounters'),
    )

    db.modify_table(
        const.AL_RELATION_TABLE,
        shortname='alr',
        relates=(const.ANNOTATION_TABLE, const.LBLANNOT_TABLE),
    )

    db.modify_table(
        const.GL_RELATION_TABLE,
        shortname='glr',
        relates=(const.IMAGE_TABLE, const.LBLIMAGE_TABLE),
    )

    db.modify_table(
        const.ANNOTATION_TABLE, shortname='annot',
    )


def update_1_3_7(db, ibs=None):
    # Part of the dependsmap property might be inferred, but at least the keys and tables are needed.
    db.modify_table(
        const.ANNOTATION_TABLE,
        extern_tables=[const.NAME_TABLE, const.SPECIES_TABLE, const.IMAGE_TABLE],
        superkeys=[(ANNOT_UUID,), (ANNOT_VISUAL_UUID,)],
        dependsmap={
            IMAGE_ROWID: (const.IMAGE_TABLE, (IMAGE_ROWID,), (IMAGE_UUID,)),
            NAME_ROWID: (const.NAME_TABLE, (NAME_ROWID,), (NAME_TEXT,)),
            SPECIES_ROWID: (const.SPECIES_TABLE, (SPECIES_ROWID,), (SPECIES_TEXT,)),
            ANNOT_PARENT_ROWID: (
                const.ANNOTATION_TABLE,
                (ANNOT_ROWID,),
                (ANNOT_VISUAL_UUID,),
            ),
        },
    )
    db.modify_table(
        const.ANNOTMATCH_TABLE,
        dependsmap={
            ANNOT_ROWID1: (const.ANNOTATION_TABLE, (ANNOT_ROWID,), (ANNOT_VISUAL_UUID,),),
            ANNOT_ROWID2: (const.ANNOTATION_TABLE, (ANNOT_ROWID,), (ANNOT_VISUAL_UUID,),),
        },
    )
    db.modify_table(
        const.IMAGE_TABLE,
        dependsmap={
            'party_rowid': (const.PARTY_TABLE, ('party_rowid',), (PARTY_TAG,)),
            'contributor_rowid': (
                const.CONTRIBUTOR_TABLE,
                ('contributor_rowid',),
                ('contributor_tag',),
            ),
        },
    )
    db.modify_table(
        const.CONFIG_TABLE,
        dependsmap={
            'contributor_rowid': (
                const.CONTRIBUTOR_TABLE,
                ('contributor_rowid',),
                ('contributor_tag',),
            ),
        },
    )

    db.modify_table(
        'encounters',
        dependsmap={
            'config_rowid': (
                const.CONFIG_TABLE,
                ('config_rowid',),
                ('contributor_rowid', CONFIG_SUFFIX,),
            ),
        },
    )

    db.modify_table(
        'encounter_image_relationship',
        dependsmap={
            'image_rowid': (const.IMAGE_TABLE, (IMAGE_ROWID,), (IMAGE_UUID,)),
            'encounter_rowid': ('encounters', ('encounter_rowid',), ('encounter_text',),),
        },
    )


def update_1_3_8(db, ibs=None):
    # Encounters only care about their text again as a uuid We are removing
    # config_rowid from encounters. Thus the dependency is not encoded
    db.modify_table(
        'encounters', superkeys=[('encounter_text',)],
    )


def update_1_3_9(db, ibs=None):
    # Remove contributors from configs
    db.modify_table(const.CONFIG_TABLE, superkeys=[(CONFIG_SUFFIX,)])
    # Add primary superkey to annotations table
    db.modify_table(
        const.ANNOTATION_TABLE,
        primary_superkey=('annot_visual_uuid',),
        docstr="""
        Mainly used to store the geometry of the annotation within its parent
        image The one-to-many relationship between images and annotations is
        encoded here
        """,
    )


def update_1_4_0(db, ibs=None):
    # Remove contributors from configs
    db.modify_table(
        'encounter_image_relationship',
        superkeys=[(IMAGE_ROWID, 'encounter_rowid',)],
        # superkeys=[(CONFIG_SUFFIX,)]
    )


def update_1_4_1(db, ibs=None):
    db.modify_table(
        'encounters',
        dependsmap={
            'config_rowid': (const.CONFIG_TABLE, ('config_rowid',), (CONFIG_SUFFIX,)),
        },
    )


def update_1_4_2(db, ibs=None):
    db.modify_table(
        const.ANNOTATION_TABLE,
        [
            (None, 'contributor_rowid', 'INTEGER', None),
            (None, 'annot_age_months_est_min', 'INTEGER DEFAULT -1', None),
            (None, 'annot_age_months_est_max', 'INTEGER DEFAULT -1', None),
        ],
        # HACK: Need a way to update the dependsmap without blowing the old one away
        # Also need to not overspecify information. colname to tablename should be fine.
        # we can have the extern colname be optional. superkey is definiately not needed
        # dependsmap=db.get_metadata_val(const.ANNOTATION_TABLE + '_dependsmap', eval_=True) +
        # {}
        dependsmap={
            IMAGE_ROWID: (const.IMAGE_TABLE, (IMAGE_ROWID,), (IMAGE_UUID,)),
            NAME_ROWID: (const.NAME_TABLE, (NAME_ROWID,), (NAME_TEXT,)),
            SPECIES_ROWID: (const.SPECIES_TABLE, (SPECIES_ROWID,), (SPECIES_TEXT,)),
            ANNOT_PARENT_ROWID: (
                const.ANNOTATION_TABLE,
                (ANNOT_ROWID,),
                (ANNOT_VISUAL_UUID,),
            ),
            'contributor_rowid': (const.CONTRIBUTOR_TABLE, None, None),
        },
    )

    db.modify_table(const.NAME_TABLE, [(None, 'name_sex', 'INTEGER DEFAULT -1', None)])


def update_1_4_3(db, ibs=None):
    db.modify_table(
        const.ANNOTATION_TABLE,
        [
            (None, 'annot_is_occluded', 'INTEGER', None),
            (None, 'annot_is_shadowed', 'INTEGER', None),
            (None, 'annot_is_washedout', 'INTEGER', None),
            (None, 'annot_is_blury', 'INTEGER', None),
            (None, 'annot_is_novelpose', 'INTEGER', None),
            (None, 'annot_is_commonpose', 'INTEGER', None),
        ],
    )

    db.modify_table(
        const.ANNOTMATCH_TABLE,
        [
            (None, 'annotmatch_is_hard', 'INTEGER', None),
            (None, 'annotmatch_is_scenerymatch', 'INTEGER', None),
            (None, 'annotmatch_is_photobomb', 'INTEGER', None),
            (None, 'annotmatch_is_nondistinct', 'INTEGER', None),
        ],
    )


def update_1_4_4(db, ibs=None):
    db.add_table(
        const.ANNOTGROUP_TABLE,
        (
            (ANNOTGROUP_ROWID, 'INTEGER PRIMARY KEY'),
            ('annotgroup_uuid', 'UUID NOT NULL'),
            ('annotgroup_text', 'TEXT NOT NULL'),
            ('annotgroup_note', 'TEXT NOT NULL'),
        ),
        superkeys=[('annotgroup_text',)],
        docstr="""
        List of all annotation groups (annotgroups)""",
    )

    db.add_table(
        const.GA_RELATION_TABLE,
        (
            ('gar_rowid', 'INTEGER PRIMARY KEY'),
            (ANNOTGROUP_ROWID, 'INTEGER NOT NULL'),
            (ANNOT_ROWID, 'INTEGER'),
        ),
        superkeys=[(ANNOTGROUP_ROWID, ANNOT_ROWID)],
        docstr="""
        Relationship between annotgroups and annots (many to many mapping) the
        many-to-many relationship between annots and annotgroups is encoded here
        annotgroup_annotation_relationship stands for annotgroup-annotation-pairs.""",
    )


def update_1_4_5(db, ibs=None):
    db.modify_table(
        const.ANNOTMATCH_TABLE, [(None, 'annotmatch_note', 'TEXT', None)],
    )


def update_1_4_6(db, ibs=None):
    # Maybe we want the notation of each annotation having having a set of
    # classes with probabilities (c, p). Or an annotation label with a
    # confidence.
    db.modify_table(
        const.ANNOTATION_TABLE,
        [
            # HACK: add a column for parsable tags, this should later be
            # replaced with a tag table and a tag-annot relation table
            (None, 'annot_tags', 'TEXT', None),
            # Remove these columns
            ('annot_is_occluded', None, None, None),
            ('annot_is_shadowed', None, None, None),
            ('annot_is_washedout', None, None, None),
            ('annot_is_blury', None, None, None),
            ('annot_is_novelpose', None, None, None),
            ('annot_is_commonpose', None, None, None),
        ],
    )
    db.modify_table(
        const.ANNOTMATCH_TABLE,
        [
            (None, 'annotmatch_reviewed', 'INTEGER', None),
            (None, 'annotmatch_reviewer', 'TEXT', None),
        ],
    )


def update_1_4_7(db, ibs=None):
    db.modify_table(const.IMAGE_TABLE, ((4, 'image_uri_original', 'TEXT', None),))


def post_1_4_7(db, ibs=None):
    if ibs is not None:
        gid_list = ibs._get_all_gids()
        image_uri_list = ibs.get_image_uris(gid_list)
        ibs.set_image_uris_original(gid_list, image_uri_list, overwrite=True)

    db.modify_table(
        const.IMAGE_TABLE,
        [
            # change type of image_uri_original
            ('image_uri_original', '', 'TEXT NOT NULL', None),
        ],
    )


def pre_1_4_8(db, ibs=None):
    """
    Args:
        ibs (wbia.IBEISController):
    """
    if ibs is not None:
        from wbia import tag_funcs

        annotmatch_rowids = ibs._get_all_annotmatch_rowids()
        id_iter = annotmatch_rowids
        annotmatch_is_photobomb_list = ibs.db.get(
            const.ANNOTMATCH_TABLE,
            ('annotmatch_is_photobomb',),
            id_iter,
            id_colname='rowid',
        )
        annotmatch_is_nondistinct = ibs.db.get(
            const.ANNOTMATCH_TABLE,
            ('annotmatch_is_nondistinct',),
            id_iter,
            id_colname='rowid',
        )
        annotmatch_is_hard = ibs.db.get(
            const.ANNOTMATCH_TABLE, ('annotmatch_is_hard',), id_iter, id_colname='rowid'
        )
        annotmatch_is_scenerymatch = ibs.db.get(
            const.ANNOTMATCH_TABLE,
            ('annotmatch_is_scenerymatch',),
            id_iter,
            id_colname='rowid',
        )

        # ibs.get_annotmatch_note(annotmatch_rowids)
        annotmatch_note_list = ibs.db.get(
            const.ANNOTMATCH_TABLE,
            ('annotmatch_note',),
            annotmatch_rowids,
            id_colname='rowid',
        )
        new_notes_list = annotmatch_note_list
        new_notes_list = tag_funcs.set_textformat_tag_flags(
            'photobomb', new_notes_list, annotmatch_is_photobomb_list
        )
        new_notes_list = tag_funcs.set_textformat_tag_flags(
            'nondistinct', new_notes_list, annotmatch_is_nondistinct
        )
        new_notes_list = tag_funcs.set_textformat_tag_flags(
            'hard', new_notes_list, annotmatch_is_hard
        )
        new_notes_list = tag_funcs.set_textformat_tag_flags(
            'scenerymatch', new_notes_list, annotmatch_is_scenerymatch
        )

        ibs.db.set(
            const.ANNOTMATCH_TABLE,
            ('annotmatch_note',),
            new_notes_list,
            annotmatch_rowids,
        )


def update_1_4_8(db, ibs=None):
    """
    change notes to tag_text_data
    add configuration that made the match
    add the score of the match
    add concept of: DEFINIATELY MATCHES, DOES NOT MATCH, CAN NOT DECIDE

    Probably want a separate table for the config_rowid matching results
    because the primary key needs to be (config_rowid, aid1, aid2) OR just
    (config_rowid, annotmatch_rowid)
    """
    db.modify_table(
        const.ANNOTATION_TABLE,
        [
            # HACK: add a column for parsable tags, this should later be
            # replaced with a tag table and a tag-annot relation table
            ('annot_tags', 'annot_tag_text', 'TEXT', None),
        ],
    )
    db.modify_table(
        const.ANNOTMATCH_TABLE,
        [
            ('annotmatch_note', 'annotmatch_tag_text', 'TEXT', None),
            (None, 'annotmatch_posixtime_modified', 'INTEGER', None),
            (None, 'annotmatch_pairwise_prob', 'REAL', None),
            (None, 'config_hashid', 'TEXT', None),
            # Remove explicit case tags in favor of consistency
            ('annotmatch_is_photobomb', None, None, None),
            ('annotmatch_is_nondistinct', None, None, None),
            ('annotmatch_is_hard', None, None, None),
            ('annotmatch_is_scenerymatch', None, None, None),
        ],
    )


def pre_1_4_9(db, ibs=None):
    if ibs is not None:
        remapping_dict = {
            'frogs': 'frog',
            'giraffe': 'giraffe_reticulated',
            'seals_spotted': 'seal_spotted',
            'seals_saimma_ringed': 'seal_saimma_ringed',
        }
        from os.path import join

        species_rowid_list = ibs._get_all_species_rowids()
        species_text_list = ibs.get_species_texts(species_rowid_list)
        for rowid, text in zip(species_rowid_list, species_text_list):
            if text in remapping_dict:
                # Update record for reticulated giraffe
                ibs._set_species_texts([rowid], [remapping_dict[text]])

                # Delete obsolete cPkl file on disk
                cPlk_path = join(ibs.get_dbdir(), '%s.cPkl' % (text,))
                ut.delete(cPlk_path)

                # Recompute all effected annotation's semantic UUIDs
                aid_list = ibs._get_all_aids()
                annot_species_rowid_list = ibs.get_annot_species_rowids(aid_list)
                flag_list = [
                    annot_species_rowid == rowid
                    for annot_species_rowid in annot_species_rowid_list
                ]
                aid_list_ = ut.filter_items(aid_list, flag_list)
                ibs.update_annot_semantic_uuids(aid_list_)


def update_1_4_9(db, ibs=None):
    db.modify_table(const.SPECIES_TABLE, ((3, 'species_code', 'TEXT', None),))

    db.modify_table(const.SPECIES_TABLE, ((3, 'species_nice', 'TEXT', None),))

    db.modify_table(
        const.SPECIES_TABLE,
        ((None, 'species_toggle_enabled', 'INTEGER DEFAULT 1', None),),
    )


def post_1_4_9(db, ibs=None):
    if ibs is not None:
        ibs._clean_species()
    db.modify_table(
        const.SPECIES_TABLE,
        [
            # change type of species_nice
            ('species_nice', '', 'TEXT NOT NULL', None),
            ('species_code', '', 'TEXT NOT NULL', None),
        ],
    )


def update_1_5_0(db, ibs=None):
    # Rename encounters to imagesets
    db.rename_table('encounters', 'imagesets')
    db.rename_table('encounter_image_relationship', 'imageset_image_relationship')
    db.modify_table(
        'imagesets',
        [
            ('encounter_rowid', 'imageset_rowid', 'INTEGER PRIMARY KEY', None),
            ('encounter_uuid', 'imageset_uuid', 'UUID NOT NULL', None),
            ('encounter_text', 'imageset_text', 'TEXT NOT NULL', None),
            ('encounter_note', 'imageset_note', 'TEXT NOT NULL', None),
            ('encounter_start_time_posix', 'imageset_start_time_posix', 'INTEGER', None,),
            ('encounter_end_time_posix', 'imageset_end_time_posix', 'INTEGER', None),
            ('encounter_gps_lat', 'imageset_gps_lat', 'INTEGER', None),
            ('encounter_gps_lon', 'imageset_gps_lon', 'INTEGER', None),
            (
                'encounter_processed_flag',
                'imageset_processed_flag',
                'INTEGER DEFAULT 0',
                None,
            ),
            (
                'encounter_shipped_flag',
                'imageset_shipped_flag',
                'INTEGER DEFAULT 0',
                None,
            ),
            ('encounter_smart_xml_fname', 'imageset_smart_xml_fname', 'TEXT', None),
            (
                'encounter_smart_waypoint_id',
                'imageset_smart_waypoint_id',
                'INTEGER',
                None,
            ),
        ],
        docstr="""
        List of all imagesets. This used to be called the encounter table.
        It represents a group of potentially many individuals seen in a
        specific place at a specific time.
        """,
        superkeys=[('imageset_text',)],
    )
    db.modify_table(
        'imageset_image_relationship',
        [
            ('egr_rowid', 'gsgr_rowid', 'INTEGER PRIMARY KEY', None),
            ('encounter_rowid', 'imageset_rowid', 'INTEGER', None),
        ],
        docstr="""
        Relationship between imagesets and images (many to many mapping) the
        many-to-many relationship between images and imagesets is encoded
        here imageset_image_relationship stands for imageset-image-pairs.
        """,
        superkeys=[('image_rowid', 'imageset_rowid')],
        relates=('images', 'imagesets'),
        shortname='gsgr',
        dependsmap={
            'imageset_rowid': ('imagesets', ('imageset_rowid',), ('imageset_text',)),
            'image_rowid': ('images', ('image_rowid',), ('image_uuid',)),
        },
    )


def update_1_5_1(db, ibs=None):
    # Rename encounters to imagesets
    db.modify_table(
        const.IMAGE_TABLE, superkeys=[(IMAGE_UUID,)],
    )
    db.modify_table(
        const.ANNOTMATCH_TABLE, superkeys=[('annot_rowid1', 'annot_rowid2',)],
    )


def update_1_5_2(db, ibs=None):
    # Add orientation to images
    db.modify_table(
        const.IMAGE_TABLE, ((12, 'image_orientation', 'INTEGER DEFAULT 0', None),)
    )


def post_1_5_2(db, ibs=None, verbose=False):
    if ibs is not None:
        from PIL import Image  # NOQA
        from wbia.algo.preproc.preproc_image import parse_exif
        from wbia.scripts import fix_annotation_orientation_issue as faoi
        from os.path import exists

        def _parse_orient(gpath):
            if verbose:
                print('[db_update (1.5.2)]     Parsing: %r' % (gpath,))
            pil_img = Image.open(gpath, 'r')  # NOQA
            time, lat, lon, orient = parse_exif(pil_img)  # Read exif tags
            return orient

        # Get images without orientations and add to the database
        gid_list_all = ibs.get_valid_gids()
        gpath_list = ibs.get_image_paths(gid_list_all)
        valid_list = [exists(gpath) for gpath in gpath_list]
        gid_list = ut.filter_items(gid_list_all, valid_list)

        orient_list = ibs.get_image_orientation(gid_list)
        zipped = zip(gid_list, orient_list)
        gid_list_ = [gid for gid, orient in zipped if orient in [0, None]]
        args = (len(gid_list_), len(gid_list_all), valid_list.count(False))
        print(
            '[db_update (1.5.2)] Parsing Exif orientations for %d / %d images (skipping %d)'
            % args
        )
        gpath_list_ = ibs.get_image_paths(gid_list_)
        orient_list_ = [_parse_orient(gpath) for gpath in gpath_list_]
        ibs._set_image_orientation(gid_list_, orient_list_)
        faoi.fix_annotation_orientation(ibs)


def update_1_5_3(db, ibs=None):
    # Add reviewed flag to annotations
    db.modify_table(
        const.ANNOTATION_TABLE,
        ((13, 'annot_toggle_reviewed', 'INTEGER DEFAULT 0', None),),
    )


def update_1_5_4(db, ibs=None):
    # Add reviewed flag to annotations
    db.modify_table(
        const.ANNOTATION_TABLE,
        ((14, 'annot_toggle_multiple', 'INTEGER DEFAULT NULL', None),),
    )


def update_1_5_5(db, ibs=None):
    # Remove the config table
    db.drop_table('configs')
    db.modify_table(
        'image_lblimage_relationship',
        drop_columns=['config_rowid'],
        superkeys=[('image_rowid', 'lblimage_rowid')],
    )
    db.modify_table(
        'annotation_lblannot_relationship',
        drop_columns=['config_rowid'],
        superkeys=[('annot_rowid', 'lblannot_rowid')],
    )
    db.modify_table('imagesets', drop_columns=['config_rowid'], dependsmap={})


def update_1_6_0(db, ibs=None):
    db.modify_table(const.IMAGE_TABLE, ((None, 'image_metadata_json', 'TEXT', None),))

    db.modify_table(
        const.ANNOTATION_TABLE, ((None, 'annot_metadata_json', 'TEXT', None),)
    )


def update_1_6_1(db, ibs=None):
    # if ibs is not None:
    #     assert ibs.get_dbname() in ['PZ_PB_RF_TRAIN', 'WWF_Lynx', 'EWT_Cheetahs'], (
    #         'this is a hacked state. to fix bug where EVIDENCE_DECISION.UNKNOWN was 2')
    db.modify_table(
        'annotmatch',
        colmap_list=[('annotmatch_truth', 'annotmatch_truth', 'INTEGER', None)],
    )


def post_1_6_1(db, ibs=None, verbose=False):
    # Find annotmatch rowids that have an old value of 2
    ams = db.get_where_eq(
        'annotmatch',
        colnames=('annotmatch_rowid',),
        params_iter=[(2,)],
        unpack_scalars=False,
        where_colnames=('annotmatch_truth',),
    )[0]
    print('Setting %d old unknown values to NULL' % (len(ams)))
    # if ibs is not None:
    #     assert ibs.get_dbname() in ['PZ_PB_RF_TRAIN', 'WWF_Lynx', 'EWT_Cheetahs'], (
    #         'this is a hacked state. to fix bug where EVIDENCE_DECISION.UNKNOWN was 2')
    # if False:
    db.set('annotmatch', ('annotmatch_truth',), [None] * len(ams), ams)


def update_1_6_2(db, ibs=None):
    # All confidences thusfar have had no meaning. Change the column to an integer
    # to user the USER_CONFIDENCE_CODES and reset all values to None
    db.modify_table(
        'annotmatch',
        colmap_list=[
            (
                'annotmatch_confidence',
                'annotmatch_confidence',
                'INTEGER',
                lambda val: None,
            )
        ],
    )


def update_1_6_3(db, ibs=None):
    db.modify_table(
        const.IMAGESET_TABLE, ((None, 'imageset_metadata_json', 'TEXT', None),)
    )

    db.modify_table(const.NAME_TABLE, ((None, 'name_metadata_json', 'TEXT', None),))


def update_1_6_4(db, ibs=None):
    db.modify_table(
        const.ANNOTATION_TABLE,
        (
            (12, 'annot_viewpoint', 'TEXT', None),
            (16, 'annot_toggle_interest', 'INTEGER DEFAULT NULL', None),
        ),
    )

    db.add_table(
        const.PART_TABLE,
        (
            (PART_ROWID, 'INTEGER PRIMARY KEY'),
            (PART_UUID, 'UUID NOT NULL'),
            (ANNOT_ROWID, 'INTEGER NOT NULL'),
            ('part_xtl', 'INTEGER NOT NULL'),
            ('part_ytl', 'INTEGER NOT NULL'),
            ('part_width', 'INTEGER NOT NULL'),
            ('part_height', 'INTEGER NOT NULL'),
            ('part_theta', 'REAL DEFAULT 0.0'),
            ('part_num_verts', 'INTEGER NOT NULL'),
            ('part_verts', 'TEXT'),
            ('part_viewpoint', 'TEXT'),
            ('part_detect_confidence', 'REAL DEFAULT -1.0'),
            ('part_toggle_reviewed', 'INTEGER DEFAULT 0'),
            ('part_quality', 'INTEGER'),
            ('part_type', 'TEXT'),
            ('part_note', 'TEXT'),
            ('part_tag_text', 'TEXT'),
        ),
        docstr="""
        Mainly used to store the geometry of the annotation parts within its parent
        annotation. The one-to-many relationship between annotations and parts is
        encoded here
        """,
        superkeys=[(PART_UUID,)],
        shortname='part',
        extern_tables=[const.ANNOTATION_TABLE],
        dependsmap={
            ANNOT_ROWID: (const.ANNOTATION_TABLE, (ANNOT_ROWID,), (ANNOT_VISUAL_UUID,)),
        },
    )


def post_1_6_4(db, ibs=None):
    if ibs is not None:
        from wbia.other import ibsfuncs

        aids = ibs.get_valid_aids(is_staged=None)
        # Get old yaw values
        yaws = db.get(const.ANNOTATION_TABLE, (ANNOT_YAW,), aids)
        yaws = [yaw if yaw is not None and yaw >= 0.0 else None for yaw in yaws]
        # Convert into viewpoint text
        viewpoint_list = ibsfuncs.get_yaw_viewtexts(yaws)
        db.set(const.ANNOTATION_TABLE, ('annot_viewpoint',), viewpoint_list, id_iter=aids)


def update_1_6_5(db, ibs=None):
    db.modify_table(
        const.IMAGE_TABLE,
        ((15, 'image_toggle_cameratrap', 'INTEGER DEFAULT NULL', None),),
    )


def update_1_6_6(db, ibs=None):
    db.modify_table(
        const.ANNOTMATCH_TABLE, ((None, 'annotmatch_count', 'INTEGER', None),)
    )


def update_1_6_7(db, ibs=None):
    db.modify_table(
        const.ANNOTMATCH_TABLE,
        (
            ('annotmatch_pairwise_prob', None, None, None),
            ('config_hashid', None, None, None),
        ),
    )


def update_1_6_8(db, ibs=None):
    db.modify_table(
        const.ANNOTMATCH_TABLE,
        (
            # Rename truth to visual decision
            ('annotmatch_truth', 'annotmatch_evidence_decision', 'INTEGER', None),
            # Add new meta_decision
            (None, 'annotmatch_meta_decision', 'INTEGER', None),
            # remove the reviewed flag. This is basically stored via userid/count
            ('annotmatch_reviewed', None, None, None),
        ),
    )


def update_1_6_9(db, ibs=None):
    db.modify_table(
        const.ANNOTATION_TABLE,
        (
            # Add column to mirror wildbook encounters
            # FIXME: make this an int that points to a different "static_encounter"
            # table rowid
            (None, 'annot_static_encounter', 'TEXT', None),
        ),
    )


# TODO: YAW TO VIEWPOINT CODE DATABASE CHANGE
def update_1_7_0(db, ibs=None):
    """
    Ignore:
        import wbia
        ibs = wbia.opendb('testdb1')
        ibs.annots().yaws
        ibs.annots().viewpoint_int
        codes = ibs.annots().viewpoint_code
        texts = ['unknown' if y is None else y for y in ibs.annots().yaw_texts]
        assert codes == texts
    """
    db.modify_table(
        const.ANNOTATION_TABLE,
        (
            # Add code to represent an arbitrary viewpoint
            # We are just depricating yaw for now.
            (None, 'annot_viewpoint_int', 'INTEGER', None),
        ),
    )


def post_1_7_0(db, ibs=None):
    if ibs is not None:
        from wbia.other import ibsfuncs

        aids = db.get_all_rowids(const.ANNOTATION_TABLE)

        # Get old yaw values
        yaws = db.get(const.ANNOTATION_TABLE, (ANNOT_YAW,), aids)
        yaws = [yaw if yaw is not None and yaw >= 0.0 else None for yaw in yaws]
        # Convert them into yaw/view codes
        view_codes = ibsfuncs.get_yaw_viewtexts(yaws)

        # Convert the codes into integers
        VIEW = ibs.const.VIEW
        UNKNOWN_CODE = VIEW.INT_TO_CODE[VIEW.UNKNOWN]
        view_codes = [UNKNOWN_CODE if y is None else y for y in view_codes]
        view_ints = ut.dict_take(ibs.const.VIEW.CODE_TO_INT, view_codes)

        ibs.db.set(
            const.ANNOTATION_TABLE, ('annot_viewpoint_int',), view_ints, id_iter=aids
        )

        # Moved here from 1.3.4 because the way it is calculated changed
        ibs.update_annot_visual_uuids(ibs.get_valid_aids(is_staged=None))


def update_1_7_1(db, ibs=None):
    db.modify_table(
        const.PART_TABLE,
        [],
        dependsmap={
            ANNOT_ROWID: (const.ANNOTATION_TABLE, (ANNOT_ROWID,), (ANNOT_UUID,)),
        },
    )


def update_1_8_0(db, ibs=None):
    db.modify_table(
        const.ANNOTATION_TABLE,
        (
            (None, 'annot_staged_flag', 'INTEGER DEFAULT 0', None),
            (None, 'annot_staged_uuid', 'UUID', None),
            (None, 'annot_staged_user_identity', 'TEXT', None),
            (None, 'annot_staged_metadata_json', 'TEXT', None),
        ),
        superkeys=[(ANNOT_UUID,), (ANNOT_VISUAL_UUID, ANNOT_STAGED_UUID,)],
        primary_superkey=(ANNOT_UUID,),
    )

    db.modify_table(
        const.PART_TABLE,
        (
            (None, 'part_staged_flag', 'INTEGER DEFAULT 0', None),
            (None, 'part_staged_uuid', 'UUID', None),
            (None, 'part_staged_user_identity', 'TEXT', None),
            (None, 'part_staged_metadata_json', 'TEXT', None),
        ),
        superkeys=[(PART_UUID,)],
    )


def update_1_8_1(db, ibs=None):
    db.modify_table(
        const.PART_TABLE,
        (
            (None, 'part_metadata_json', 'TEXT', None),
            (None, 'part_contour_json', 'TEXT', None),
        ),
    )


def update_1_8_2(db, ibs=None):
    db.modify_table(
        const.ANNOTATION_TABLE,
        ((17, 'annot_toggle_canonical', 'INTEGER DEFAULT NULL', None),),
    )


def update_1_8_3(db, ibs=None):
    db.modify_table(
        const.IMAGESET_TABLE,
        ((3, 'imageset_occurrence_flag', 'INTEGER DEFAULT 0', None),),
    )


def update_2_0_0(db, ibs=None):
    # This update is simply a status marker for the software state
    pass


# ========================
# Valid Versions & Mapping
# ========================

# TODO: do we save a backup with the older version number in the file name?


base = const.BASE_DATABASE_VERSION
VALID_VERSIONS = ut.odict(
    [
        # version:   (Pre-Update Function,  Update Function,    Post-Update Function)
        (base, (None, None, None)),
        ('1.0.0', (None, update_1_0_0, post_1_0_0)),
        ('1.0.1', (None, update_1_0_1, None)),
        ('1.0.2', (None, update_1_0_2, None)),
        ('1.1.0', (None, update_1_1_0, None)),
        ('1.1.1', (None, update_1_1_1, None)),
        ('1.2.0', (None, update_1_2_0, post_1_2_0)),
        ('1.2.1', (None, update_1_2_1, post_1_2_1)),
        ('1.3.0', (None, update_1_3_0, None)),
        ('1.3.1', (pre_1_3_1, update_1_3_1, None)),
        ('1.3.2', (None, update_1_3_2, None)),
        ('1.3.3', (None, update_1_3_3, None)),
        ('1.3.4', (None, update_1_3_4, post_1_3_4)),
        ('1.3.5', (None, update_1_3_5, None)),
        ('1.3.6', (None, update_1_3_6, None)),
        ('1.3.7', (None, update_1_3_7, None)),
        ('1.3.8', (None, update_1_3_8, None)),
        ('1.3.9', (None, update_1_3_9, None)),
        ('1.4.0', (None, update_1_4_0, None)),
        ('1.4.1', (None, update_1_4_1, None)),
        ('1.4.2', (None, update_1_4_2, None)),
        ('1.4.3', (None, update_1_4_3, None)),
        ('1.4.4', (None, update_1_4_4, None)),
        ('1.4.5', (None, update_1_4_5, None)),
        ('1.4.6', (None, update_1_4_6, None)),
        ('1.4.7', (None, update_1_4_7, post_1_4_7)),
        ('1.4.8', (pre_1_4_8, update_1_4_8, None)),
        ('1.4.9', (pre_1_4_9, update_1_4_9, post_1_4_9)),
        ('1.5.0', (None, update_1_5_0, None)),
        ('1.5.1', (None, update_1_5_1, None)),
        ('1.5.2', (None, update_1_5_2, post_1_5_2)),
        ('1.5.3', (None, update_1_5_3, None)),
        ('1.5.4', (None, update_1_5_4, None)),
        ('1.5.5', (None, update_1_5_5, None)),
        ('1.6.0', (None, update_1_6_0, None)),
        ('1.6.1', (None, update_1_6_1, post_1_6_1)),
        ('1.6.2', (None, update_1_6_2, None)),
        ('1.6.3', (None, update_1_6_3, None)),
        ('1.6.4', (None, update_1_6_4, post_1_6_4)),
        ('1.6.5', (None, update_1_6_5, None)),
        ('1.6.6', (None, update_1_6_6, None)),
        ('1.6.7', (None, update_1_6_7, None)),
        ('1.6.8', (None, update_1_6_8, None)),
        ('1.6.9', (None, update_1_6_9, None)),
        ('1.7.0', (None, update_1_7_0, post_1_7_0)),
        ('1.7.1', (None, update_1_7_1, None)),
        ('1.8.0', (None, update_1_8_0, None)),
        ('1.8.1', (None, update_1_8_1, None)),
        ('1.8.2', (None, update_1_8_2, None)),
        ('1.8.3', (None, update_1_8_3, None)),
        ('2.0.0', (None, update_2_0_0, None)),
    ]
)
"""
SeeAlso:
    When updating versions need to test and modify in
    IBEISController._init_sqldbcore
"""


LEGACY_UPDATE_FUNCTIONS = [
    ('1.4.1', _sql_helpers.fix_metadata_consistency),
]


def __test_db_version_table_constraints():
    """
    test for updating from version x to version y

    There is a problem where the contributor_table superkey is not in
    PZ_Master0 and I don't know why. Perhaps it was just a fluke, and it will
    be ensured from now on.

    Here is the hacky fix script:
        assert 'contributors_superkeys' not in ut.get_list_column(ibs.db.get_metadata_items(), 0)
        sorted(ibs.db.get_metadata_items())
        # So weird that the constraint was set, but not the superkeys
        constraint_str = ibs.db.get_metadata_val('contributors_constraint')
        parse_result = parse.parse('CONSTRAINT superkey UNIQUE ({superkey})', constraint_str)
        superkey = parse_result['superkey']
        assert superkey == 'contributor_tag'
        assert None is ibs.db.get_metadata_val('contributors_superkey')
        ibs.db.set_metadata_val('contributors_superkeys', "[('contributor_tag',)]")

        # Made a mistake
        print(ibs.db.get_table_csv_header('metadata'))
        badrowid = ibs.db.get_rowid_from_superkey('metadata', [('contributors_superkey',)], ('metadata_key',))
        assert len(badrowid) == 1
        ibs.db.delete('metadata', [badrowid[0]])

    TODO: make a script that generates an empty database at version X

    """
    import wbia

    tmpdir = ut.ensuredir('tmpsqltestdir')
    ut.delete(tmpdir)
    tmpdir = ut.ensuredir('tmpsqltestdir')
    tmpdir3 = ut.ensuredir('tmpsqltestdir3')
    ut.delete(tmpdir3)
    tmpdir3 = ut.ensuredir('tmpsqltestdir3')
    # Should not show contributor table
    ibs1 = wbia.opendb(dbdir=tmpdir, request_dbversion='1.0.0', use_cache=False)
    ibs1.db.print_schema()
    assert 'contributors' not in ibs1.db.get_table_names()

    ibs2 = wbia.opendb(dbdir=tmpdir, request_dbversion='1.0.3', use_cache=False)
    ibs2.db.print_schema()
    assert 'contributors' in ibs2.db.get_table_names()
    print(ibs2.db.get_schema_current_autogeneration_str('foo'))

    assert 'contributors_superkeys' in ut.get_list_column(ibs2.db.get_metadata_items(), 0)

    ibs3 = wbia.opendb(dbdir=tmpdir3, use_cache=False)
    ibs3.db.print_schema()
    assert 'contributors' in ibs1.db.get_table_names()

    # wbia.control.IBEISControl.__ALL_CONTROLLERS__

    ibs1.db.close()
    ibs2.db.close()
    ibs1.depc.close()
    ibs2.depc.close()

    del ibs1
    del ibs2


def autogen_db_schema():
    """
    autogen_db_schema

    CommandLine:
        python -m wbia.control.DB_SCHEMA --test-autogen_db_schema
        python -m wbia.control.DB_SCHEMA --test-autogen_db_schema --diff=1
        python -m wbia.control.DB_SCHEMA --test-autogen_db_schema -n=-1
        python -m wbia.control.DB_SCHEMA --test-autogen_db_schema -n=0
        python -m wbia.control.DB_SCHEMA --test-autogen_db_schema -n=1
        python -m wbia.control.DB_SCHEMA --force-incremental-db-update
        python -m wbia.control.DB_SCHEMA --test-autogen_db_schema --write
        python -m wbia.control.DB_SCHEMA --test-autogen_db_schema --force-incremental-db-update --dump-autogen-schema
        python -m wbia.control.DB_SCHEMA --test-autogen_db_schema --force-incremental-db-update


    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.DB_SCHEMA import *  # NOQA
        >>> autogen_db_schema()
    """
    from wbia.control import DB_SCHEMA
    from wbia.control import _sql_helpers

    n = ut.get_argval('-n', int, default=-1)
    schema_spec = DB_SCHEMA
    db = _sql_helpers.autogenerate_nth_schema_version(schema_spec, n=n)
    return db


def dump_schema_sql():
    """
    CommandLine:
        python -m wbia.control.DB_SCHEMA dump_schema_sql
    """
    from wbia import dtool as dt
    from wbia.control import DB_SCHEMA_CURRENT

    db = dt.SQLDatabaseController(fpath=':memory:')
    DB_SCHEMA_CURRENT.update_current(db)
    dump_str = db.dump_to_string()
    print(dump_str)

    for tablename in db.get_table_names():
        autogen_dict = db.get_table_autogen_dict(tablename)
        coldef_list = autogen_dict['coldef_list']
        str_ = db._make_add_table_sqlstr(tablename, coldef_list=coldef_list, sep='\n    ')
        print(str_)


if __name__ == '__main__':
    """
    python -m wbia.algo.preproc.preproc_chip
    python -m wbia.control.DB_SCHEMA --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()
    import utool as ut

    ut.doctest_funcs()
