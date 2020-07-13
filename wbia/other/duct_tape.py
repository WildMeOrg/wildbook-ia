# -*- coding: utf-8 -*-
# developer convenience functions for ibs
from __future__ import absolute_import, division, print_function, unicode_literals
import utool as ut
from six.moves import zip
from wbia import constants as const

(print, rrr, profile) = ut.inject2(__name__, '[duct_tape]')


def fix_compname_configs(ibs):
    """ duct tape to keep version in check """
    # ibs.MANUAL_CONFIG_SUFFIX = '_MANUAL_'  #+ ut.get_computer_name()
    # ibs.MANUAL_CONFIGID = ibs.add_config(ibs.MANUAL_CONFIG_SUFFIX)
    # We need to fix the manual config suffix to not use computer names anymore

    configid_list = ibs.get_valid_configids()
    cfgsuffix_list = ibs.get_config_suffixes(configid_list)

    ibs.MANUAL_CONFIG_SUFFIX = 'MANUAL_CONFIG'
    ibs.MANUAL_CONFIGID = ibs.add_config(ibs.MANUAL_CONFIG_SUFFIX)

    for rowid, suffix in filter(
        lambda tup: tup[1].startswith('_MANUAL_'), zip(configid_list, cfgsuffix_list)
    ):
        print('EVALUATING: %r, %r' % (rowid, suffix))
        # Fix the tables with bad config_rowids
        ibs.db.executeone(
            """
            UPDATE {AL_RELATION_TABLE}
            SET config_rowid=?
            WHERE config_rowid=?
            """.format(
                **const.__dict__
            ),
            params=(ibs.MANUAL_CONFIGID, rowid),
        )

        # Delete the bad config_suffixes
        ibs.db.executeone(
            """
            DELETE
            FROM {CONFIG_TABLE}
            WHERE config_rowid=?
            """.format(
                **const.__dict__
            ),
            params=(rowid,),
        )


def remove_database_slag(
    ibs,
    delete_empty_names=False,
    delete_empty_imagesets=False,
    delete_annotations_for_missing_images=False,
    delete_image_labels_for_missing_types=False,
    delete_annot_labels_for_missing_types=False,
    delete_chips_for_missing_annotations=False,
    delete_features_for_missing_annotations=False,
    delete_invalid_eg_relations=False,
    delete_invalid_gl_relations=False,
    delete_invalid_al_relations=True,
):
    # ZERO ORDER
    if delete_empty_names:
        ibs.delete_empty_nids()

    if delete_empty_imagesets:
        ibs.delete_empty_imgsetids()

    # FIRST ORDER
    if delete_annotations_for_missing_images:
        ibs.db.executeone(
            """
            DELETE
            FROM {ANNOTATION_TABLE}
            WHERE
                image_rowid NOT IN (SELECT rowid FROM {IMAGE_TABLE})
            """.format(
                **const.__dict__
            )
        )

    if delete_image_labels_for_missing_types:
        ibs.db.executeone(
            """
            DELETE
            FROM {LBLIMAGE_TABLE}
            WHERE
                lbltype_rowid NOT IN (SELECT rowid FROM {LBLTYPE_TABLE})
            """.format(
                **const.__dict__
            )
        )

    if delete_annot_labels_for_missing_types:
        ibs.db.executeone(
            """
            DELETE
            FROM {LBLANNOT_TABLE}
            WHERE
                lbltype_rowid NOT IN (SELECT rowid FROM {LBLTYPE_TABLE})
            """.format(
                **const.__dict__
            )
        )

    # SECOND ORDER
    if delete_chips_for_missing_annotations:
        ibs.db.executeone(
            """
            DELETE
            FROM {CHIP_TABLE}
            WHERE
                annot_rowid NOT IN (SELECT rowid FROM {ANNOTATION_TABLE})
            """.format(
                **const.__dict__
            )
        )
        # OR config_rowid NOT IN (SELECT rowid FROM {CONFIG_TABLE})

    if delete_features_for_missing_annotations:
        ibs.db.executeone(
            """
            DELETE
            FROM {FEATURE_TABLE}
            WHERE
                chip_rowid NOT IN (SELECT rowid FROM {CHIP_TABLE})
            """.format(
                **const.__dict__
            )
        )
        # OR config_rowid NOT IN (SELECT rowid FROM {CONFIG_TABLE})

    if delete_invalid_eg_relations:
        ibs.db.executeone(
            """
            DELETE
            FROM {GSG_RELATION_TABLE}
            WHERE
                image_rowid NOT IN (SELECT rowid FROM {IMAGE_TABLE}) OR
                imageset_rowid NOT IN (SELECT rowid FROM {IMAGESET_TABLE})
            """.format(
                **const.__dict__
            )
        )

    # THIRD ORDER
    if delete_invalid_gl_relations:
        ibs.db.executeone(
            """
            DELETE
            FROM {GL_RELATION_TABLE}
            WHERE
                image_rowid NOT IN (SELECT rowid FROM {IMAGE_TABLE}) OR
                lblimage_rowid NOT IN (SELECT rowid FROM {LBLIMAGE_TABLE})
            """.format(
                **const.__dict__
            )
        )
        # OR config_rowid NOT IN (SELECT rowid FROM {CONFIG_TABLE})

    if delete_invalid_al_relations:
        ibs.db.executeone(
            """
            DELETE
            FROM {AL_RELATION_TABLE}
            WHERE
                annot_rowid NOT IN (SELECT rowid FROM {ANNOTATION_TABLE}) OR
                lblannot_rowid NOT IN (SELECT rowid FROM {LBLANNOT_TABLE})
            """.format(
                **const.__dict__
            )
        )
        # OR config_rowid NOT IN (SELECT rowid FROM {CONFIG_TABLE})


def enforce_unkonwn_name_is_explicit(ibs):
    nid_list = ibs.get_valid_nids()
    text_list = ibs.get_name_texts(nid_list)
    problem_nids = [
        text for text, nid in zip(text_list, nid_list) if text == const.UNKNOWN
    ]
    unknown_aids = ibs.get_name_aids(problem_nids)
    assert len(ut.flatten(unknown_aids)) == 0
    # TODO Take unknown_aids and remove any name relationships to make unknown
    # implicit


def fix_nulled_yaws(ibs):
    aid_list = ibs.get_valid_aids()
    yaw_list = ibs.get_annot_yaws(aid_list)
    valid_list = [yaw == 0.0 for yaw in yaw_list]
    dirty_aid_list = ut.filter_items(aid_list, valid_list)
    print('[duct_tape] Nulling %d annotation yaws' % len(dirty_aid_list))
    ibs.set_annot_viewpoints(dirty_aid_list, [None] * len(dirty_aid_list))
