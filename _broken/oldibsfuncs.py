def merge_species_databases(species_prefix):
    """ Build a merged database """
    from ibeis.control import IBEISControl
    from ibeis.dev import sysres
    print('[ibsfuncs] Merging species with prefix: %r' % species_prefix)
    ut.util_parallel.ensure_pool(warn=False)
    with ut.Indenter('    '):
        # Build / get target database
        all_db = '__ALL_' + species_prefix + '_'
        all_dbdir = sysres.db_to_dbdir(all_db, allow_newdir=True)
        ibs_target = IBEISControl.IBEISController(all_dbdir)
        # Build list of databases to merge
        species_dbdir_list = get_species_dbs(species_prefix)
        ibs_source_list = []
        for dbdir in species_dbdir_list:
            ibs_source = IBEISControl.IBEISController(dbdir)
            ibs_source_list.append(ibs_source)
    print('[ibsfuncs] Destination database: %r' % all_db)
    print('[ibsfuncs] Source databases:' +
          ut.indentjoin(species_dbdir_list, '\n *   '))
    #Merge the databases into ibs_target
    merge_databases(ibs_target, ibs_source_list)
    return ibs_target


def merge_databases(ibs_target, ibs_source_list):
    """ Merges a list of databases into a target
    This is OLD. use export_subset instead
    """
    raise AssertionError('Use transfer_subset instead')

    def merge_images(ibs_target, ibs_source):
        """ merge image helper """
        gid_list1   = ibs_source.get_valid_gids()
        uuid_list1  = ibs_source.get_image_uuids(gid_list1)
        gpath_list1 = ibs_source.get_image_paths(gid_list1)
        reviewed_list1 = ibs_source.get_image_reviewed(gid_list1)
        # Add images to target
        ibs_target.add_images(gpath_list1)
        # Merge properties
        gid_list2  = ibs_target.get_image_gids_from_uuid(uuid_list1)
        ibs_target.set_image_reviewed(gid_list2, reviewed_list1)

    def merge_annotations(ibs_target, ibs_source):
        """ merge annotations helper """
        aid_list1   = ibs_source.get_valid_aids()
        uuid_list1  = ibs_source.get_annot_uuids(aid_list1)
        # Get the images in target_db
        gid_list1   = ibs_source.get_annot_gids(aid_list1)
        bbox_list1  = ibs_source.get_annot_bboxes(aid_list1)
        theta_list1 = ibs_source.get_annot_thetas(aid_list1)
        name_list1  = ibs_source.get_annot_names(aid_list1)
        notes_list1 = ibs_source.get_annot_notes(aid_list1)

        image_uuid_list1 = ibs_source.get_image_uuids(gid_list1)
        gid_list2  = ibs_target.get_image_gids_from_uuid(image_uuid_list1)
        image_uuid_list2 = ibs_target.get_image_uuids(gid_list2)
        # Assert that the image uuids have not changed
        assert image_uuid_list1 == image_uuid_list2, 'error merging annotation image uuids'
        aid_list2 = ibs_target.add_annots(gid_list2,
                                          bbox_list1,
                                          theta_list=theta_list1,
                                          name_list=name_list1,
                                          notes_list=notes_list1)
        uuid_list2 = ibs_target.get_annot_uuids(aid_list2)
        assert uuid_list2 == uuid_list1, 'error merging annotation uuids'

    # Do the merging
    for ibs_source in ibs_source_list:
        try:
            print('Merging ' + ibs_source.get_dbname() +
                  ' into ' + ibs_target.get_dbname())
            merge_images(ibs_target, ibs_source)
            merge_annotations(ibs_target, ibs_source)
        except Exception as ex:
            ut.printex(ex, 'error merging ' + ibs_source.get_dbname() +
                          ' into ' + ibs_target.get_dbname())


# from params
parser2.add_str(('--merge-species'), help='merges all databases of given species')

if params.args.merge_species is not None:
    ibsfuncs.merge_species_databases(params.args.merge_species)


# Merge all jaguar databases into single big database
python main.py --merge-species JAG_
